"""議事録の生成（LLM の呼び出しと、チャンクの結果のつなぎ合わせ）。

手順:
1. 発話を minutes.chunk_minutes 分ごと（重なり 0）のチャンクに分ける。
2. チャンクごとに Ollama へ投げ、ChunkMinutes（topics、decisions、todos、open_issues）を得る。
3. 全チャンクの結果をつなげる（同じ議題名は要点をまとめ、完全に同じ文は1つにする）。
4. つなげた結果（文字起こしの全文ではない）をもう1回 Ollama に投げ、概要（3〜5文）を得る。

どれか1つのチャンクが失敗したら、会議全体を失敗にする（議事録を書かない）。失敗したチャンクを
飛ばして書くと、その区間の内容が黙って議事録から消えるためである（「失うより止める」）。
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import duckdb

from kijun import ollama_api
from kijun.config import Config
from kijun.db import repo
from kijun.llm.chat import JsonChat, LlmError
from kijun.llm.tokens import estimate_messages_tokens
from kijun.minutes.chunk import Chunk, split_into_chunks
from kijun.minutes.models import ChunkMinutes, Overview, Todo, Topic
from kijun.minutes.prompt import build_chunk_messages, build_overview_messages
from kijun.minutes.render import minutes_path, render_markdown
from kijun.models import Utterance

NO_CONTENT_OVERVIEW = "要点として取り出せる内容がありませんでした。"
NO_UTTERANCE_OVERVIEW = "文字起こしに発話がありませんでした。"

# 決定事項が1件も無いのに、概要がこれらの語を含むときは警告する(自動では書き換えない)。
DECISION_WORDS = ("決定", "決まっ")


class MinutesError(Exception):
    """議事録を作れなかったときの例外（チャンクの失敗、概要の失敗）。"""


@dataclass
class MergedMinutes:
    """全チャンクの結果をつなげたもの（概要は含まない）。"""

    topics: list[Topic] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    todos: list[Todo] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.topics or self.decisions or self.todos or self.open_issues)


@dataclass
class MinutesDoc:
    overview: str
    merged: MergedMinutes
    warnings: list[str] = field(default_factory=list)


@dataclass
class MinutesRun:
    conversation_id: str
    model_name: str
    created_at: datetime
    chunk_count: int
    path: str
    doc: MinutesDoc


def _clean(text: str) -> str:
    return " ".join(text.split())


_IGNORED_CATEGORIES = ("P", "S", "Z", "C")  # 句読点・記号・空白・制御文字


def _norm_key(text: str) -> str:
    """重複判定用の鍵。NFKC に正規化し、句読点・記号・空白・制御文字を除き、英字を小文字にする。

    意味の同じ文を見つけるものではない。文字が(空白と句読点を除いて)一致する文だけを同じとみなす。
    """
    nfkc = unicodedata.normalize("NFKC", text)
    return "".join(c for c in nfkc if unicodedata.category(c)[0] not in _IGNORED_CATEGORIES).lower()


def _unique(items: list[str], exclude_keys: set[str] | None = None) -> list[str]:
    seen: set[str] = set(exclude_keys or ())
    out: list[str] = []
    for raw in items:
        text = _clean(raw)
        key = _norm_key(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _clean_optional(text: str | None) -> str | None:
    cleaned = _clean(text) if text else ""
    return cleaned or None


def merge_chunk_minutes(parts: list[ChunkMinutes]) -> MergedMinutes:
    """チャンクごとの結果を、時間順のまま1つにつなげる。

    - 議題名が完全に同じ（空白を除く）議題は、最初に出た議題にまとめ、要点を足す。
    - 要点・決定事項・未決事項は、空白と句読点を除いて一致する文を1つにする(意味の重複は判定しない)。
    - 同じ文が決定事項と未決事項の両方にあるときは、未決事項に残し、決定事項から除く(迷ったら決定事項にしない)。
    - ToDo は、内容(空白と句読点を除く)・担当・期限がすべて同じものを1つにする。
    - 空の文字列は捨てる。
    """
    topics: dict[str, list[str]] = {}
    decisions: list[str] = []
    todos: list[Todo] = []
    todo_keys: set[tuple] = set()
    open_issues: list[str] = []
    for part in parts:
        for topic in part.topics:
            title = _clean(topic.title)
            if not title:
                continue
            topics.setdefault(title, []).extend(topic.points)
        decisions.extend(part.decisions)
        open_issues.extend(part.open_issues)
        for todo in part.todos:
            task = _clean(todo.task)
            if not task:
                continue
            owner, due = _clean_optional(todo.owner), _clean_optional(todo.due)
            key = (_norm_key(task), owner, due)
            if key not in todo_keys:
                todo_keys.add(key)
                todos.append(Todo(task=task, owner=owner, due=due))
    unique_open = _unique(open_issues)
    return MergedMinutes(
        topics=[Topic(title=t, points=_unique(p)) for t, p in topics.items()],
        decisions=_unique(decisions, {_norm_key(t) for t in unique_open}),
        todos=todos,
        open_issues=unique_open,
    )


def check_overview(overview: str, merged: MergedMinutes) -> list[str]:
    """概要の検査。決定事項が無いのに「決定」「決まっ」を含むなら、警告の文を返す(書き換えない)。"""
    if merged.decisions:
        return []
    found = [w for w in DECISION_WORDS if w in overview]
    if not found:
        return []
    words = "、".join(f"「{w}」" for w in found)
    return [f"決定事項が無いのに、概要が {words} を含んでいます。依頼や意向を決定と書いた可能性があります。"]


def check_chunk_input_size(chunk: Chunk, cfg: Config) -> str | None:
    """チャンクの推定入力トークン数が上限（num_ctx - num_predict）を超えていれば、警告の文を返す。

    超えていても処理は止めない。概算が外れている可能性があるので、実際に投げて Ollama が返す
    prompt_eval_count を見る方が確かだからである（llm/chat.py が検査する）。
    """
    system, user = build_chunk_messages(chunk)
    estimated = estimate_messages_tokens(system, user, cfg.minutes.tokens_per_character)
    limit = ollama_api.input_limit(cfg.llm)
    if estimated <= limit:
        return None
    return (
        f"チャンク {chunk.index + 1}（発話 {chunk.seq_from}〜{chunk.seq_to}）の推定入力は約 {estimated} トークンで、"
        f"入力に使える上限（llm.num_ctx - llm.num_predict = {limit}）を超える見込みです。"
        f"minutes.chunk_minutes（現在 {cfg.minutes.chunk_minutes}）を下げるか、llm.num_ctx を上げてください。"
        "概算の係数は `uv run kijun doctor` で実測し直せます。"
    )


def _usage_text(chat: JsonChat) -> str:
    reply = chat.last_reply
    if reply is None or reply.prompt_eval_count is None or reply.eval_count is None:
        return "トークン数は取れず"
    return f"入力 {reply.prompt_eval_count} トークン、出力 {reply.eval_count} トークン"


def build_minutes(
    chat: JsonChat,
    cfg: Config,
    utterances: list[Utterance],
    log: Callable[[str], None] | None = None,
) -> tuple[MinutesDoc, int]:
    """発話から議事録の中身（概要と各節）を作る。(議事録の中身, チャンク数) を返す。LLM の失敗は MinutesError。"""
    log = log or (lambda m: print(m, file=sys.stderr))
    if not utterances:
        return MinutesDoc(NO_UTTERANCE_OVERVIEW, MergedMinutes()), 0

    chunks = split_into_chunks(utterances, cfg.minutes.chunk_minutes, 0)
    parts: list[ChunkMinutes] = []
    for chunk in chunks:
        warning = check_chunk_input_size(chunk, cfg)
        if warning is not None:
            log(f"警告: {warning}")
        label = f"チャンク {chunk.index + 1}/{len(chunks)}"
        system, user = build_chunk_messages(chunk)
        estimated = estimate_messages_tokens(system, user, cfg.minutes.tokens_per_character)
        log(f"{label}（発話 {chunk.seq_from}〜{chunk.seq_to}、推定 {estimated} トークン）を送ります")
        started = time.monotonic()
        try:
            parts.append(chat.ask(system, user, ChunkMinutes, label))
        except LlmError as e:
            raise MinutesError(
                f"{label}（発話 {chunk.seq_from}〜{chunk.seq_to}）が失敗しました。"
                f"失敗した区間を飛ばすと内容が欠けるので、議事録は作りません: {e}"
            ) from e
        log(f"{label} 完了（{time.monotonic() - started:.0f} 秒、{_usage_text(chat)}）")

    merged = merge_chunk_minutes(parts)
    if merged.is_empty:
        return MinutesDoc(NO_CONTENT_OVERVIEW, merged), len(chunks)

    system, user = build_overview_messages(
        {
            "topics": [t.model_dump() for t in merged.topics],
            "decisions": merged.decisions,
            "todos": [t.model_dump() for t in merged.todos],
            "open_issues": merged.open_issues,
        }
    )
    started = time.monotonic()
    try:
        overview = chat.ask(system, user, Overview, "概要")
    except LlmError as e:
        raise MinutesError(f"概要の生成が失敗しました: {e}") from e
    log(f"概要 完了（{time.monotonic() - started:.0f} 秒、{_usage_text(chat)}）")
    text = _clean(overview.overview)
    if not text:
        raise MinutesError("概要の生成が空の文字列を返しました")
    warnings = check_overview(text, merged)
    for w in warnings:
        log(f"警告: {w}")
    return MinutesDoc(text, merged, warnings), len(chunks)


def generate_minutes(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    conversation_id: str,
    chat: JsonChat,
    now: datetime | None = None,
    log: Callable[[str], None] | None = None,
) -> MinutesRun:
    """会議の議事録を作り、Markdown に書いて minutes テーブルに記録する（作り直しは上書き）。

    - 文字起こし済みでない会議は LookupError。
    - LLM が失敗したら MinutesError にし、ファイルも DB も書かない（既存の議事録を壊さない）。
    - 作り直しのとき、Discord に投稿済みの記録（discord_message_id、posted_at）は引き継ぐ。
      そのため、作り直した議事録は自動では再投稿されない。
    """
    now = now or datetime.now()
    conv = repo.get_conversation(con, conversation_id)
    if conv is None:
        raise LookupError(f"会議 {conversation_id} が conversations にありません。先に `kijun ingest` を実行してください。")
    if conv.transcribed_at is None:
        raise LookupError(f"会議 {conversation_id} は文字起こしが済んでいません。先に `kijun transcribe` を実行してください。")
    utterances = repo.get_utterances(con, conversation_id)

    doc, chunk_count = build_minutes(chat, cfg, utterances, log)
    model_name = cfg.llm.model
    text = render_markdown(conv, utterances, doc, model_name, now)
    path = minutes_path(cfg, conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    repo.upsert_minutes(con, conversation_id, path.as_posix(), model_name, now)
    return MinutesRun(conversation_id, model_name, now, chunk_count, path.as_posix(), doc)
