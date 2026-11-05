"""extract サブコマンドの本体（計画書 7-7）。

会議の発話をチャンクに分け、Extractor で抽出し、重複を除き、extracted_requests に書く。
Extractor は引数で受け取るので、テストでは FakeExtractor を渡せる。
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.extract.base import ExtractionError, Extractor, RepairEvent
from kijun.extract.chunk import Chunk, MergeOutcome, merge_chunk_results, split_into_chunks
from kijun.extract.models import ExtractedRequest
from kijun.extract.prompt import build_messages
from kijun.extract.tokens import estimate_messages_tokens
from kijun.extract.verbatim import Rejection, check_items, rejections, source_text


@dataclass(frozen=True)
class ChunkFailure:
    """抽出に失敗したチャンク。seq は会議全体の番号。"""

    chunk_index: int
    seq_from: int
    seq_to: int
    reason: str


@dataclass
class ExtractionRun:
    model_name: str
    chunk_count: int
    failures: list[ChunkFailure] = field(default_factory=list)
    outcome: MergeOutcome = field(default_factory=lambda: MergeOutcome(items=[]))
    saved_count: int = 0
    # 背面が壊れた JSON を返して修復した記録（背面から受け取る）
    repair_events: list[RepairEvent] = field(default_factory=list)
    # 推定トークン数が max_input_tokens を超えたチャンクの警告
    oversize_warnings: list[str] = field(default_factory=list)
    # stated_reason / applies_when を原文と照合して外した値（保存した項目のぶん。extract/verbatim.py）
    verbatim_rejections: list[Rejection] = field(default_factory=list)

    def rejected_count(self, field_name: str) -> int:
        return sum(1 for r in self.verbatim_rejections if r.field == field_name)


def extract_log_path(cfg: Config, conversation_id: str, model_name: str) -> Path:
    """失敗したチャンクの記録の保存先。

    失敗したチャンクは DuckDB のスキーマに列が無いので、DB と同じディレクトリの
    extract_logs/ に JSON で残す。比較表（compare）がここを読む。
    """
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model_name)
    return cfg.paths.db.parent / "extract_logs" / f"{conversation_id}__{safe_model}.json"


def write_extract_log(path: Path, conversation_id: str, run: ExtractionRun, at: datetime) -> None:
    payload = {
        "conversation_id": conversation_id,
        "model_name": run.model_name,
        "extracted_at": at.isoformat(timespec="seconds"),
        "chunk_count": run.chunk_count,
        "failed_chunks": [
            {"chunk_index": f.chunk_index, "seq_from": f.seq_from, "seq_to": f.seq_to, "reason": f.reason}
            for f in run.failures
        ],
        "repaired_chunks": [
            {
                "chunk_index": r.chunk_index,
                "seq_from": r.seq_from,
                "seq_to": r.seq_to,
                "actions": r.actions,
                "truncated_items": r.truncated_items,
                "attempt": r.attempt,
            }
            for r in run.repair_events
        ],
        "oversize_warnings": run.oversize_warnings,
        "verbatim_rejected": {
            "stated_reason": [
                {"value": r.value, "quote": r.quote}
                for r in run.verbatim_rejections if r.field == "stated_reason"
            ],
            "applies_when": [
                {"value": r.value, "quote": r.quote}
                for r in run.verbatim_rejections if r.field == "applies_when"
            ],
        },
        "invalid_count": run.outcome.invalid_count,
        "overlap_dropped_count": run.outcome.overlap_dropped_count,
        "duplicate_count": run.outcome.duplicate_count,
        "saved_count": run.saved_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_failed_chunk_count(path: Path) -> int | None:
    """記録から失敗したチャンク数を読む。記録が無ければ None。"""
    if not path.exists():
        return None
    return len(json.loads(path.read_text(encoding="utf-8")).get("failed_chunks", []))


def request_id_model(model_name: str) -> str:
    """request_id に入れるモデル名。使える文字（英数字と . _ + -）以外を - に置き換える。

    Ollama のモデル名には ':' が入る（qwen3.5:9b）し、取得元つきの名前には '/' も入る。
    設定 run_label が英数字と . _ - に限られているのは「request_id にも入るから」なので、
    request_id の文字の規則をモデル名にも揃える。extracted_requests.model_name の列には、
    置き換えない元の名前（':' を含む）を入れる。'+' は model_name と run_label をつなぐ記号で、残す。
    """
    return re.sub(r"[^A-Za-z0-9._+-]", "-", model_name)


def _to_row(
    item: ExtractedRequest, conversation_id: str, model_name: str, number: int, at: datetime
) -> repo.RequestRow:
    return repo.RequestRow(
        request_id=f"{conversation_id}#{request_id_model(model_name)}#{number}",
        conversation_id=conversation_id,
        seq_from=item.seq_from,
        seq_to=item.seq_to,
        speaker_from_label=item.speaker_from_label,
        speaker_to_label=item.speaker_to_label,
        # speaker_map が埋まるまで方向は確定できないので NULL にする（計画書 4-1）
        direction=None,
        counterpart_role=item.counterpart_role,
        artifact_type=item.artifact_type,
        quote=item.quote,
        request_summary=item.request_summary,
        stated_reason=item.stated_reason,
        applies_when=item.applies_when,
        reusable=item.reusable,
        pass_criterion=item.pass_criterion,
        transcription_unclear=item.transcription_unclear,
        stated_reason_unverified=getattr(item, "stated_reason_unverified", None),
        applies_when_unverified=getattr(item, "applies_when_unverified", None),
        source_type=repo.SOURCE_LIVE_CONVERSATION,
        model_name=model_name,
        extracted_at=at,
    )


def check_chunk_input_size(
    chunk: Chunk, cfg: Config, max_input_tokens: int | None
) -> str | None:
    """チャンクの推定入力トークン数が上限を超えていれば、警告の文を返す。

    max_input_tokens が None のときは検査しない。推測で止めないため。
    超えていても処理は止めない。概算が外れている可能性があるので、実際に投げて
    Ollama が返す prompt_eval_count を見る方が確かだからである（ollama_impl が検査する）。
    """
    if max_input_tokens is None:
        return None
    system, user = build_messages(chunk, cfg)
    estimated = estimate_messages_tokens(system, user, cfg.extract.tokens_per_character)
    if estimated <= max_input_tokens:
        return None
    return (
        f"チャンク {chunk.index + 1}（発話 {chunk.seq_from}〜{chunk.seq_to}）の推定入力は "
        f"約 {estimated} トークンで、入力に使える上限（extract.ollama.num_ctx - num_predict = "
        f"{max_input_tokens}）を超える見込みです。"
        f"extract.chunk_minutes（現在 {cfg.extract.chunk_minutes}）を下げるか、num_ctx を上げてください。"
        "概算の係数は `uv run kijun doctor` で実測し直せます。"
    )


def _usage_text(usage: dict | None) -> str:
    """完了の表示に入れる入力・出力のトークン数。取れなければその旨を返す。"""
    if not usage or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return "トークン数は取れず（Ollama が prompt_eval_count / eval_count を返さない）"
    return f"入力 {usage['prompt_tokens']} トークン、出力 {usage['completion_tokens']} トークン"


def run_extraction(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    conversation_id: str,
    extractor: Extractor,
    now: datetime | None = None,
    log: Callable[[str], None] | None = None,
) -> ExtractionRun:
    """会議を抽出して extracted_requests に保存する。

    - 失敗したチャンクは記録して次に進む。会議全体は落とさない。失敗したチャンクの
      発話の範囲は、標準エラーに出す。
    - 同じ会議・同じモデルの抽出結果が既にある場合は、置き換える。
      ただし、その結果が item_evidence に紐付いている場合は、根拠の参照先が無くなるので中止する。
    - 全チャンクが失敗した場合は、既存の結果を消さずに例外にする。
    """
    now = now or datetime.now()
    log = log or (lambda m: print(m, file=sys.stderr))

    utterances = repo.get_utterances(con, conversation_id)
    if not utterances:
        raise LookupError(
            f"会議 {conversation_id} の発話がありません。先に `kijun transcribe` を実行してください。"
        )

    chunks = split_into_chunks(
        utterances, cfg.extract.chunk_minutes, cfg.extract.overlap_minutes
    )
    run = ExtractionRun(model_name=extractor.model_name, chunk_count=len(chunks))
    results: list[tuple] = []
    for chunk in chunks:
        warning = check_chunk_input_size(chunk, cfg, extractor.max_input_tokens)
        if warning is not None:
            run.oversize_warnings.append(warning)
            log(f"警告: {warning}")
        system, user = build_messages(chunk, cfg)
        estimated = estimate_messages_tokens(system, user, cfg.extract.tokens_per_character)
        label = f"チャンク {chunk.index + 1}/{len(chunks)}"
        log(f"{label}（発話 {chunk.seq_from}〜{chunk.seq_to}、推定 {estimated} トークン）を送ります")
        started = time.monotonic()
        try:
            result = extractor.extract(chunk)
            # 渡したチャンクの全発話を原文として、stated_reason / applies_when を照合する。
            checked = check_items(result.items, source_text(chunk.utterances))
            results.append((chunk, checked))
            log(
                f"{label} 完了（{time.monotonic() - started:.0f} 秒、"
                f"{_usage_text(getattr(extractor, 'last_usage', None))}、抽出 {len(result.items)} 件）"
            )
        except ExtractionError as e:
            run.failures.append(ChunkFailure(chunk.index, chunk.seq_from, chunk.seq_to, str(e)))
            log(
                f"失敗: {label}（発話 {chunk.seq_from}〜{chunk.seq_to}、"
                f"{time.monotonic() - started:.0f} 秒で失敗）: {e}"
            )
            results.append((chunk, None))

    # 修復の記録は背面が持っている（Protocol の属性）。extract_logs に書くために受け取る。
    run.repair_events = list(extractor.repair_events)

    if len(run.failures) == len(chunks):
        write_extract_log(extract_log_path(cfg, conversation_id, run.model_name), conversation_id, run, now)
        raise ExtractionError(f"すべてのチャンク（{len(chunks)} 個）の抽出に失敗しました")

    run.outcome = merge_chunk_results(results)
    run.verbatim_rejections = rejections(run.outcome.items)

    linked = repo.count_linked_requests(con, conversation_id, run.model_name)
    if linked:
        raise RuntimeError(
            f"会議 {conversation_id}・モデル {run.model_name} の抽出結果のうち {linked} 件が "
            "item_evidence に紐付いているため、置き換えられません。"
        )
    repo.delete_requests(con, conversation_id, run.model_name)
    rows = [
        _to_row(item, conversation_id, run.model_name, i, now)
        for i, item in enumerate(run.outcome.items, start=1)
    ]
    repo.insert_requests(con, rows)
    run.saved_count = len(rows)
    write_extract_log(extract_log_path(cfg, conversation_id, run.model_name), conversation_id, run, now)
    return run
