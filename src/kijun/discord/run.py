"""discord サブコマンドの本体(仕様 2-1)。

`kijun discord read`(CLAUDE.md 4-8 の手順1)と `kijun discord post`(手順8)の2つに分ける。
夜間バッチが順に起動し、処理が終われば接続を閉じる。bot は常駐しない。

DiscordClient と LLM の実装は引数で受け取るので、テストでは偽の実装を渡せる。
実際の実装(discord.py、Strata)は build_* が作る。
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.deps import MissingDependencyError
from kijun.discord.base import DiscordClient, DiscordError
from kijun.discord.post import PostReport, run_post
from kijun.discord.read import ReadReport, run_read
from kijun.review.importance import StrataImportanceJudge
from kijun.review.llm import StrataJsonChat
from kijun.review.reply import StrataReplyRouter
from kijun.review.speaker import StrataSpeakerInterpreter
from kijun.transport import HttpTransport

TOKEN_ENV = "DISCORD_BOT_TOKEN"


def build_client(cfg: Config) -> DiscordClient:
    """環境変数 DISCORD_BOT_TOKEN から discord.py の実装を作る。トークンが無ければ RuntimeError。"""
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"環境変数 {TOKEN_ENV} が設定されていません。Discord Developer Portal で bot を作ってトークンを取得し、"
            f'PowerShell で $env:{TOKEN_ENV} = "..." を実行してください(設定ファイルには書きません)。'
        )
    from kijun.discord.impl import DiscordPyClient

    return DiscordPyClient(token)


@dataclass
class LlmParts:
    """Strata を使う3つの部品。httpx が入っていないときは、すべて None(LLM を使えない)。"""

    router: StrataReplyRouter | None = None
    interpreter: StrataSpeakerInterpreter | None = None
    judge: StrataImportanceJudge | None = None


def build_llm(cfg: Config, transport: HttpTransport | None = None) -> LlmParts:
    """Strata の部品を作る。

    httpx が無いときは、止めずに LLM 無しで続ける。返信は原文だけ記録して未処理に残り、
    重要度は NULL のまま投稿される(判定できないものを推測で埋めない)。
    """
    try:
        chat = StrataJsonChat(cfg, transport)
    except MissingDependencyError as e:
        print(f"警告: LLM を使えません。LLM 無しで続けます: {e}", file=sys.stderr)
        return LlmParts()
    return LlmParts(StrataReplyRouter(chat), StrataSpeakerInterpreter(chat), StrataImportanceJudge(chat))


def print_read_report(report: ReadReport, echo: Callable[[str], None]) -> None:
    for line in report.messages:
        echo(line)
    for line in report.ambiguous:
        echo(f"判定できませんでした: {line}")
    echo(
        f"読み取り: 処理済み {report.processed_posts} 件、status の更新 {report.status_changes} 件、"
        f"重要度の上書き {report.importance_overrides} 件、返信の振り分け {report.replies_routed} 件"
        f"(失敗 {report.replies_failed} 件)、判定できないリアクション {len(report.ambiguous)} 件"
    )


def write_review_log(path: Path, kind: str, report, at: datetime) -> Path:
    """post / read の実行の報告を、そのまま JSON に書き出す。

    DB に残るのは「投稿したこと」（discord_posts の行）と「処理済みになった時刻」
    （processed_at）だけで、**どのリアクションが付いたかは残らない。** 6章の指標と、
    7章の「文字起こしが読めなかった投稿に ✅ と ❌ のどちらが付いたか」は、この記録から数える。

    「✅ が付いた件数」は、`processed_by_type["transcription_unclear"]` から
    `len(unclear_to_fix)`（❌ の件数）を引いて求める。

    書き出しに失敗しても処理は止めない。記録は後から数えるためのもので、
    レビューの反映そのもの（DB への書き込み）は既に終わっているためである。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kind": kind, "at": at.isoformat(timespec="seconds"), **asdict(report)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def review_log_path(cfg: Config, kind: str, at: datetime) -> Path:
    return cfg.paths.review_logs / f"{kind}_{at.strftime('%Y%m%d_%H%M%S')}.json"


def _save_review_log(cfg: Config, kind: str, report, at: datetime, echo: Callable[[str], None]) -> None:
    try:
        path = write_review_log(review_log_path(cfg, kind, at), kind, report, at)
    except OSError as e:
        echo(f"注意: 実行の記録を書けませんでした（レビューの反映は済んでいます）: {e}")
        return
    echo(f"実行の記録を保存しました: {path}")


def read_command(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    client: DiscordClient | None = None,
    llm: LlmParts | None = None,
    echo: Callable[[str], None] = print,
    now: datetime | None = None,
) -> int:
    """リアクションと返信を読み取る。終了コードを返す(0: 成功。設定が空で何もしなかった場合も 0)。"""
    at = now or datetime.now()
    if not cfg.discord.review_channel_id and not cfg.discord.contradictions_channel_id:
        report = run_read(con, cfg, _NoClient(), None, None, now)  # 設定が空のときの報告を作らせる
        print_read_report(report, echo)
        return 0  # 何も読んでいないので記録しない
    client = client or build_client(cfg)
    llm = llm or build_llm(cfg)
    try:
        report = run_read(con, cfg, client, llm.router, llm.interpreter, now)
    except DiscordError as e:
        echo(f"エラー: {e}")
        return 1
    print_read_report(report, echo)
    _save_review_log(cfg, "read", report, at, echo)
    return 0


def print_post_report(report: PostReport, echo: Callable[[str], None]) -> None:
    for line in report.messages:
        echo(line)
    for line in report.failures:
        echo(f"失敗: {line}")
    if report.skipped_for_config:
        return
    posted = "、".join(f"{k} {v} 件" for k, v in report.posted.items()) or "なし"
    echo(f"投稿: {posted}")
    echo(f"まだ投稿していない新規候補: {report.not_posted_candidates} 件")


def post_command(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    client: DiscordClient | None = None,
    llm: LlmParts | None = None,
    echo: Callable[[str], None] = print,
    now: datetime | None = None,
) -> int:
    """5種類を投稿する。終了コードを返す(投稿に失敗したものがあれば 1)。"""
    at = now or datetime.now()
    if not cfg.discord.review_channel_id and not cfg.discord.contradictions_channel_id:
        print_post_report(run_post(con, cfg, _NoClient(), None, now), echo)
        return 0  # 何も投稿していないので記録しない
    client = client or build_client(cfg)
    llm = llm or build_llm(cfg)
    report = run_post(con, cfg, client, llm.judge, now)
    print_post_report(report, echo)
    _save_review_log(cfg, "post", report, at, echo)
    return 1 if report.failed else 0


class _NoClient:
    """設定が空のときに run_read / run_post へ渡す代わりの値。接続は一切しない(呼ばれたら誤り)。"""

    def post(self, channel_id, content):  # pragma: no cover
        raise AssertionError("設定が空のときは呼ばれない")

    def fetch_reactions(self, channel_id, message_ids):  # pragma: no cover
        raise AssertionError("設定が空のときは呼ばれない")

    def fetch_thread_replies(self, channel_id, message_ids):  # pragma: no cover
        raise AssertionError("設定が空のときは呼ばれない")
