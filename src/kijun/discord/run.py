"""`kijun discord post` の本体。夜間バッチが子プロセスとして起動する。

DiscordClient は引数で受け取るので、テストでは偽の実装を渡せる。実際の実装(discord.py)は build_client が作る。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.discord.base import DiscordClient, DiscordError
from kijun.discord.post import print_post_report, run_post

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


def post_command(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    client: DiscordClient | None = None,
    echo: Callable[[str], None] = print,
    now: datetime | None = None,
) -> int:
    """未投稿の議事録を投稿する。終了コードを返す(投稿に失敗したものがあれば 1。設定が空で何もしなかった場合は 0)。"""
    if not cfg.discord.minutes_channel_id:
        print_post_report(run_post(con, cfg, _NoClient(), now), echo)  # 設定が空のときの報告を作らせる
        return 0
    if not repo.list_unposted_minutes(con):
        echo("未投稿の議事録はありません。")
        return 0  # トークンが無くても、投稿するものが無ければ失敗にしない
    client = client or build_client(cfg)
    try:
        report = run_post(con, cfg, client, now)
    except DiscordError as e:
        echo(f"エラー: {e}")
        return 1
    print_post_report(report, echo)
    return 1 if report.failed else 0


class _NoClient:
    """設定が空のときに run_post へ渡す代わりの値。接続は一切しない(呼ばれたら誤り)。"""

    def post_file(self, channel_id, content, filename, data):  # pragma: no cover
        raise AssertionError("設定が空のときは呼ばれない")
