"""夜間バッチが、どの会議に何の処理が要るかを DB から調べる(読み取りだけ)。

子プロセスが DB を書き換えるので、各段の直前に調べ直す。子プロセスが動いている間は、この接続を開かない
(DuckDB は同時に複数のプロセスから書き込めない)。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager

import duckdb

from kijun.config import Config
from kijun.db import migrate, repo


@contextmanager
def open_db_if_ready(cfg: Config) -> Iterator[duckdb.DuckDBPyConnection | None]:
    """スキーマ適用済みの DB に接続する。DB が無い・スキーマが未適用なら None を渡す。古いバージョンなら最新にする。"""
    if not cfg.paths.db.exists():
        yield None
        return
    with closing(repo.connect(cfg.paths.db)) as con:
        yield con if migrate.ensure_current(con) is not None else None


def pending_transcription(con: duckdb.DuckDBPyConnection) -> list[str]:
    """登録済みで、まだ文字起こしが済んでいない会議(録音の古い順)。"""
    return [c.conversation_id for c in repo.list_conversations(con) if c.transcribed_at is None]


def pending_minutes(con: duckdb.DuckDBPyConnection) -> list[str]:
    """文字起こしが済んでいて、議事録がまだ無い会議(録音の古い順)。

    議事録の生成に失敗した会議は行が作られないので、翌晩も対象になる(再試行される)。
    """
    return [c.conversation_id for c in repo.list_conversations_without_minutes(con)]
