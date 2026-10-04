"""スキーマの適用とバージョン管理。

schema.sql は `CREATE ... IF NOT EXISTS` で書いてあるので、何度適用しても安全である。
適用済みのバージョンは schema_version テーブルに記録する。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

# schema.sql を変更したら、この値を1つ上げ、差分を適用する処理を apply_schema に足す。
SCHEMA_VERSION = 1

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def read_schema_sql() -> str:
    """schema.sql の全文を返す。"""
    return SCHEMA_PATH.read_text(encoding="utf-8")


def current_version(con: duckdb.DuckDBPyConnection) -> int | None:
    """適用済みの最新バージョンを返す。schema_version テーブルが無い、または空なら None。"""
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'schema_version'"
    ).fetchone()[0]
    if not exists:
        return None
    row = con.execute("SELECT max(version) FROM schema_version").fetchone()
    return row[0] if row else None


def apply_schema(con: duckdb.DuckDBPyConnection) -> int:
    """スキーマを適用し、適用後のバージョンを返す。

    すでに最新バージョンが記録されていれば、バージョン行は追加しない。
    DB のバージョンがこのコードより新しい場合は、古いコードで新しい DB を
    壊さないよう例外にする。
    """
    before = current_version(con)
    if before is not None and before > SCHEMA_VERSION:
        raise RuntimeError(
            f"DB のスキーマバージョン {before} が、このプログラムの対応バージョン "
            f"{SCHEMA_VERSION} より新しいです。kijun を更新してください。"
        )
    con.execute(read_schema_sql())
    if before is None or before < SCHEMA_VERSION:
        con.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            [SCHEMA_VERSION, datetime.now()],
        )
    return SCHEMA_VERSION
