"""スキーマの適用とバージョン管理。

schema.sql は `CREATE ... IF NOT EXISTS` で書いてあるので、何度適用しても安全である。
適用済みのバージョンは schema_version テーブルに記録する。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

# schema.sql を変更したら、この値を1つ上げ、差分を適用する処理を MIGRATIONS に足す。
# バージョン2: discord_posts に channel_id と pair_item_id を足した（第2弾の Discord 連携）。
SCHEMA_VERSION = 2

# 移行先のバージョン -> 既存の DB に適用する文。
# schema.sql は `CREATE TABLE IF NOT EXISTS` なので、既存のテーブルに列を足さない。
# 列を足す変更は、新規の DB には schema.sql の CREATE で入り、既存の DB にはここで入る。
# `ADD COLUMN IF NOT EXISTS` にしてあるのは、途中で失敗して再実行しても安全にするため。
MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE discord_posts ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
        "ALTER TABLE discord_posts ADD COLUMN IF NOT EXISTS pair_item_id VARCHAR",
    ],
}

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
    記録されているバージョンが古い DB には、MIGRATIONS の差分を順に適用する
    （記録が無い = 新規の DB には、schema.sql の CREATE が最新の形で作るので適用しない）。
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
    if before is not None:
        for version in sorted(v for v in MIGRATIONS if before < v <= SCHEMA_VERSION):
            for statement in MIGRATIONS[version]:
                con.execute(statement)
    if before is None or before < SCHEMA_VERSION:
        con.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            [SCHEMA_VERSION, datetime.now()],
        )
    return SCHEMA_VERSION
