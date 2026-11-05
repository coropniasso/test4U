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
# バージョン3: extracted_requests に transcription_unclear を、discord_posts に request_id を
# 足した（CLAUDE.md 7章の案A。文字起こしが壊れた発話から作られた抽出結果を Discord に出す）。
# バージョン4: extracted_requests に stated_reason_unverified と applies_when_unverified を足した
# （stated_reason / applies_when を原文と照合し、一致しなかった元の値を残す。2026-11-05）。
SCHEMA_VERSION = 4

# 移行先のバージョン -> 既存の DB に適用する文。
# schema.sql は `CREATE TABLE IF NOT EXISTS` なので、既存のテーブルに列を足さない。
# 列を足す変更は、新規の DB には schema.sql の CREATE で入り、既存の DB にはここで入る。
# `ADD COLUMN IF NOT EXISTS` にしてあるのは、途中で失敗して再実行しても安全にするため。
MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE discord_posts ADD COLUMN IF NOT EXISTS channel_id VARCHAR",
        "ALTER TABLE discord_posts ADD COLUMN IF NOT EXISTS pair_item_id VARCHAR",
    ],
    # 既存の行は FALSE になる。既存の抽出結果について、文字起こしが壊れていたかどうかは
    # 後から判定できない（判定していたのは抽出のときのモデルである）。推測で埋めず FALSE にする。
    # 既存の会議を見直したい場合は、その会議を再抽出する。
    # DuckDB は ALTER ADD COLUMN に制約を付けられない（Parser Error: Adding columns with
    # constraints not yet supported）。そのため、DEFAULT 付きで足してから NOT NULL にする2文に
    # 分ける。既存の行には DEFAULT の FALSE が入るので、NOT NULL を付けても失敗しない。
    # こうするのは、新規に作った DB（schema.sql の CREATE）と移行した DB のスキーマを
    # 同じにするためである。食い違うと、片方では NULL を書けて片方では書けない状態になる。
    # 2文目に IF NOT EXISTS は付けられないが、既に NOT NULL の列に対して実行しても通る。
    3: [
        "ALTER TABLE extracted_requests ADD COLUMN IF NOT EXISTS "
        "transcription_unclear BOOLEAN DEFAULT FALSE",
        "ALTER TABLE extracted_requests ALTER COLUMN transcription_unclear SET NOT NULL",
        "ALTER TABLE discord_posts ADD COLUMN IF NOT EXISTS request_id VARCHAR",
    ],
    # 既存の行は NULL になる。これらの行は照合の前に保存されたので、「照合で外れた」とは言えない。
    4: [
        "ALTER TABLE extracted_requests ADD COLUMN IF NOT EXISTS stated_reason_unverified VARCHAR",
        "ALTER TABLE extracted_requests ADD COLUMN IF NOT EXISTS applies_when_unverified VARCHAR",
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
