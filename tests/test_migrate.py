"""スキーマの移行。バージョン4(チェックリスト時代)の DB を、データを消さずに議事録の版へ移す。"""

from __future__ import annotations

from datetime import datetime

import duckdb

from kijun.db import migrate, repo

# バージョン4の DB にあった、議事録の版では作らないテーブルとビューの見本(列は一部だけ)。
V4_ONLY = """
CREATE TABLE extracted_requests (request_id VARCHAR PRIMARY KEY, conversation_id VARCHAR NOT NULL, quote VARCHAR NOT NULL);
CREATE TABLE knowledge_items (item_id VARCHAR PRIMARY KEY, check_text VARCHAR NOT NULL);
CREATE TABLE discord_posts (message_id VARCHAR PRIMARY KEY, post_type VARCHAR NOT NULL);
CREATE VIEW v_weekly_metrics AS SELECT conversation_id, count(*) AS requests FROM extracted_requests GROUP BY 1;
"""


def tables(con) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
    ).fetchall()}


def make_v4_db(path):
    con = duckdb.connect(str(path))
    con.execute(migrate.read_schema_sql())
    con.execute("DROP TABLE minutes")
    for statement in V4_ONLY.strip().split(";\n"):
        con.execute(statement)
    con.execute("DELETE FROM schema_version")
    con.execute("INSERT INTO schema_version VALUES (4, ?)", [datetime(2026, 11, 5)])
    con.execute("INSERT INTO extracted_requests VALUES ('r1', 'c1', '結論を先に書いて')")
    con.execute("INSERT INTO knowledge_items VALUES ('item_0001', '冒頭に行動を書くか')")
    con.execute("INSERT INTO discord_posts VALUES ('m1', 'new_candidate')")
    return con


def test_バージョン4のDBに_minutes_が足され_古いテーブルとデータはそのまま残る(tmp_path):
    con = make_v4_db(tmp_path / "v4.duckdb")
    assert "minutes" not in tables(con) and migrate.current_version(con) == 4

    assert migrate.apply_schema(con) == 5

    assert {"minutes", "extracted_requests", "knowledge_items", "discord_posts"} <= tables(con)
    assert con.execute("SELECT quote FROM extracted_requests").fetchall() == [("結論を先に書いて",)]
    assert con.execute("SELECT check_text FROM knowledge_items").fetchall() == [("冒頭に行動を書くか",)]
    assert con.execute("SELECT post_type FROM discord_posts").fetchall() == [("new_candidate",)]
    # 古いテーブルに依存するビューも残り、使える
    assert con.execute("SELECT requests FROM v_weekly_metrics").fetchall() == [(1,)]
    assert migrate.current_version(con) == 5
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 2  # 4 と 5 の履歴


def test_移行後の_minutes_に書き込める(tmp_path):
    con = make_v4_db(tmp_path / "v4.duckdb")
    migrate.apply_schema(con)
    repo.upsert_minutes(con, "c1", "c1.md", "qwen3.5:9b", datetime(2026, 11, 6))
    assert repo.get_minutes(con, "c1").path == "c1.md"


def test_移行は何度実行しても安全(tmp_path):
    con = make_v4_db(tmp_path / "v4.duckdb")
    migrate.apply_schema(con)
    migrate.apply_schema(con)
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM extracted_requests").fetchone()[0] == 1


def test_新規のDBは古いテーブルを作らない(con):
    assert tables(con) == {"schema_version", "conversations", "utterances", "minutes"}
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 1


def test_ensure_current_は古い_DB_を最新にし_未適用の_DB_には何もしない(tmp_path):
    con = make_v4_db(tmp_path / "v4.duckdb")
    assert migrate.ensure_current(con) == 5
    assert "minutes" in tables(con) and con.execute("SELECT count(*) FROM extracted_requests").fetchone()[0] == 1
    assert migrate.ensure_current(con) == 5  # 最新なら何もしない
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 2

    empty = duckdb.connect(str(tmp_path / "empty.duckdb"))
    assert migrate.ensure_current(empty) is None
    assert tables(empty) == set()  # 未適用の DB には何も作らない(db init を促す)


def test_batch_の_open_db_if_ready_は古い_DB_を移行する(cfg, tmp_path):
    from kijun.batch.pending import open_db_if_ready, pending_minutes

    make_v4_db(cfg.paths.db).close()
    with open_db_if_ready(cfg) as con:
        assert con is not None
        assert pending_minutes(con) == []  # minutes テーブルが無くて落ちない
