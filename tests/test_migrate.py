"""スキーマの移行（仕様 2-10、テスト 8）と、第2弾で足した repo 関数の検査。"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from kijun.db import migrate, repo

# バージョン1の discord_posts（channel_id と pair_item_id が無い形）
V1_DISCORD_POSTS = """
CREATE TABLE discord_posts (
    message_id      VARCHAR PRIMARY KEY,
    post_type       VARCHAR NOT NULL,
    item_id         VARCHAR,
    conversation_id VARCHAR,
    posted_at       TIMESTAMP NOT NULL,
    processed_at    TIMESTAMP
)
"""


def columns(con, table: str) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? "
            "ORDER BY ordinal_position",
            [table],
        ).fetchall()
    ]


def make_v1_db(path):
    """バージョン1のスキーマで作った DB。discord_posts に行が1つ入っている。"""
    con = duckdb.connect(str(path))
    con.execute(migrate.read_schema_sql())
    con.execute("DROP TABLE discord_posts")
    con.execute(V1_DISCORD_POSTS)
    con.execute("DELETE FROM schema_version")
    con.execute("INSERT INTO schema_version VALUES (1, ?)", [datetime(2026, 10, 4)])
    con.execute(
        "INSERT INTO discord_posts VALUES ('m1', 'new_candidate', 'item_0001', NULL, ?, NULL)",
        [datetime(2026, 11, 1, 23, 30)],
    )
    return con


def test_バージョン1のDBにALTER_TABLEで列が足され_既存の行が残る(tmp_path):
    con = make_v1_db(tmp_path / "v1.duckdb")
    assert "channel_id" not in columns(con, "discord_posts")
    assert migrate.current_version(con) == 1

    assert migrate.apply_schema(con) == 2

    assert columns(con, "discord_posts")[-2:] == ["channel_id", "pair_item_id"]
    rows = con.execute("SELECT message_id, post_type, item_id, channel_id, pair_item_id FROM discord_posts").fetchall()
    assert rows == [("m1", "new_candidate", "item_0001", None, None)]  # 既存の行が残り、新しい列は NULL
    assert migrate.current_version(con) == 2
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 2  # 1 と 2 の履歴


def test_移行後の表に新しい列を使って書き込める(tmp_path):
    con = make_v1_db(tmp_path / "v1.duckdb")
    migrate.apply_schema(con)
    repo.insert_discord_post(
        con,
        repo.DiscordPostRow("m2", repo.POST_CONTRADICTION, "item_0001", None, datetime(2026, 11, 2), None, "999", "item_0002"),
    )
    got = {p.message_id: p for p in repo.list_discord_posts(con)}
    assert got["m2"].channel_id == "999" and got["m2"].pair_item_id == "item_0002"
    assert got["m1"].channel_id is None


def test_移行は何度実行しても安全(tmp_path):
    con = make_v1_db(tmp_path / "v1.duckdb")
    migrate.apply_schema(con)
    migrate.apply_schema(con)
    assert columns(con, "discord_posts").count("channel_id") == 1
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 2


def test_新規のDBは移行を経ずに最新の形で作られる(con):
    assert columns(con, "discord_posts") == [
        "message_id", "post_type", "item_id", "conversation_id", "posted_at",
        "processed_at", "channel_id", "pair_item_id",
    ]
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 1


def test_review_log_の_action_は定数に無い値を拒否する(con):
    with pytest.raises(ValueError, match="未定義"):
        repo.insert_review_log(con, "i1", "approve", None, None, datetime.now())
    assert len(set(repo.REVIEW_LOG_ACTIONS)) == 7
    repo.insert_review_log(con, "i1", repo.ACTION_STATUS_CHANGED, "candidate", "approved", datetime(2026, 11, 3))
    rows = repo.list_review_log(con, "i1")
    assert [(r.action, r.before, r.after) for r in rows] == [("status_changed", "candidate", "approved")]


def test_項目の欄の更新は_version_を上げ_許可した列だけ(con):
    item = repo.KnowledgeItemRow("i1", None, "文言", None, None, None, None, None, "candidate", 1, datetime(2026, 11, 1))
    repo.insert_knowledge_item(con, item)
    repo.update_item_fields(con, "i1", {"reason": "理由"}, datetime(2026, 11, 3))
    got = repo.get_knowledge_item(con, "i1")
    assert got.reason == "理由" and got.version == 2 and got.updated_at == datetime(2026, 11, 3)
    with pytest.raises(ValueError):
        repo.update_item_fields(con, "i1", {"status": "approved"}, datetime.now())
