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

    assert migrate.apply_schema(con) == 3

    assert columns(con, "discord_posts")[-3:] == ["channel_id", "pair_item_id", "request_id"]
    rows = con.execute("SELECT message_id, post_type, item_id, channel_id, pair_item_id FROM discord_posts").fetchall()
    assert rows == [("m1", "new_candidate", "item_0001", None, None)]  # 既存の行が残り、新しい列は NULL
    assert migrate.current_version(con) == 3
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 2  # 1 と 3 の履歴


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
        "processed_at", "channel_id", "pair_item_id", "request_id",
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


# --- バージョン2 から 3 への移行 -------------------------------------------------------
#
# 上の make_v1_db は、いまの schema.sql で作った DB から discord_posts だけを古い形に差し替える
# ものなので、extracted_requests には最初から transcription_unclear がある。つまり移行3の
# 「列を足す」部分を試していない。ここでは列を落として、本当にバージョン2の形から始める。


def make_v2_db(path):
    """バージョン2のスキーマで作った DB。extracted_requests に行が1つ入っている。"""
    con = duckdb.connect(str(path))
    con.execute(migrate.read_schema_sql())
    con.execute("ALTER TABLE extracted_requests DROP COLUMN transcription_unclear")
    con.execute("ALTER TABLE discord_posts DROP COLUMN request_id")
    con.execute("DELETE FROM schema_version")
    con.execute("INSERT INTO schema_version VALUES (2, ?)", [datetime(2026, 11, 3)])
    con.execute(
        "INSERT INTO conversations (conversation_id, recorded_at, setting_format, setting_kind, "
        "source_files, created_at) VALUES ('c1', ?, 'f2f', 'teirei', '[]', ?)",
        [datetime(2026, 11, 3, 19, 30), datetime(2026, 11, 3)],
    )
    con.execute(
        "INSERT INTO extracted_requests (request_id, conversation_id, seq_from, seq_to, quote, "
        "request_summary, reusable, source_type, model_name, extracted_at) "
        "VALUES ('c1#m#0', 'c1', 0, 0, '結論を先に書いて', '結論を先に書くことを求めた', TRUE, "
        "'live_conversation', 'm', ?)",
        [datetime(2026, 11, 3)],
    )
    return con


def test_バージョン2のDBに_transcription_unclear_が足され_既存の行は_FALSE_になる(tmp_path):
    con = make_v2_db(tmp_path / "v2.duckdb")
    assert "transcription_unclear" not in columns(con, "extracted_requests")
    assert migrate.current_version(con) == 2

    assert migrate.apply_schema(con) == 3

    assert columns(con, "extracted_requests")[-1] == "transcription_unclear"
    assert columns(con, "discord_posts")[-1] == "request_id"
    # 既存の行は NULL ではなく FALSE になる。文字起こしが壊れていたかは後から判定できないので
    # 推測で埋めず、「読めた」側の FALSE にする。
    assert con.execute("SELECT transcription_unclear FROM extracted_requests").fetchall() == [(False,)]


def test_バージョン2から移行した表は新規のDBと同じく_NOT_NULL(tmp_path):
    # 新規の DB（schema.sql の CREATE）と移行した DB でスキーマが食い違うと、片方では NULL を
    # 書けて片方では書けない状態になる。DuckDB は ALTER ADD COLUMN に制約を付けられないので、
    # DEFAULT で足してから SET NOT NULL する2文に分けている。それが効いているかを確かめる。
    con = make_v2_db(tmp_path / "v2.duckdb")
    migrate.apply_schema(con)
    notnull = {
        r[1]: r[3]
        for r in con.execute("PRAGMA table_info('extracted_requests')").fetchall()
    }
    assert notnull["transcription_unclear"] is True


def test_バージョン2からの移行後に新しい列を使って読み書きできる(tmp_path):
    con = make_v2_db(tmp_path / "v2.duckdb")
    migrate.apply_schema(con)
    repo.insert_requests(
        con,
        [
            repo.RequestRow(
                request_id="c1#m#1", conversation_id="c1", seq_from=1, seq_to=1,
                speaker_from_label="SPEAKER_00", speaker_to_label=None, direction=None,
                counterpart_role=None, artifact_type=None, quote="3つ目簡単フォーイスなんか減らしてください",
                request_summary="「簡単フォーイス」を減らすことを求めた", stated_reason=None,
                applies_when=None, reusable=False, pass_criterion=None,
                source_type=repo.SOURCE_LIVE_CONVERSATION, model_name="m",
                extracted_at=datetime(2026, 11, 4), transcription_unclear=True,
            )
        ],
    )
    got = {r.request_id: r for r in repo.get_requests(con, "c1")}
    assert got["c1#m#1"].transcription_unclear is True
    assert got["c1#m#0"].transcription_unclear is False

    repo.insert_discord_post(
        con,
        repo.DiscordPostRow(
            message_id="m9", post_type=repo.POST_TRANSCRIPTION_UNCLEAR, item_id=None,
            conversation_id="c1", posted_at=datetime(2026, 11, 4), processed_at=None,
            channel_id="123", pair_item_id=None, request_id="c1#m#1",
        ),
    )
    posts = {p.message_id: p for p in repo.list_discord_posts(con)}
    assert posts["m9"].request_id == "c1#m#1" and posts["m9"].item_id is None
