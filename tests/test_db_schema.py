from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from kijun.db import migrate, repo
from kijun.models import Utterance
from tests.conftest import make_conversation

EXPECTED_TABLES = {
    "schema_version",
    "conversations",
    "utterances",
    "extracted_requests",
    "knowledge_items",
    "item_evidence",
    "review_log",
    "speaker_map",
    "discord_posts",
    "embeddings",
}


def columns(con, table: str) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()]


def test_8テーブルと追加の2テーブルとビューができる(con):
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
    ).fetchall()}
    assert tables == EXPECTED_TABLES
    views = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
    ).fetchall()}
    assert views == {"v_weekly_metrics"}


def test_conversations_は_setting_を2列に分けている(con):
    cols = columns(con, "conversations")
    assert "setting" not in cols
    assert cols == [
        "conversation_id", "recorded_at", "setting_format", "setting_kind", "source_files",
        "transcript_path", "duration_sec", "transcribed_at", "created_at",
    ]


def test_extracted_requests_の列は計画書どおり(con):
    assert columns(con, "extracted_requests") == [
        "request_id", "conversation_id", "seq_from", "seq_to", "speaker_from_label",
        "speaker_to_label", "direction", "counterpart_role", "artifact_type", "quote",
        "request_summary", "stated_reason", "applies_when", "reusable", "pass_criterion",
        "source_type", "model_name", "extracted_at",
    ]


def test_knowledge_items_と_embeddings_の列(con):
    assert columns(con, "knowledge_items") == [
        "item_id", "artifact_type", "check_text", "pass_criterion", "reason", "applies_when",
        "not_applies_when", "importance", "status", "version", "updated_at",
    ]
    assert columns(con, "embeddings") == [
        "owner_type", "owner_id", "model_name", "dim", "vector", "created_at",
    ]


def test_スキーマの適用は何度でも安全で_バージョン行は1つだけ(con):
    migrate.apply_schema(con)
    migrate.apply_schema(con)
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 1
    assert migrate.current_version(con) == migrate.SCHEMA_VERSION == 1


def test_適用前のバージョンは_None(tmp_path):
    c = duckdb.connect(str(tmp_path / "x.duckdb"))
    assert migrate.current_version(c) is None


def test_DBのバージョンがコードより新しければ拒否する(con):
    con.execute("INSERT INTO schema_version VALUES (99, ?)", [datetime.now()])
    with pytest.raises(RuntimeError, match="新しい"):
        migrate.apply_schema(con)


def test_conversations_の書き込みと読み出し(con):
    conv = make_conversation(source_files=["a.m4a", "b.m4a"])
    repo.insert_conversation(con, conv)
    assert repo.get_conversation(con, conv.conversation_id) == conv
    assert repo.get_conversation(con, "none") is None
    assert [c.conversation_id for c in repo.list_conversations(con)] == [conv.conversation_id]


def test_文字起こしの完了を記録できる(con):
    repo.insert_conversation(con, make_conversation())
    at = datetime(2026, 9, 21, 0, 10)
    repo.mark_transcribed(con, "20260920_1930_f2f_teirei", "t/x.json", 1234.5, at)
    got = repo.get_conversation(con, "20260920_1930_f2f_teirei")
    assert got.transcribed_at == at and got.transcript_path == "t/x.json" and got.duration_sec == 1234.5


def test_発話は入れ直せる(con):
    u1 = [Utterance(0, "SPEAKER_00", 0.0, 1.0, "a"), Utterance(1, None, 1.0, 2.0, "b")]
    repo.replace_utterances(con, "c1", u1)
    assert repo.get_utterances(con, "c1") == u1
    # 文字起こしをやり直した場合。主キー違反にならず、古い行が残らない
    u2 = [Utterance(0, "SPEAKER_01", 0.0, 3.0, "c")]
    repo.replace_utterances(con, "c1", u2)
    assert repo.get_utterances(con, "c1") == u2


def test_重要度は_1から3だけ(con):
    item = repo.KnowledgeItemRow("i1", None, "x", None, None, None, None, 4, "candidate", 1, datetime.now())
    with pytest.raises(duckdb.ConstraintException):
        repo.insert_knowledge_item(con, item)


def test_item_id_の採番(con):
    assert repo.next_item_id(con) == "item_0001"
    repo.insert_knowledge_item(
        con, repo.KnowledgeItemRow("item_0007", None, "x", None, None, None, None, None, "candidate", 1, datetime.now())
    )
    assert repo.next_item_id(con) == "item_0008"


def test_ベクトルは次元数が違っても保存できる(con):
    import numpy as np

    now = datetime.now()
    repo.put_embeddings(con, "knowledge_item", "m", {"a": np.array([1, 2, 3], dtype=np.float32)}, now)
    repo.put_embeddings(con, "knowledge_item", "m2", {"a": np.arange(1024, dtype=np.float32)}, now)
    got = repo.get_embeddings(con, "knowledge_item", ["a", "b"], "m")
    assert list(got) == ["a"] and got["a"].tolist() == [1.0, 2.0, 3.0]
    assert con.execute("SELECT dim FROM embeddings ORDER BY dim").fetchall() == [(3,), (1024,)]
    # model_name が違う行は返さない
    assert repo.get_embeddings(con, "knowledge_item", ["a"], "other") == {}
    # 上書き
    repo.put_embeddings(con, "knowledge_item", "m", {"a": np.array([9, 9, 9], dtype=np.float32)}, now)
    assert repo.get_embeddings(con, "knowledge_item", ["a"], "m")["a"].tolist() == [9.0, 9.0, 9.0]


def test_review_log_の連番とビューが使える(con):
    assert con.execute("SELECT nextval('review_log_seq')").fetchone()[0] == 1
    repo.insert_conversation(con, make_conversation())
    rows = con.execute("SELECT conversations, requests, reusable_ratio FROM v_weekly_metrics").fetchall()
    assert rows == [(1, 0, None)]
