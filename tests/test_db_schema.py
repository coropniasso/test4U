from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from kijun.db import migrate, repo
from kijun.models import Utterance
from tests.conftest import make_conversation

EXPECTED_TABLES = {"schema_version", "conversations", "utterances", "minutes"}


def columns(con, table: str) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()]


def test_新しい_DB_には議事録に必要なテーブルだけができる(con):
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
    ).fetchall()}
    assert tables == EXPECTED_TABLES  # チェックリスト時代のテーブルは作らない
    views = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_type = 'VIEW'").fetchone()[0]
    assert views == 0


def test_conversations_は_setting_を2列に分けている(con):
    cols = columns(con, "conversations")
    assert "setting" not in cols
    assert cols == [
        "conversation_id", "recorded_at", "setting_format", "setting_kind", "source_files",
        "transcript_path", "duration_sec", "transcribed_at", "created_at",
    ]


def test_utterances_と_minutes_の列(con):
    assert columns(con, "utterances") == ["conversation_id", "seq", "speaker_label", "start_sec", "end_sec", "text"]
    assert columns(con, "minutes") == [
        "conversation_id", "path", "model_name", "created_at", "discord_message_id", "posted_at",
    ]


def test_スキーマの適用は何度でも安全で_バージョン行は1つだけ(con):
    migrate.apply_schema(con)
    migrate.apply_schema(con)
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 1
    assert migrate.current_version(con) == migrate.SCHEMA_VERSION == 5


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


# --- minutes ------------------------------------------------------------------------------


def test_議事録の記録と投稿済みの記録(con):
    t1, t2 = datetime(2026, 11, 6, 1, 0), datetime(2026, 11, 6, 2, 0)
    repo.upsert_minutes(con, "c1", "data/minutes/c1.md", "qwen3.5:9b", t1)
    got = repo.get_minutes(con, "c1")
    assert got == repo.MinutesRow("c1", "data/minutes/c1.md", "qwen3.5:9b", t1, None, None)
    assert [m.conversation_id for m in repo.list_unposted_minutes(con)] == ["c1"]

    repo.mark_minutes_posted(con, "c1", "m123", t2)
    assert repo.list_unposted_minutes(con) == []
    posted = repo.get_minutes(con, "c1")
    assert posted.discord_message_id == "m123" and posted.posted_at == t2


def test_議事録を作り直しても投稿済みの記録は残る(con):
    t1, t2, t3 = datetime(2026, 11, 6, 1), datetime(2026, 11, 6, 2), datetime(2026, 11, 7, 1)
    repo.upsert_minutes(con, "c1", "a.md", "m1", t1)
    repo.mark_minutes_posted(con, "c1", "m123", t2)
    repo.upsert_minutes(con, "c1", "b.md", "m2", t3)  # 作り直し
    got = repo.get_minutes(con, "c1")
    assert (got.path, got.model_name, got.created_at) == ("b.md", "m2", t3)
    assert got.discord_message_id == "m123" and got.posted_at == t2  # 自動では再投稿されない
    assert repo.list_unposted_minutes(con) == []
    assert con.execute("SELECT count(*) FROM minutes").fetchone()[0] == 1


def test_議事録の無い文字起こし済みの会議だけが対象になる(con):
    repo.insert_conversation(con, make_conversation("c1"))  # 文字起こし前
    repo.insert_conversation(con, make_conversation("c2", transcribed_at=datetime(2026, 9, 21)))
    repo.insert_conversation(con, make_conversation("c3", transcribed_at=datetime(2026, 9, 21)))
    repo.upsert_minutes(con, "c3", "c3.md", "m", datetime(2026, 9, 22))
    assert [c.conversation_id for c in repo.list_conversations_without_minutes(con)] == ["c2"]
