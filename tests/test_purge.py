from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from kijun.db import repo
from kijun.retention.purge import find_purge_candidates, format_candidates, purge_candidates
from tests.conftest import make_conversation

NOW = datetime(2026, 10, 31, 12, 0)


@pytest.fixture
def processed(cfg):
    cfg.paths.processed.mkdir(parents=True)
    return cfg.paths.processed


def register(con, cid: str, files: list[str], transcribed_days_ago: float | None):
    at = None if transcribed_days_ago is None else NOW - timedelta(days=transcribed_days_ago)
    repo.insert_conversation(con, make_conversation(cid, source_files=files, transcribed_at=at))


def test_30日以上前に文字起こしした会議の音声が対象になる(con, cfg, processed):
    (processed / "old.m4a").write_bytes(b"x")
    (processed / "new.m4a").write_bytes(b"x")
    register(con, "old", ["old.m4a"], 45)
    register(con, "new", ["new.m4a"], 10)
    cands = find_purge_candidates(con, cfg, NOW)
    assert [(c.conversation_id, c.file_name, c.elapsed_days) for c in cands] == [("old", "old.m4a", 45)]
    assert cands[0].path == processed / "old.m4a"


def test_ちょうど30日は対象_30日に満たなければ対象外(con, cfg, processed):
    for name in ("exact", "almost"):
        (processed / f"{name}.m4a").write_bytes(b"x")
    register(con, "exact", ["exact.m4a"], 30)
    register(con, "almost", ["almost.m4a"], 29.9)
    assert [c.conversation_id for c in find_purge_candidates(con, cfg, NOW)] == ["exact"]


def test_audio_days_の設定が効く(con, cfg, processed):
    (processed / "a.m4a").write_bytes(b"x")
    register(con, "a", ["a.m4a"], 10)
    assert find_purge_candidates(con, cfg, NOW) == []
    cfg.retention.audio_days = 7
    assert len(find_purge_candidates(con, cfg, NOW)) == 1


def test_文字起こし前の会議は対象外(con, cfg, processed):
    (processed / "a.m4a").write_bytes(b"x")
    register(con, "a", ["a.m4a"], None)
    assert find_purge_candidates(con, cfg, NOW) == []


def test_processed_に実在しないファイルは対象にしない(con, cfg, processed):
    (processed / "exists.m4a").write_bytes(b"x")
    register(con, "a", ["exists.m4a", "gone.m4a"], 40)
    assert [c.file_name for c in find_purge_candidates(con, cfg, NOW)] == ["exists.m4a"]


def test_1つの会議の複数ファイルがすべて対象になる(con, cfg, processed):
    for n in ("a.m4a", "b.m4a"):
        (processed / n).write_bytes(b"x")
    register(con, "a", ["a.m4a", "b.m4a"], 40)
    assert [c.file_name for c in find_purge_candidates(con, cfg, NOW)] == ["a.m4a", "b.m4a"]


def test_processed_の外は対象にしない(con, cfg, processed, tmp_path):
    outside = tmp_path / "outside.m4a"
    outside.write_bytes(b"x")
    register(con, "a", ["../outside.m4a"], 40)
    assert find_purge_candidates(con, cfg, NOW) == []


def test_一覧の表示では何も削除しない_削除は_purge_candidates_だけ(con, cfg, processed):
    f = processed / "a.m4a"
    f.write_bytes(b"x")
    register(con, "a", ["a.m4a"], 40)
    cands = find_purge_candidates(con, cfg, NOW)
    table = format_candidates(cands)
    assert f.exists()  # 探索と表示では削除しない
    assert "a.m4a" in table and "2026-09-21" in table and "40" in table
    for header in ("ファイル名", "会議ID", "文字起こし完了日", "経過日数"):
        assert header in table

    deleted, failed = purge_candidates(cands)
    assert [d.file_name for d in deleted] == ["a.m4a"] and failed == []
    assert not f.exists()


def test_削除するのは音声だけで_文字起こしの_JSON_は残る(con, cfg, processed):
    cfg.paths.transcripts.mkdir(parents=True)
    transcript = cfg.paths.transcripts / "a.json"
    transcript.write_text("{}", encoding="utf-8")
    (processed / "a.m4a").write_bytes(b"x")
    register(con, "a", ["a.m4a"], 40)
    purge_candidates(find_purge_candidates(con, cfg, NOW))
    assert transcript.exists()
    # DB の行も残る
    assert repo.get_conversation(con, "a") is not None


def test_既に消えているファイルの削除は失敗として報告する(con, cfg, processed):
    f = processed / "a.m4a"
    f.write_bytes(b"x")
    register(con, "a", ["a.m4a"], 40)
    cands = find_purge_candidates(con, cfg, NOW)
    f.unlink()
    deleted, failed = purge_candidates(cands)
    assert deleted == [] and len(failed) == 1


def test_対象が無ければ表は見出しだけ():
    assert "ファイル名" in format_candidates([])
