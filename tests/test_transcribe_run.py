from __future__ import annotations

import json
from datetime import datetime

import pytest

from kijun.db import repo
from kijun.transcribe.base import SpeakerTurn, TranscribedSegment
from kijun.transcribe.run import find_audio_file, transcribe_conversation, transcribe_single_audio
from tests.conftest import make_conversation
from tests.fakes import FakeDiarizer, FakeTranscriber

CID = "20260920_1930_f2f_teirei"
NOW = datetime(2026, 9, 21, 0, 10)

SEGMENTS = [
    TranscribedSegment(0.0, 2.0, "おはよう"),
    TranscribedSegment(2.2, 4.0, "ございます"),
    TranscribedSegment(5.0, 7.0, "資料を見ました"),
]
TURNS = [SpeakerTurn(0.0, 4.5, "SPEAKER_00"), SpeakerTurn(4.5, 8.0, "SPEAKER_01")]


@pytest.fixture
def audio(cfg):
    cfg.paths.inbox.mkdir(parents=True)
    paths = [cfg.paths.inbox / "a.m4a", cfg.paths.inbox / "b.m4a"]
    for p in paths:
        p.write_bytes(b"audio")
    return paths


def test_会議を文字起こしして_DB_と_JSON_と_processed_に反映する(con, cfg, audio):
    repo.insert_conversation(con, make_conversation(CID, source_files=["a.m4a", "b.m4a"]))
    tr, di = FakeTranscriber(SEGMENTS), FakeDiarizer(TURNS)

    utterances = transcribe_conversation(con, cfg, CID, tr, di, now=NOW)

    # 同一話者の連続区間は結合され、話者が付く
    assert [(u.seq, u.speaker_label, u.text) for u in utterances] == [
        (0, "SPEAKER_00", "おはようございます"),
        (1, "SPEAKER_01", "資料を見ました"),
    ]
    assert repo.get_utterances(con, CID) == utterances
    # 文字起こしと話者分離に、束ねた全ファイルが順に渡る
    assert [p.name for p in tr.calls[0]] == ["a.m4a", "b.m4a"]
    assert [p.name for p in di.calls[0]] == ["a.m4a", "b.m4a"]
    # conversations の更新
    conv = repo.get_conversation(con, CID)
    assert conv.transcribed_at == NOW
    json_path = cfg.paths.transcripts / f"{CID}.json"
    assert conv.transcript_path == str(json_path)
    # JSON の内容
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["conversation_id"] == CID
    assert data["source_files"] == ["a.m4a", "b.m4a"]
    assert data["utterances"][0] == {
        "seq": 0, "speaker_label": "SPEAKER_00", "start_sec": 0.0, "end_sec": 4.0, "text": "おはようございます",
    }
    # 音声が inbox から processed に移った
    assert not any(cfg.paths.inbox.iterdir())
    assert sorted(p.name for p in cfg.paths.processed.iterdir()) == ["a.m4a", "b.m4a"]


def test_processed_にある音声も文字起こしし直せる(con, cfg, audio):
    repo.insert_conversation(con, make_conversation(CID, source_files=["a.m4a"]))
    transcribe_conversation(con, cfg, CID, FakeTranscriber(SEGMENTS), FakeDiarizer(TURNS), now=NOW)
    # 2回目: 音声は processed にある。utterances は入れ直される
    again = transcribe_conversation(con, cfg, CID, FakeTranscriber(SEGMENTS[:1]), FakeDiarizer(TURNS), now=NOW)
    assert len(again) == 1 and len(repo.get_utterances(con, CID)) == 1
    assert (cfg.paths.processed / "a.m4a").exists()


def test_登録されていない会議は例外(con, cfg):
    with pytest.raises(LookupError, match="ingest"):
        transcribe_conversation(con, cfg, "none", FakeTranscriber([]), FakeDiarizer([]))


def test_音声が見つからなければ例外_DB_は変えない(con, cfg):
    repo.insert_conversation(con, make_conversation(CID, source_files=["missing.m4a"]))
    with pytest.raises(FileNotFoundError):
        transcribe_conversation(con, cfg, CID, FakeTranscriber(SEGMENTS), FakeDiarizer(TURNS), now=NOW)
    assert repo.get_conversation(con, CID).transcribed_at is None


def test_文字起こしが失敗したら音声は_inbox_に残る(con, cfg, audio):
    class Boom:
        def transcribe(self, paths):
            raise RuntimeError("GPU が足りない")

    repo.insert_conversation(con, make_conversation(CID, source_files=["a.m4a"]))
    with pytest.raises(RuntimeError):
        transcribe_conversation(con, cfg, CID, Boom(), FakeDiarizer(TURNS), now=NOW)
    assert (cfg.paths.inbox / "a.m4a").exists()
    assert repo.get_conversation(con, CID).transcribed_at is None


def test_単発の試し実行は_DB_に書かず_音声も動かさない(cfg, audio, tmp_path):
    out = tmp_path / "out" / "a.json"
    utterances = transcribe_single_audio(audio[0], out, FakeTranscriber(SEGMENTS), FakeDiarizer(TURNS), 0.4, now=NOW)
    assert len(utterances) == 2
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["conversation_id"] is None and data["source_files"] == ["a.m4a"]
    assert audio[0].exists()
    assert not cfg.paths.db.exists()  # DuckDB のファイルは作らない


def test_find_audio_file(cfg, audio):
    assert find_audio_file(cfg, "a.m4a") == cfg.paths.inbox / "a.m4a"
    with pytest.raises(FileNotFoundError):
        find_audio_file(cfg, "zzz.m4a")
