from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kijun.cli import app
from kijun.db import migrate, repo
from kijun.models import Utterance
from kijun.transcribe.base import SpeakerTurn, TranscribedSegment
from kijun.transcribe.restore import (
    REQUIRED_KEYS,
    STATUS_ERROR,
    STATUS_OVERWRITTEN,
    STATUS_RESTORED,
    STATUS_SKIPPED_EXISTS,
    STATUS_SKIPPED_SINGLE,
    parse_transcript,
    restore_transcript,
    restore_transcripts,
)
from kijun.transcribe.run import transcribe_conversation, transcribe_single_audio
from tests.conftest import make_conversation
from tests.fakes import FakeDiarizer, FakeTranscriber

CID = "20260920_1930_f2f_teirei"


def write_json(cfg, cid: str = CID, **overrides) -> Path:
    data = {
        "conversation_id": cid,
        "recorded_at": "2026-09-20T19:30:00",
        "setting_format": "f2f",
        "setting_kind": "teirei",
        "source_files": ["a.m4a", "b.m4a"],
        "duration_sec": 3600.5,
        "created_at": "2026-09-20T23:30:00",
        "transcribed_at": "2026-09-21T00:10:00",
        "utterances": [
            {"seq": 0, "speaker_label": "SPEAKER_00", "start_sec": 0.0, "end_sec": 4.0, "text": "おはようございます"},
            {"seq": 1, "speaker_label": None, "start_sec": 5.0, "end_sec": 7.0, "text": "資料を見ました"},
            {"seq": 2, "speaker_label": "SPEAKER_01", "start_sec": 8.0, "end_sec": 9.5, "text": "はい"},
        ],
    }
    data.update(overrides)
    cfg.paths.transcripts.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.transcripts / f"{cid}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# 1. 空の DB に復元する -------------------------------------------------------------------


def test_空の_DB_に_conversations_と_utterances_が復元される(con, cfg):
    path = write_json(cfg)
    outcome = restore_transcript(con, path)

    assert (outcome.status, outcome.conversation_id, outcome.utterance_count) == (STATUS_RESTORED, CID, 3)
    conv = repo.get_conversation(con, CID)
    assert conv.recorded_at == datetime(2026, 9, 20, 19, 30)
    assert (conv.setting_format, conv.setting_kind) == ("f2f", "teirei")
    assert conv.source_files == ["a.m4a", "b.m4a"]
    assert conv.duration_sec == 3600.5
    assert conv.created_at == datetime(2026, 9, 20, 23, 30)
    assert conv.transcribed_at == datetime(2026, 9, 21, 0, 10)
    assert conv.transcript_path == str(path)  # 読み込んだ JSON 自身のパス
    assert [(u.seq, u.speaker_label, u.text) for u in repo.get_utterances(con, CID)] == [
        (0, "SPEAKER_00", "おはようございます"), (1, None, "資料を見ました"), (2, "SPEAKER_01", "はい"),
    ]


def test_duration_sec_が_null_でも復元できる(con, cfg):
    restore_transcript(con, write_json(cfg, duration_sec=None))
    assert repo.get_conversation(con, CID).duration_sec is None


def test_dry_run_は何も書かない(con, cfg):
    outcome = restore_transcript(con, write_json(cfg), dry_run=True)
    assert outcome.status == STATUS_RESTORED and outcome.utterance_count == 3
    assert repo.get_conversation(con, CID) is None
    assert repo.get_utterances(con, CID) == []


# 2. 既にある会議はスキップ -------------------------------------------------------------


def existing(con):
    repo.insert_conversation(con, make_conversation(CID, source_files=["keep.m4a"]))
    repo.replace_utterances(con, CID, [Utterance(0, "SPEAKER_09", 0.0, 1.0, "既にある発話")])


def test_同じ会議が既にあればスキップし_utterances_も変えない(con, cfg):
    existing(con)
    outcome = restore_transcript(con, write_json(cfg))
    assert outcome.status == STATUS_SKIPPED_EXISTS
    assert "--force" in outcome.message
    assert repo.get_conversation(con, CID).source_files == ["keep.m4a"]
    assert [u.text for u in repo.get_utterances(con, CID)] == ["既にある発話"]


# 3. --force ----------------------------------------------------------------------------


def test_force_を付けると上書きする(con, cfg):
    existing(con)
    outcome = restore_transcript(con, write_json(cfg), force=True)
    assert outcome.status == STATUS_OVERWRITTEN
    assert repo.get_conversation(con, CID).source_files == ["a.m4a", "b.m4a"]
    assert [u.text for u in repo.get_utterances(con, CID)] == ["おはようございます", "資料を見ました", "はい"]


# 4. --all ------------------------------------------------------------------------------


def test_all_で複数の会議が復元される(con, cfg):
    write_json(cfg, "20260920_1930_f2f_teirei")
    write_json(cfg, "20260921_1000_call_sagyou", setting_format="call", setting_kind="sagyou")
    write_json(cfg, "20260922_0900_f2f_zatsudan")
    outcomes = restore_transcripts(con, cfg)
    assert [o.status for o in outcomes] == [STATUS_RESTORED] * 3
    assert [c.conversation_id for c in repo.list_conversations(con)] == [
        "20260920_1930_f2f_teirei", "20260921_1000_call_sagyou", "20260922_0900_f2f_zatsudan",
    ]
    assert repo.get_conversation(con, "20260921_1000_call_sagyou").setting_format == "call"


def test_all_は既にある会議だけをスキップし_残りは復元する(con, cfg):
    existing(con)
    write_json(cfg, CID)
    write_json(cfg, "20260921_1000_call_sagyou")
    outcomes = {o.conversation_id: o.status for o in restore_transcripts(con, cfg)}
    assert outcomes == {CID: STATUS_SKIPPED_EXISTS, "20260921_1000_call_sagyou": STATUS_RESTORED}


def test_all_は単発の試し実行の_JSON_をスキップし_壊れた_JSON_があっても他を復元する(con, cfg):
    write_json(cfg)
    single = cfg.paths.transcripts / "sample.json"
    transcribe_single_audio(Path("sample.m4a"), single, FakeTranscriber([]), FakeDiarizer([]), 0.4)
    (cfg.paths.transcripts / "broken.json").write_text("{ 壊れている", encoding="utf-8")
    outcomes = {o.path.name: o.status for o in restore_transcripts(con, cfg)}
    assert outcomes == {
        f"{CID}.json": STATUS_RESTORED, "sample.json": STATUS_SKIPPED_SINGLE, "broken.json": STATUS_ERROR,
    }


def test_会議を指定して_JSON_が無ければ例外(con, cfg):
    cfg.paths.transcripts.mkdir(parents=True)
    with pytest.raises(LookupError, match="見つかりません"):
        restore_transcripts(con, cfg, "none")


def test_conversation_を指定すればその会議だけ復元する(con, cfg):
    write_json(cfg, CID)
    write_json(cfg, "20260921_1000_call_sagyou")
    outcomes = restore_transcripts(con, cfg, CID)
    assert [o.conversation_id for o in outcomes] == [CID]
    assert len(repo.list_conversations(con)) == 1


# 5. 項目の欠け -------------------------------------------------------------------------


@pytest.mark.parametrize("key", [k for k in REQUIRED_KEYS if k != "conversation_id"])
def test_必要な項目が欠けていれば会議IDと項目名を出す(con, cfg, key):
    path = write_json(cfg)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data[key]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    outcome = restore_transcript(con, path)
    assert outcome.status == STATUS_ERROR
    assert CID in outcome.message and key in outcome.message
    assert repo.get_conversation(con, CID) is None  # 何も書かない


def test_複数の項目が欠けていればすべて挙げる(con, cfg):
    path = write_json(cfg)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["recorded_at"], data["setting_kind"]
    path.write_text(json.dumps(data), encoding="utf-8")
    msg = restore_transcript(con, path).message
    assert "recorded_at" in msg and "setting_kind" in msg


def test_発話の項目が欠けていればその位置と項目名を出す(con, cfg):
    path = write_json(cfg, utterances=[{"seq": 0, "speaker_label": None, "start_sec": 0.0, "end_sec": 1.0}])
    msg = restore_transcript(con, path).message
    assert CID in msg and "utterances[0]" in msg and "text" in msg


def test_日時が不正ならエラー(con, cfg):
    msg = restore_transcript(con, write_json(cfg, recorded_at="昨日")).message
    assert CID in msg and "不正" in msg


def test_壊れた_JSON_はエラー(con, cfg):
    cfg.paths.transcripts.mkdir(parents=True)
    path = cfg.paths.transcripts / "x.json"
    path.write_text("not json", encoding="utf-8")
    outcome = restore_transcript(con, path)
    assert outcome.status == STATUS_ERROR and "x.json" in outcome.message


# 6. transcribe が書いた JSON を読み戻す往復 ---------------------------------------------------------


SEGMENTS = [
    TranscribedSegment(0.0, 2.0, "おはよう"),
    TranscribedSegment(2.2, 4.0, "ございます"),
    TranscribedSegment(5.0, 7.0, "資料を見ました"),
    TranscribedSegment(9.0, 10.0, "誰の発言か不明"),  # 話者区間の外 → speaker_label が None
]
TURNS = [SpeakerTurn(0.0, 4.5, "SPEAKER_00"), SpeakerTurn(4.5, 8.0, "SPEAKER_01")]


def test_transcribe_が書いた_JSON_を_restore_で読み戻すと_DB_の内容が元と一致する(con, cfg, tmp_path):
    cfg.paths.inbox.mkdir(parents=True)
    for n in ("a.m4a", "b.m4a"):
        (cfg.paths.inbox / n).write_bytes(b"audio")
    created = datetime(2026, 9, 20, 23, 30, 15, 123456)  # マイクロ秒つきでも往復できる
    conv = make_conversation(CID, source_files=["a.m4a", "b.m4a"], duration_sec=1234.5678)
    repo.insert_conversation(con, type(conv)(**{**conv.__dict__, "created_at": created}))
    now = datetime(2026, 9, 21, 0, 10, 5, 987654)

    transcribe_conversation(con, cfg, CID, FakeTranscriber(SEGMENTS), FakeDiarizer(TURNS), now=now)
    original_conv = repo.get_conversation(con, CID)
    original_utts = repo.get_utterances(con, CID)
    assert any(u.speaker_label is None for u in original_utts)  # None を含む場合も確かめる

    # DuckDB を失った状態を作る（別の空の DB に接続し直す）
    fresh = repo.connect(tmp_path / "fresh.duckdb")
    migrate.apply_schema(fresh)
    outcomes = restore_transcripts(fresh, cfg)

    assert [(o.conversation_id, o.status) for o in outcomes] == [(CID, STATUS_RESTORED)]
    assert repo.get_utterances(fresh, CID) == original_utts  # 発話が一字一句・一秒まで一致
    assert repo.get_conversation(fresh, CID) == original_conv  # conversations の全列が一致
    fresh.close()


def test_JSON_は_conversations_の再構成に必要な項目をすべて含む(con, cfg):
    cfg.paths.inbox.mkdir(parents=True)
    (cfg.paths.inbox / f"{CID}.m4a").write_bytes(b"audio")
    repo.insert_conversation(con, make_conversation(CID))
    transcribe_conversation(con, cfg, CID, FakeTranscriber(SEGMENTS), FakeDiarizer(TURNS))
    data = json.loads((cfg.paths.transcripts / f"{CID}.json").read_text(encoding="utf-8"))
    assert set(REQUIRED_KEYS) <= set(data)
    assert parse_transcript(cfg.paths.transcripts / f"{CID}.json") is not None


# CLI ------------------------------------------------------------------------------------------

runner = CliRunner()


def cli_config(tmp_path: Path) -> list[str]:
    p = tmp_path / "config.toml"
    p.write_text(
        f"""
[paths]
inbox = "{(tmp_path / 'inbox').as_posix()}"
processed = "{(tmp_path / 'processed').as_posix()}"
transcripts = "{(tmp_path / 'transcripts').as_posix()}"
db = "{(tmp_path / 'data' / 'kijun.duckdb').as_posix()}"
""",
        encoding="utf-8",
    )
    return ["--config", str(p)]


def test_CLI_dry_run_と本実行_と_force(tmp_path):
    from kijun.config import load_config

    base = cli_config(tmp_path)
    cfg = load_config(base[1])
    write_json(cfg)
    assert runner.invoke(app, [*base, "db", "init"]).exit_code == 0

    r = runner.invoke(app, [*base, "restore-transcript", "--all", "--dry-run"])
    assert r.exit_code == 0 and "復元します（発話 3 件）" in r.output and "書いていません" in r.output
    r = runner.invoke(app, [*base, "restore-transcript", "--conversation", CID])
    assert r.exit_code == 0 and "復元しました（発話 3 件）" in r.output
    assert "kijun extract" in r.output and "kijun match" in r.output  # やり直しが必要なことを出す
    r = runner.invoke(app, [*base, "restore-transcript", "--all"])
    assert "既にあるのでスキップ" in r.output
    r = runner.invoke(app, [*base, "restore-transcript", "--all", "--force"])
    assert "上書きしました" in r.output


def test_CLI_conversation_と_all_のどちらか一方を要求する(tmp_path):
    base = cli_config(tmp_path)
    assert runner.invoke(app, [*base, "restore-transcript"]).exit_code == 1
    assert runner.invoke(app, [*base, "restore-transcript", "--all", "--conversation", "x"]).exit_code == 1


def test_CLI_壊れた_JSON_があれば終了コード_1(tmp_path):
    from kijun.config import load_config

    base = cli_config(tmp_path)
    cfg = load_config(base[1])
    path = write_json(cfg)
    path.write_text("{}", encoding="utf-8")  # conversation_id も無い
    runner.invoke(app, [*base, "db", "init"])
    r = runner.invoke(app, [*base, "restore-transcript", "--all"])
    assert r.exit_code == 1 and "必要な項目がありません" in r.output
