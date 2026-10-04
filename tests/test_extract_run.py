from __future__ import annotations

import json
from datetime import datetime

import pytest

from kijun.db import repo
from kijun.extract.base import ExtractionError
from kijun.extract.run import extract_log_path, read_failed_chunk_count, run_extraction
from tests.conftest import make_conversation, make_utterances
from tests.fakes import FakeExtractor, make_item

CID = "20260920_1930_f2f_teirei"
NOW = datetime(2026, 9, 21, 0, 0)


@pytest.fixture
def conv(con):
    repo.insert_conversation(con, make_conversation(CID))
    repo.replace_utterances(con, CID, make_utterances(100))  # 100分 → 25分・重複2分でチャンクが複数
    return CID


def test_抽出結果が_グローバルな_seq_で_extracted_requests_に保存される(con, cfg, conv):
    extractor = FakeExtractor(
        {
            0: [make_item(3, 4, quote="a", summary="s1", reason="理由あり")],
            1: [make_item(5, 6, quote="b", summary="s2")],  # チャンク1 は seq 23 から
        },
        model_name="qwen3:14b",
    )
    run = run_extraction(con, cfg, conv, extractor, now=NOW)

    rows = repo.get_requests(con, conv)
    assert [(r.seq_from, r.seq_to) for r in rows] == [(3, 4), (28, 29)]
    assert run.saved_count == 2 and run.chunk_count >= 4
    assert rows[0].request_id == f"{CID}#qwen3:14b#1"
    assert rows[1].request_id == f"{CID}#qwen3:14b#2"
    assert rows[0].model_name == "qwen3:14b"
    assert rows[0].source_type == "live_conversation"
    assert rows[0].direction is None  # speaker_map が埋まるまで確定しない
    assert rows[0].stated_reason == "理由あり" and rows[1].stated_reason is None
    assert rows[0].extracted_at == NOW


def test_重複区間の結果は1件にまとまる(con, cfg, conv):
    extractor = FakeExtractor(
        {0: [make_item(23, quote="結論を先に")], 1: [make_item(0, quote="結論を先に")]}
    )
    run = run_extraction(con, cfg, conv, extractor, now=NOW)
    assert run.saved_count == 1
    assert run.outcome.overlap_dropped_count == 1


def test_失敗したチャンクは記録して次に進む(con, cfg, conv, capsys):
    extractor = FakeExtractor({0: [make_item(1, quote="a")], 2: [make_item(1, quote="c")]}, fail_chunks={1})
    run = run_extraction(con, cfg, conv, extractor, now=NOW)

    assert run.saved_count == 2  # 失敗したチャンクがあっても、他のチャンクの結果は保存する
    assert [f.chunk_index for f in run.failures] == [1]
    f = run.failures[0]
    assert f.seq_from == 23  # 失敗したチャンクの発話の範囲
    assert f"発話 {f.seq_from}〜{f.seq_to}" in capsys.readouterr().err
    # 比較表のために、失敗したチャンク数がファイルに残る
    log = extract_log_path(cfg, conv, extractor.model_name)
    assert read_failed_chunk_count(log) == 1
    data = json.loads(log.read_text(encoding="utf-8"))
    assert data["failed_chunks"][0]["seq_from"] == f.seq_from and data["saved_count"] == 2


def test_全チャンクが失敗したら例外にし_既存の結果を消さない(con, cfg, conv):
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}), now=NOW)
    all_fail = FakeExtractor(fail_chunks=set(range(100)))
    with pytest.raises(ExtractionError, match="すべてのチャンク"):
        run_extraction(con, cfg, conv, all_fail, now=NOW)
    assert len(repo.get_requests(con, conv)) == 1  # 前回の結果は残っている


def test_同じモデルで再抽出すると置き換える(con, cfg, conv):
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a"), make_item(2, quote="b")]}), now=NOW)
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(5, quote="c")]}), now=NOW)
    assert [r.seq_from for r in repo.get_requests(con, conv)] == [5]


def test_モデルが違えば別々に保存される(con, cfg, conv):
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}, model_name="qwen"), now=NOW)
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}, model_name="claude"), now=NOW)
    assert repo.list_request_models(con, conv) == ["claude", "qwen"]
    ids = {r.request_id for r in repo.get_requests(con, conv)}
    assert ids == {f"{CID}#qwen#1", f"{CID}#claude#1"}  # model_name を含むので衝突しない


def test_item_evidence_に紐付いた結果がある場合は置き換えない(con, cfg, conv):
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}), now=NOW)
    rid = repo.get_requests(con, conv)[0].request_id
    repo.add_evidence(con, "item_0001", rid, None, NOW)
    with pytest.raises(RuntimeError, match="item_evidence"):
        run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(2, quote="b")]}), now=NOW)
    assert [r.seq_from for r in repo.get_requests(con, conv)] == [1]


def test_範囲外の番号は捨てる(con, cfg, conv):
    run = run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(999, quote="a"), make_item(1, quote="b")]}), now=NOW)
    assert run.saved_count == 1 and run.outcome.invalid_count == 1


def test_発話が無い会議は例外(con, cfg):
    repo.insert_conversation(con, make_conversation("20260101_0900_call_zatsudan"))
    with pytest.raises(LookupError, match="transcribe"):
        run_extraction(con, cfg, "20260101_0900_call_zatsudan", FakeExtractor(), now=NOW)


def test_失敗の記録が無ければ_None(tmp_path):
    assert read_failed_chunk_count(tmp_path / "none.json") is None
