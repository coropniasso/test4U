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


@pytest.fixture(autouse=True)
def _チャンクは25分_重複2分(cfg):
    """このファイルの期待値(チャンク1が seq 23 から始まる等)は、25分・重複2分で書いてある。

    既定値(12分・重複1分)とは独立に、チャンク分割の挙動を検査するための固定。代入は設定の検査を通らないので、
    入力が num_ctx に収まるかの検査(tests/test_config.py)には影響しない。
    """
    cfg.extract.chunk_minutes = 25
    cfg.extract.overlap_minutes = 2


@pytest.fixture
def conv(con):
    repo.insert_conversation(con, make_conversation(CID))
    repo.replace_utterances(con, CID, make_utterances(100))  # 100分 → 25分・重複2分でチャンクが複数
    return CID


def test_抽出結果が_グローバルな_seq_で_extracted_requests_に保存される(con, cfg, conv):
    extractor = FakeExtractor(
        {
            0: [make_item(3, 4, quote="a", summary="s1", reason="発話3")],  # reason は原文照合を通すため、発話の本文の一部にする
            1: [make_item(5, 6, quote="b", summary="s2")],  # チャンク1 は seq 23 から
        },
        model_name="qwen3.8-flash-next",
    )
    run = run_extraction(con, cfg, conv, extractor, now=NOW)

    rows = repo.get_requests(con, conv)
    assert [(r.seq_from, r.seq_to) for r in rows] == [(3, 4), (28, 29)]
    assert run.saved_count == 2 and run.chunk_count >= 4
    assert rows[0].request_id == f"{CID}#qwen3.8-flash-next#1"
    assert rows[1].request_id == f"{CID}#qwen3.8-flash-next#2"
    assert rows[0].model_name == "qwen3.8-flash-next"
    assert rows[0].source_type == "live_conversation"
    assert rows[0].direction is None  # speaker_map が埋まるまで確定しない
    assert rows[0].stated_reason == "発話3" and rows[1].stated_reason is None
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
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}, model_name="q4"), now=NOW)
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}, model_name="q8"), now=NOW)
    assert repo.list_request_models(con, conv) == ["q4", "q8"]
    ids = {r.request_id for r in repo.get_requests(con, conv)}
    assert ids == {f"{CID}#q4#1", f"{CID}#q8#1"}  # model_name を含むので衝突しない


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


# --- 修復の記録と入力長の検査 ------------------------------------------------------------


def test_背面が修復した記録が_extract_logs_に入る(con, cfg, conv):
    from kijun.extract.base import RepairEvent

    events = [
        RepairEvent(
            chunk_index=0, seq_from=0, seq_to=24,
            actions=["removed_prose_before"], truncated_items=1, attempt=2,
        )
    ]
    extractor = FakeExtractor({0: [make_item(1, quote="a")]}, repair_events=events)
    run = run_extraction(con, cfg, conv, extractor, now=NOW)

    assert run.repair_events == events
    data = json.loads(extract_log_path(cfg, conv, extractor.model_name).read_text(encoding="utf-8"))
    assert data["repaired_chunks"] == [
        {
            "chunk_index": 0, "seq_from": 0, "seq_to": 24,
            "actions": ["removed_prose_before"], "truncated_items": 1, "attempt": 2,
        }
    ]


def test_修復が無ければ記録は空(con, cfg, conv):
    run = run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}), now=NOW)
    assert run.repair_events == []
    data = json.loads(extract_log_path(cfg, conv, "fake-model").read_text(encoding="utf-8"))
    assert data["repaired_chunks"] == [] and data["oversize_warnings"] == []


def test_max_input_tokens_が未確認なら入力長を検査しない(con, cfg, conv, capsys):
    extractor = FakeExtractor({0: [make_item(1, quote="a")]}, max_input_tokens=None)
    run = run_extraction(con, cfg, conv, extractor, now=NOW)
    assert run.oversize_warnings == []
    assert "警告" not in capsys.readouterr().err


def test_入力が_max_input_tokens_を超えたら警告するが止めない(con, cfg, conv, capsys):
    # 上限を極端に小さくすると、どのチャンクも超える
    extractor = FakeExtractor({0: [make_item(1, quote="a")]}, max_input_tokens=10)
    run = run_extraction(con, cfg, conv, extractor, now=NOW)

    assert len(run.oversize_warnings) == run.chunk_count  # 全チャンクが超える
    assert "chunk_minutes" in run.oversize_warnings[0]
    assert "kijun doctor" in run.oversize_warnings[0]
    assert run.saved_count == 1  # 警告を出しても処理は続ける
    assert "警告" in capsys.readouterr().err
    data = json.loads(extract_log_path(cfg, conv, "fake-model").read_text(encoding="utf-8"))
    assert len(data["oversize_warnings"]) == run.chunk_count


def test_入力が上限に収まっていれば警告しない(con, cfg, conv):
    extractor = FakeExtractor({0: [make_item(1, quote="a")]}, max_input_tokens=1_000_000)
    assert run_extraction(con, cfg, conv, extractor, now=NOW).oversize_warnings == []


def test_check_chunk_input_size_は概算の係数を使う(cfg):
    from kijun.extract.chunk import split_into_chunks
    from kijun.extract.prompt import build_messages
    from kijun.extract.run import check_chunk_input_size
    from kijun.extract.tokens import estimate_messages_tokens
    from tests.conftest import make_utterances

    chunk = split_into_chunks(make_utterances(5), 25, 2)[0]
    system, user = build_messages(chunk, cfg)
    per_char = cfg.extract.tokens_per_character
    estimated = estimate_messages_tokens(system, user, per_char)
    assert estimated == int(len(system) * per_char) + int(len(user) * per_char)
    assert check_chunk_input_size(chunk, cfg, None) is None
    assert check_chunk_input_size(chunk, cfg, estimated) is None  # ちょうどなら警告しない
    assert check_chunk_input_size(chunk, cfg, estimated - 1) is not None


def test_概算の係数は設定から読む(cfg):
    from kijun.extract.chunk import split_into_chunks
    from kijun.extract.run import check_chunk_input_size
    from tests.conftest import make_utterances

    chunk = split_into_chunks(make_utterances(5), 25, 2)[0]
    cfg.extract.tokens_per_character = 0.55
    assert check_chunk_input_size(chunk, cfg, 10000) is None
    cfg.extract.tokens_per_character = 1000.0
    assert check_chunk_input_size(chunk, cfg, 10000) is not None


def test_概算の係数の既定値():
    from kijun.config import Config

    c = Config().extract
    assert c.tokens_per_character == 0.55 and c.tokens_per_meeting_minute == 180


def test_チャンクごとに開始と完了を標準エラーに出す(con, cfg, conv, capsys):
    extractor = FakeExtractor({0: [make_item(1, quote="a")]})
    extractor.last_usage = {"prompt_tokens": 2325, "completion_tokens": 13050}
    run_extraction(con, cfg, conv, extractor, now=NOW)
    err = capsys.readouterr().err
    assert "チャンク 1/" in err and "を送ります" in err and "推定" in err
    assert "完了（" in err and "入力 2325 トークン、出力 13050 トークン" in err and "抽出 1 件" in err


def test_usage_が取れなければその旨を出す(con, cfg, conv, capsys):
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}), now=NOW)
    assert "トークン数は取れず" in capsys.readouterr().err


def test_失敗の表示にチャンクの番号が入る(con, cfg, conv, capsys):
    run_extraction(con, cfg, conv, FakeExtractor({0: [make_item(1, quote="a")]}, fail_chunks={1}), now=NOW)
    err = capsys.readouterr().err
    assert "失敗: チャンク 2/" in err


def test_モデル名の_コロン_は_DB_に残し_request_id_では置き換える(con, cfg, conv):
    # Ollama のモデル名には ':' が入る。model_name の列には元の名前を入れ(compare や match が一致を見るため)、
    # request_id は使える文字(英数字と . _ + -)に揃える。run_label の規則と同じ理由。
    extractor = FakeExtractor({0: [make_item(1, quote="a")]}, model_name="qwen3.5:9b+think")
    run_extraction(con, cfg, conv, extractor, now=NOW)
    row = repo.get_requests(con, conv)[0]
    assert row.model_name == "qwen3.5:9b+think"
    assert row.request_id == f"{CID}#qwen3.5-9b+think#1"
    assert repo.get_requests(con, conv, "qwen3.5:9b+think")  # 元の名前で引ける


def test_request_id_model_は使えない文字を置き換える():
    from kijun.extract.run import request_id_model

    assert request_id_model("qwen3.5:9b") == "qwen3.5-9b"
    assert request_id_model("hf.co/org/model:Q4_K_M+nothink") == "hf.co-org-model-Q4_K_M+nothink"
    assert request_id_model("plain-name_1.0") == "plain-name_1.0"


def test_モデル名に_コロン_があっても_extract_logs_のファイル名は安全(cfg):
    path = extract_log_path(cfg, CID, "qwen3.5:9b+think")
    assert ":" not in path.name and path.name == f"{CID}__qwen3.5_9b_think.json"


def test_入力長の警告は_num_ctx_を案内する(cfg):
    from kijun.extract.chunk import split_into_chunks
    from kijun.extract.run import check_chunk_input_size

    chunk = split_into_chunks(make_utterances(5), 12, 1)[0]
    warning = check_chunk_input_size(chunk, cfg, 10)
    assert warning is not None and "num_ctx" in warning and "chunk_minutes" in warning
    assert "strata" not in warning.lower()


def test_原文に無い_stated_reason_と_applies_when_は_NULL_にして_元の値を_unverified_に残す(con, cfg, conv):
    # 発話の本文は「発話3」「発話4」…。「発話4」は原文の一部、「忙しいから」「社外向け」は原文に無い。
    extractor = FakeExtractor(
        {
            0: [
                make_item(3, 3, quote="a", summary="s1", reason="忙しいから", applies_when="発話4"),
                make_item(5, 5, quote="b", summary="s2", reason="発話5", applies_when="社外向け"),
            ],
        }
    )
    run = run_extraction(con, cfg, conv, extractor, now=NOW)

    rows = repo.get_requests(con, conv)
    assert (rows[0].stated_reason, rows[0].stated_reason_unverified) == (None, "忙しいから")
    assert (rows[0].applies_when, rows[0].applies_when_unverified) == ("発話4", None)
    assert (rows[1].stated_reason, rows[1].stated_reason_unverified) == ("発話5", None)
    assert (rows[1].applies_when, rows[1].applies_when_unverified) == (None, "社外向け")
    assert run.rejected_count("stated_reason") == 1 and run.rejected_count("applies_when") == 1
    log = json.loads(extract_log_path(cfg, conv, run.model_name).read_text(encoding="utf-8"))
    assert log["verbatim_rejected"]["stated_reason"] == [{"value": "忙しいから", "quote": "a"}]
    assert [r["value"] for r in log["verbatim_rejected"]["applies_when"]] == ["社外向け"]


def test_原文照合の原文は_seq_の範囲ではなくチャンクの全発話(con, cfg, conv):
    # モデルが返す seq は不正確なので、seq 3 の項目でも、同じチャンクの別の発話（発話10）の文字列を照合に通す。
    extractor = FakeExtractor({0: [make_item(3, 3, quote="a", summary="s", reason="発話10")]})
    run_extraction(con, cfg, conv, extractor, now=NOW)
    assert repo.get_requests(con, conv)[0].stated_reason == "発話10"
