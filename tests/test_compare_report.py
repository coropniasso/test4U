from __future__ import annotations

from datetime import datetime

import pytest

from kijun.compare.report import (
    build_report,
    compute_stats,
    pair_requests,
    quote_ratio,
    select_models,
)
from kijun.db import repo
from kijun.extract.run import ExtractionRun, extract_log_path, write_extract_log
from kijun.extract.run import ChunkFailure
from tests.conftest import make_conversation

CID = "20260920_1930_f2f_teirei"
NOW = datetime(2026, 9, 21)


def row(request_id, model, seq_from, quote, reason=None, applies=None, criterion=None, reusable=True, summary="要約"):
    return repo.RequestRow(
        request_id=request_id, conversation_id=CID, seq_from=seq_from, seq_to=seq_from,
        speaker_from_label="SPEAKER_00", speaker_to_label="SPEAKER_01", direction=None,
        counterpart_role=None, artifact_type=None, quote=quote, request_summary=summary,
        stated_reason=reason, applies_when=applies, reusable=reusable, pass_criterion=criterion,
        source_type="live_conversation", model_name=model, extracted_at=NOW,
    )


QWEN, CLAUDE = "qwen3:14b", "claude-sonnet-5-5"


def test_統計の計算():
    rows = [
        row("1", QWEN, 1, "a", reason="r", criterion="c"),
        row("2", QWEN, 5, "b", applies="x", reusable=False),
        row("3", QWEN, 9, "c"),
    ]
    s = compute_stats(QWEN, rows, failed_chunks=2)
    assert (s.total, s.reusable, s.reason_filled, s.applies_when_filled, s.pass_criterion_filled, s.failed_chunks) == (
        3, 2, 1, 1, 1, 2,
    )
    assert s.reusable_ratio == pytest.approx(2 / 3)
    assert compute_stats(QWEN, [], None).reusable_ratio is None


def test_対応付け_seq_from_が近く_quote_が似ているものを組にする():
    a = [row("a1", QWEN, 10, "募集要項の最初に、何をしてほしいかを書いて"), row("a2", QWEN, 50, "グラフに単位を付けて")]
    b = [
        row("b1", CLAUDE, 11, "募集要項の最初に何をしてほしいかを書いて"),  # seq の差 1、quote はほぼ同じ
        row("b2", CLAUDE, 80, "全然違う発言です"),
    ]
    p = pair_requests(a, b)
    assert [(x.request_id, y.request_id) for x, y in p.matched] == [("a1", "b1")]
    assert [r.request_id for r in p.only_a] == ["a2"]
    assert [r.request_id for r in p.only_b] == ["b2"]


def test_対応付け_seq_from_の差が3以上なら_quote_が同じでも組にしない():
    p = pair_requests([row("a", QWEN, 10, "同じ発言")], [row("b", CLAUDE, 13, "同じ発言")])
    assert p.matched == [] and len(p.only_a) == 1 and len(p.only_b) == 1


def test_対応付け_差がちょうど2なら組にする():
    p = pair_requests([row("a", QWEN, 10, "同じ発言です")], [row("b", CLAUDE, 12, "同じ発言です")])
    assert len(p.matched) == 1


def test_対応付け_quote_の類似度が0_6未満なら組にしない():
    assert quote_ratio("あいうえおかきくけこ", "さしすせそたちつてと") < 0.6
    p = pair_requests([row("a", QWEN, 10, "あいうえおかきくけこ")], [row("b", CLAUDE, 10, "さしすせそたちつてと")])
    assert p.matched == []


def test_対応付けは1対1():
    a = [row("a1", QWEN, 10, "同じ発言です"), row("a2", QWEN, 10, "同じ発言です")]
    b = [row("b1", CLAUDE, 10, "同じ発言です")]
    p = pair_requests(a, b)
    assert len(p.matched) == 1 and len(p.only_a) == 1 and p.only_b == []


def test_モデルの選択(cfg):
    assert select_models([CLAUDE, QWEN, "x"], cfg) == [QWEN, CLAUDE]  # Qwen、Claude の順
    assert select_models(["x", "y"], cfg) == ["x", "y"]
    assert select_models([QWEN, CLAUDE], cfg, ["a", "b"]) == ["a", "b"]


@pytest.fixture
def two_models(con, cfg):
    repo.insert_conversation(con, make_conversation(CID))
    repo.insert_requests(
        con,
        [
            row("q1", QWEN, 10, "募集要項の最初に、何をしてほしいかを書いて", reason="忙しいから", criterion="冒頭2行以内か"),
            row("q2", QWEN, 30, "誤字を直して", reusable=False),
            row("c1", CLAUDE, 10, "募集要項の最初に、何をしてほしいかを書いて", criterion="冒頭に行動があるか"),
            row("c2", CLAUDE, 60, "グラフに単位を付けて|必ず", summary="改行\nあり"),
        ],
    )
    # Qwen では1チャンクが失敗していた
    write_extract_log(
        extract_log_path(cfg, CID, QWEN), CID,
        ExtractionRun(QWEN, 4, [ChunkFailure(1, 23, 47, "壊れた JSON")]), NOW,
    )


def test_比較表の_Markdown(con, cfg, two_models):
    md = build_report(con, cfg, CID)
    assert md.startswith(f"# 抽出結果の比較: {CID}")
    assert "承認件数は、レビューして初めて決まる数字なので、この表には出ていません" in md
    # 数字の表（列は Qwen、Claude の順）
    assert f"| 項目 | {QWEN} | {CLAUDE} |" in md
    assert "| 抽出件数 | 2 | 2 |" in md
    assert "| reusable = true の件数 | 1 | 2 |" in md
    assert "| reusable = true の割合 | 50% | 100% |" in md
    assert "| stated_reason が非 NULL の件数 | 1 | 0 |" in md
    assert "| pass_criterion が非 NULL の件数 | 1 | 1 |" in md
    assert "| 失敗したチャンク数 | 1 | 記録なし |" in md
    # 対応付け: q1 と c1 が組、q2 は Qwen だけ、c2 は Claude だけ
    assert "| 両方が抽出 | 1 |" in md
    assert f"| {QWEN} だけ | 1 |" in md and f"| {CLAUDE} だけ | 1 |" in md
    # 対照表に両方の出力が左右に並ぶ
    assert "## 3. レビュー用の対照表" in md
    table_lines = [l for l in md.splitlines() if l.startswith("| 10 |")]
    assert len(table_lines) == 1
    assert "冒頭2行以内か" in table_lines[0] and "冒頭に行動があるか" in table_lines[0]
    assert "（抽出なし）" in md
    # セル内の縦線と改行はエスケープされ、表が崩れない
    assert "\\|必ず" in md and "改行<br>あり" in md
    # seq_from の昇順
    seqs = [int(l.split("|")[1]) for l in md.splitlines() if l.startswith("| ") and l.split("|")[1].strip().isdigit()]
    assert seqs == sorted(seqs)
    assert "## 4. 承認件数" in md


def test_モデルが1つだけなら対照表は出さない(con, cfg, two_models):
    md = build_report(con, cfg, CID, models=[QWEN])
    assert "対照表は出しません" in md
    assert "| 抽出件数 | 2 |" in md


def test_抽出結果が無ければ例外(con, cfg):
    repo.insert_conversation(con, make_conversation(CID))
    with pytest.raises(LookupError, match="extract"):
        build_report(con, cfg, CID)


def test_存在しないモデルを指定すれば例外(con, cfg, two_models):
    with pytest.raises(LookupError, match="nothing"):
        build_report(con, cfg, CID, models=[QWEN, "nothing"])
