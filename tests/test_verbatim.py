"""stated_reason / applies_when の原文照合（extract/verbatim.py）の検査。"""

from __future__ import annotations

from kijun.extract.verbatim import check_item, is_verbatim, normalize, rejections, source_text
from tests.conftest import make_utterances
from tests.fakes import make_item


def test_normalize_は句読点と空白と記号を取り除き_小文字にする():
    assert normalize("甘みが、出る から。") == "甘みが出るから"
    assert normalize("A　B\nC\t！？「」（）") == "abc"
    assert normalize("A+B=C") == "abc"  # + と = は記号(S)


def test_normalize_は全角と半角をそろえる():
    assert normalize("ＡＢＣ１２３") == "abc123"
    assert normalize("ﾊﾝｶｸ") == "ハンカク"


def test_is_verbatim_は連続した一部なら真():
    src = "玉ねぎは先に炒めて。甘みが出るから、お願いします。"
    assert is_verbatim("甘みが出るから", src)
    assert is_verbatim("甘みが 出るから。", src)  # 句読点・空白の違いは無視
    assert is_verbatim("玉ねぎは先に炒めて甘みが出るから", src)  # 句読点をまたぐ連続


def test_言い換えは偽():
    assert not is_verbatim("甘みを引き出すため", "玉ねぎは先に炒めて。甘みが出るから")
    assert not is_verbatim("甘みが出るので", "玉ねぎは先に炒めて。甘みが出るから")


def test_語順を変えたものや_飛び飛びの抜き出しは偽():
    assert not is_verbatim("出るから甘みが", "甘みが出るから")
    assert not is_verbatim("玉ねぎ甘み", "玉ねぎは先に炒めて。甘みが出るから")


def test_文字起こしの誤りは_誤りのまま写してあれば真():
    assert is_verbatim("過剰書きにして", "ここは過剰書きにして、お願いします")
    assert not is_verbatim("箇条書きにして", "ここは過剰書きにして、お願いします")


def test_空の文字列や記号だけの文字列は偽():
    assert not is_verbatim("", "何か")
    assert not is_verbatim("、。 ", "何か、。")
    assert not is_verbatim("何か", "")


def test_発話を区切りなしでつないだ原文に対して_改行や発話の境目をまたぐ値も真():
    utterances = make_utterances(3)  # 発話0 発話1 発話2
    src = source_text(utterances)
    assert src == "発話0発話1発話2"
    assert is_verbatim("発話1\n発話2", src)
    assert is_verbatim("発話0 発話1", src)


def test_check_item_は外れた値を_NULL_にして元の値を残す():
    item = make_item(0, quote="q", reason="忙しいから", applies_when="発話1")
    got = check_item(item, "発話0発話1")
    assert got.stated_reason is None and got.stated_reason_unverified == "忙しいから"
    assert got.applies_when == "発話1" and got.applies_when_unverified is None
    assert got.quote == "q"  # 他の項目は変えない


def test_check_item_は_NULL_の値には何もしない():
    got = check_item(make_item(0), "発話0")
    assert got.stated_reason is None and got.stated_reason_unverified is None
    assert got.applies_when is None and got.applies_when_unverified is None


def test_rejections_は外した値を項目名と引用つきで返す():
    items = [
        check_item(make_item(0, quote="q1", reason="x", applies_when="y"), "発話"),
        check_item(make_item(1, quote="q2", reason="発話"), "発話"),
    ]
    got = rejections(items)
    assert [(r.field, r.value, r.quote) for r in got] == [
        ("stated_reason", "x", "q1"),
        ("applies_when", "y", "q1"),
    ]


def test_モデルに渡す_JSON_スキーマに_unverified_の項目は入らない():
    from kijun.extract.models import ExtractionResult

    assert "unverified" not in str(ExtractionResult.model_json_schema())
