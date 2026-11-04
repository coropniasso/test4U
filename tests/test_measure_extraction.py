"""scripts/measure_extraction.py の検査。

この測定スクリプトの結論は「文字起こしに無い語が出力に入っていたか」で決まる。
その判定が誤っていると測定そのものが無意味になるので、検出の部分に検査を掛ける。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from kijun.config import Config
from kijun.extract.chunk import split_into_chunks
from kijun.models import Utterance
from kijun.transport import HttpResponse


def _load_module():
    """scripts/ はパッケージではないので、パスから読み込む。"""
    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_extraction.py"
    spec = importlib.util.spec_from_file_location("measure_extraction", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # exec_module の前に登録する。dataclass のデコレータが sys.modules[cls.__module__] を引くため、
    # 登録前に実行すると、モジュール内の @dataclass が AttributeError で落ちる。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m = _load_module()

TRANSCRIPT = "3つ目簡単フォーイスなんか減らしてください\n日時と場所を見出しのすぐ下に過剰書きで書いて"


def test_文字起こしに無いカタカナ語を拾う():
    # 実機で medium が作った語。文字起こしに「フォントサイズ」は無い。
    assert m.novel_terms("フォントサイズを小さくすることを求めた", TRANSCRIPT) == ["フォントサイズ"]


def test_文字起こしにあるカタカナ語は拾わない():
    assert m.novel_terms("「簡単フォーイス」を減らすことを求めた", TRANSCRIPT) == []


def test_文字起こしに無い漢語を拾う():
    # 「箇条書き」は文字起こしでは「過剰書き」に化けている。当てはめれば検出されるべき。
    assert "箇条書" in m.novel_terms("箇条書きで書くことを求めた", TRANSCRIPT)


def test_1文字の漢字は拾わない():
    # 「求めた」「書く」のような要約の言い回しで必ず出るので、拾うと一覧が読めなくなる。
    assert m.novel_terms("書くことを求めた", TRANSCRIPT) == []


def test_値が_null_なら何も返さない():
    assert m.novel_terms(None, TRANSCRIPT) == []
    assert m.novel_terms("", TRANSCRIPT) == []


def test_同じ語は1回だけ返す():
    assert m.novel_terms("フォントサイズとフォントサイズ", TRANSCRIPT) == ["フォントサイズ"]


def test_内容語の取り出しは出た順で重複を除く():
    assert m.content_terms("アンケートと日時、アンケート") == ["アンケート", "日時"]


# --- variant の適用 -------------------------------------------------------------


def test_variant_は3つの値だけを差し替え_元の設定を変えない():
    cfg = Config()
    cfg.extract.chunk_minutes = 7
    v = m.Variant("x", reasoning_effort="medium", reasoning_budget_tokens=8000, max_tokens=30000)
    applied = v.applied_to(cfg)
    assert applied.extract.strata.reasoning_effort == "medium"
    assert applied.extract.strata.reasoning_budget_tokens == 8000
    assert applied.extract.strata.max_tokens == 30000
    # 設定の違い以外が揃っていることが測定の前提
    assert applied.extract.chunk_minutes == 7
    # 元の Config は書き換えない（次の variant に漏れると比較が崩れる）
    assert cfg.extract.strata.reasoning_effort == ""
    assert cfg.extract.strata.max_tokens == 20000


def test_label_が重複する一覧は拒否する(tmp_path):
    p = tmp_path / "v.toml"
    p.write_text('[[variant]]\nlabel = "a"\n[[variant]]\nlabel = "a"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="重複"):
        m.load_variants(p)


def test_variant_が空の一覧は拒否する(tmp_path):
    p = tmp_path / "v.toml"
    p.write_text("# 何も無い\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="variant"):
        m.load_variants(p)


def test_同梱の_variants_example_toml_が読める():
    path = Path(__file__).resolve().parents[1] / "scripts" / "variants.example.toml"
    variants = m.load_variants(path)
    assert [v.label for v in variants] == ["medium-mt20k", "xhigh-rb40k", "default-mt20k"]
    # 推論の予算を明示する variant は、答えの分を max_tokens に残していること
    rb = next(v for v in variants if v.reasoning_budget_tokens)
    assert rb.max_tokens > rb.reasoning_budget_tokens


# --- 1設定の実行 -----------------------------------------------------------------


class _Transport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = []

    def get(self, url, timeout_sec):
        raise AssertionError("使わない")

    def post(self, url, payload, timeout_sec):
        self.posts.append(payload)
        return self._responses.pop(0)


def _utterances():
    return [
        Utterance(seq=i, speaker_label="SPEAKER_00", start_sec=i * 2.0, end_sec=i * 2.0 + 1.5, text=t)
        for i, t in enumerate(["結論を先に書いて", "過剰書きにして"])
    ]


def _body(items, finish_reason="stop", completion_tokens=100):
    return HttpResponse(
        status_code=200,
        text=json.dumps(
            {
                "choices": [{"message": {"content": json.dumps({"items": items})}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 50, "completion_tokens": completion_tokens},
            }
        ),
        elapsed_sec=1.0,
    )


def _item(**over):
    base = {
        "seq_from": 0,
        "seq_to": 0,
        "speaker_from_label": "SPEAKER_00",
        "speaker_to_label": None,
        "counterpart_role": None,
        "artifact_type": None,
        "quote": "結論を先に書いて",
        "request_summary": "結論を先に書くことを求めた",
        "stated_reason": None,
        "applies_when": None,
        "reusable": True,
        "pass_criterion": None,
    }
    base.update(over)
    return base


def test_リクエストには_variant_の値が入り_本番と同じ組み立てになる():
    from kijun.extract.strata_impl import build_payload

    cfg = Config()
    chunks = split_into_chunks(_utterances(), cfg.extract.chunk_minutes, cfg.extract.overlap_minutes)
    v = m.Variant("x", reasoning_effort="medium", reasoning_budget_tokens=5000, max_tokens=20000)
    transport = _Transport([_body([_item()])])
    m.run_variant(cfg, v, chunks, transport)
    sent = transport.posts[0]
    assert sent["reasoning_effort"] == "medium"
    assert sent["reasoning_budget_tokens"] == 5000
    assert sent["max_tokens"] == 20000
    # build_payload をそのまま使っているので、本番が送るものと一字も違わない
    assert sent == build_payload(v.applied_to(cfg), chunks[0], cfg.extract.strata.enforces_json_schema)


def test_HTTP_エラーは記録して次のチャンクへ進む():
    cfg = Config()
    chunks = split_into_chunks(_utterances(), cfg.extract.chunk_minutes, cfg.extract.overlap_minutes)
    transport = _Transport([HttpResponse(status_code=400, text="unknown reasoning effort", elapsed_sec=0.1)])
    outcomes = m.run_variant(cfg, m.Variant("x"), chunks, transport)
    assert outcomes[0].error is not None and "400" in outcomes[0].error
    assert outcomes[0].items == []


def test_まとめは非_null_の数と文字起こしに無い語を数える():
    transcript = TRANSCRIPT
    outcomes = [
        m.ChunkOutcome(
            0, 0, 1,
            elapsed_sec=10.0,
            completion_tokens=100,
            prompt_tokens=50,
            finish_reason="stop",
            items=[
                _item(stated_reason="読む人は忙しいから"),
                _item(request_summary="フォントサイズを小さくする", reusable=False),
            ],
        )
    ]
    s = m.summarize(outcomes, transcript)
    assert s["item_count"] == 2
    assert s["reusable_true"] == 1
    assert s["non_null"]["stated_reason"] == 1
    assert s["non_null"]["applies_when"] == 0
    assert "フォントサイズ" in s["novel_terms"]["request_summary"]
    assert s["novel_term_count"] >= 1


def test_対照表に設定と結果と無い語が並ぶ():
    variants = [m.Variant("a", reasoning_effort="medium"), m.Variant("b", reasoning_effort="xhigh")]
    results = {
        v.label: m.summarize(
            [m.ChunkOutcome(0, 0, 1, elapsed_sec=1.0, completion_tokens=5, finish_reason="stop",
                            items=[_item(request_summary="フォントサイズ")])],
            TRANSCRIPT,
        )
        for v in variants
    }
    md = m.build_markdown("20261103_1930_f2f_teirei", variants, results)
    assert "| a | b |" in md
    assert "medium" in md and "xhigh" in md
    assert "フォントサイズ" in md
    # すべてが誤りではないと明記していること（人が判断する前提を外さないため）
    assert "すべてが誤りではない" in md
