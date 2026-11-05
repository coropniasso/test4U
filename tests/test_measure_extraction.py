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
from tests.fakes import ollama_body


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


def test_variant_は指定した値だけを差し替え_元の設定を変えない():
    cfg = Config()
    cfg.extract.chunk_minutes = 7
    v = m.Variant("x", model="qwen3:14b", think=False, num_ctx=20000, num_predict=4096)
    applied = v.applied_to(cfg)
    o = applied.extract.ollama
    assert (o.model, o.think, o.num_ctx, o.num_predict) == ("qwen3:14b", False, 20000, 4096)
    # 設定の違い以外が揃っていることが測定の前提
    assert applied.extract.chunk_minutes == 7
    assert o.base_url == cfg.extract.ollama.base_url and o.keep_alive == cfg.extract.ollama.keep_alive
    # timeout_sec は 0 なら設定ファイルの値をそのまま使う
    assert o.timeout_sec == cfg.extract.ollama.timeout_sec
    # 元の Config は書き換えない（次の variant に漏れると比較が崩れる）
    assert (cfg.extract.ollama.model, cfg.extract.ollama.think, cfg.extract.ollama.num_ctx) == ("qwen3.5:9b", None, 16384)


def test_何も指定しない_variant_は_config_のまま():
    cfg = Config()
    cfg.extract.ollama.think = True
    assert m.Variant("x").applied_to(cfg) == cfg


def test_think_は_False_も上書きとして扱う():
    # think = false と「指定なし」を混同すると、think を切る測定が config の値のままで走る。
    cfg = Config()
    cfg.extract.ollama.think = True
    assert m.Variant("x", think=False).applied_to(cfg).extract.ollama.think is False


def test_timeout_sec_は正の値のときだけ上書きする():
    cfg = Config()
    cfg.extract.ollama.timeout_sec = 2400
    assert m.Variant("x", timeout_sec=3600).applied_to(cfg).extract.ollama.timeout_sec == 3600
    assert m.Variant("x").applied_to(cfg).extract.ollama.timeout_sec == 2400


def test_上書きの結果が入力が収まらない設定なら拒否する():
    # 入力が切り捨てられた結果を品質として並べないため、測る前に検査する。
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="収まりません"):
        m.Variant("x", num_ctx=9000).applied_to(Config())
    with pytest.raises(ValidationError, match="num_predict"):
        m.Variant("x", num_predict=100).applied_to(Config())


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


def test_variants_を省略すると_config_のままの1通り():
    assert [v.label for v in m.load_variants(None)] == ["config"]


def test_同梱の_variants_example_toml_が読める():
    path = Path(__file__).resolve().parents[1] / "scripts" / "variants.example.toml"
    variants = m.load_variants(path)
    assert [(v.label, v.model, v.think) for v in variants] == [
        ("q35-9b-think", "qwen3.5:9b", True),
        ("q35-9b-nothink", "qwen3.5:9b", False),
        ("q3-14b-think", "qwen3:14b", True),
        ("q3-14b-nothink", "qwen3:14b", False),
    ]
    # どの variant も、既定の設定の検査に通る
    for v in variants:
        v.applied_to(Config())


# --- 1設定の実行 -----------------------------------------------------------------


class _Transport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = []
        self.urls = []

    def get(self, url, timeout_sec):
        raise AssertionError("使わない")

    def post(self, url, payload, timeout_sec):
        self.urls.append(url)
        self.posts.append(payload)
        return self._responses.pop(0)


def _utterances():
    return [
        Utterance(seq=i, speaker_label="SPEAKER_00", start_sec=i * 2.0, end_sec=i * 2.0 + 1.5, text=t)
        for i, t in enumerate(["結論を先に書いて", "過剰書きにして"])
    ]


def _body(items, done_reason="stop", eval_count=100, prompt_eval_count=50, thinking=None):
    return HttpResponse(
        status_code=200,
        text=ollama_body(
            json.dumps({"items": items}), done_reason=done_reason, eval_count=eval_count,
            prompt_eval_count=prompt_eval_count, thinking=thinking,
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
    from kijun.extract.ollama_impl import build_payload

    cfg = Config()
    chunks = split_into_chunks(_utterances(), cfg.extract.chunk_minutes, cfg.extract.overlap_minutes)
    v = m.Variant("x", model="qwen3:14b", think=True, num_ctx=20000, num_predict=4096)
    transport = _Transport([_body([_item()])])
    m.run_variant(cfg, v, chunks, transport)
    sent = transport.posts[0]
    assert transport.urls[0] == "http://127.0.0.1:11434/api/chat"
    assert sent["model"] == "qwen3:14b" and sent["think"] is True
    assert sent["options"]["num_ctx"] == 20000 and sent["options"]["num_predict"] == 4096
    # build_payload をそのまま使っているので、本番が送るものと一字も違わない
    assert sent == build_payload(v.applied_to(cfg), chunks[0])


def test_HTTP_エラーは記録して次のチャンクへ進む():
    cfg = Config()
    chunks = split_into_chunks(_utterances(), cfg.extract.chunk_minutes, cfg.extract.overlap_minutes)
    transport = _Transport([HttpResponse(status_code=400, text="does not support thinking", elapsed_sec=0.1)])
    outcomes = m.run_variant(cfg, m.Variant("x", think=True), chunks, transport)
    assert outcomes[0].error is not None and "400" in outcomes[0].error
    assert outcomes[0].items == []


def test_入力が切り捨てられた疑いのあるチャンクは失敗として記録する():
    # 切り捨てられた入力から出た結果を並べると、発話が欠けた結果を品質として比べてしまう。
    cfg = Config()
    chunks = split_into_chunks(_utterances(), cfg.extract.chunk_minutes, cfg.extract.overlap_minutes)
    limit = cfg.extract.ollama.num_ctx - cfg.extract.ollama.num_predict
    transport = _Transport([_body([_item()], prompt_eval_count=limit)])
    outcomes = m.run_variant(cfg, m.Variant("x"), chunks, transport)
    assert outcomes[0].error is not None and "先頭を捨てる" in outcomes[0].error and outcomes[0].items == []


def test_推論の文字数と_done_reason_を記録する():
    cfg = Config()
    chunks = split_into_chunks(_utterances(), cfg.extract.chunk_minutes, cfg.extract.overlap_minutes)
    transport = _Transport([_body([_item()], done_reason="length", thinking="あ" * 50)])
    out = m.run_variant(cfg, m.Variant("x", think=True), chunks, transport)[0]
    assert out.done_reason == "length" and out.thinking_chars == 50
    assert (out.prompt_tokens, out.completion_tokens) == (50, 100) and len(out.items) == 1


def test_まとめは非_null_の数と文字起こしに無い語を数える():
    transcript = TRANSCRIPT
    outcomes = [
        m.ChunkOutcome(
            0, 0, 1,
            elapsed_sec=10.0,
            completion_tokens=100,
            prompt_tokens=50,
            done_reason="stop",
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
    assert s["done_reasons"] == ["stop"]


# --- 人が確認する2点 -------------------------------------------------------------


def test_人が確認する2点を該当する項目として取り出す():
    items = [
        _item(quote="3つ目簡単フォーイスなんか減らしてください", transcription_unclear=True, reusable=False),
        _item(quote="日時と場所を見出しのすぐ下に", applies_when="参加者が募りたい場合"),
        _item(quote="結論を先に", applies_when="社外向け"),
    ]
    checks = m.human_checks(items)
    assert checks["unclear_items"] == [
        {
            "quote": "3つ目簡単フォーイスなんか減らしてください",
            "transcription_unclear": True, "pass_criterion": None, "reusable": False,
        }
    ]
    assert checks["applies_when_hits"] == ["参加者が募りたい場合"]


def test_該当する項目が無ければ空():
    assert m.human_checks([_item()]) == {"unclear_items": [], "applies_when_hits": []}


# --- 対照表 -------------------------------------------------------------------------


def _results(variants, items=None):
    return {
        v.label: m.summarize(
            [m.ChunkOutcome(0, 0, 1, elapsed_sec=1.0, completion_tokens=5, done_reason="stop",
                            items=items or [_item(request_summary="フォントサイズ")])],
            TRANSCRIPT,
        )
        for v in variants
    }


def test_対照表に設定と結果と無い語が並ぶ():
    variants = [m.Variant("a", think=True), m.Variant("b", think=False)]
    md = m.build_markdown("20261103_1930_f2f_teirei", variants, _results(variants), Config())
    assert "| a | b |" in md
    assert "| a | qwen3.5:9b（config） | true |" in md and "| b | qwen3.5:9b（config） | false |" in md
    assert "フォントサイズ" in md
    # すべてが誤りではないと明記していること（人が判断する前提を外さないため）
    assert "すべてが誤りではない" in md


def test_同じ会議なら_Strata_の基準を並べる():
    variants = [m.Variant("a")]
    md = m.build_markdown(m.BASELINE_CONVERSATION, variants, _results(variants), Config())
    assert "Strata xhigh" in md and "Strata medium" in md
    # 基準の値: 抽出件数 6、stated_reason 4/4、applies_when 2/1、pass_criterion 4/4、文字起こしに無い語 5/17
    assert "| 抽出件数 | 1 | 6 | 6 |" in md
    assert "| applies_when が非 null | 0 | 2 | 1 |" in md
    assert "| 文字起こしに無い語の数 | 1 | 5 | 17 |" in md
    assert "どれを採るかは人が決める" in md


def test_別の会議には基準を並べない():
    variants = [m.Variant("a")]
    md = m.build_markdown("20260920_1930_f2f_teirei", variants, _results(variants), Config())
    assert "Strata xhigh" not in md and "比べられない" in md


def test_人が確認する2点が対照表に出る():
    variants = [m.Variant("a")]
    items = [
        _item(quote="3つ目簡単フォーイス", transcription_unclear=True, reusable=False),
        _item(applies_when="参加者が募りたい場合"),
    ]
    md = m.build_markdown(m.BASELINE_CONVERSATION, variants, _results(variants, items), Config())
    assert "transcription_unclear = True" in md and "reusable = False" in md
    assert "参加者が募りたい場合" in md
    empty = m.build_markdown(m.BASELINE_CONVERSATION, variants, _results(variants, [_item()]), Config())
    assert "該当する項目がない" in empty


def test_基準の値は_CLAUDE_md_7章の記録と一致する():
    # 基準を書き換えるときに、CLAUDE.md の記録との食い違いに気付くための検査。
    text = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text(encoding="utf-8")
    assert "| `xhigh`、`max_tokens = 48000`、予算40000を明示 | 36,000 | 21,201 | 22,408 | 409秒 | 6 | 4/6 | **2/6** | 4/6 |" in text
    assert m.STRATA_BASELINE["Strata xhigh"] == {
        "item_count": 6, "stated_reason": 4, "applies_when": 2, "pass_criterion": 4, "novel": 5,
    }
    assert m.STRATA_BASELINE["Strata medium"]["applies_when"] == 1
