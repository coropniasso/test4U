from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kijun.config import Config
from kijun.extract.chunk import split_into_chunks
from kijun.extract.models import ExtractedRequest, ExtractionResult
from kijun.extract.prompt import SYSTEM_PROMPT, build_messages
from tests.conftest import make_utterances

VALID = {
    "seq_from": 3,
    "seq_to": 4,
    "speaker_from_label": "SPEAKER_00",
    "speaker_to_label": "SPEAKER_01",
    "counterpart_role": "後輩",
    "artifact_type": "announcement",
    "quote": "募集要項の最初に、何をしてほしいかを書いて",
    "request_summary": "告知文の冒頭に、読み手にしてほしい行動を書くことを求めた",
    "stated_reason": None,
    "applies_when": None,
    "reusable": True,
    "pass_criterion": "冒頭2行以内に、読み手にしてほしい行動が書かれているか",
    "transcription_unclear": False,
}


def test_正しい_JSON_を受け取れる():
    result = ExtractionResult.model_validate_json(json.dumps({"items": [VALID]}))
    assert len(result.items) == 1
    assert result.items[0].stated_reason is None
    assert result.items[0].pass_criterion.endswith("か")


def test_該当なしは空の配列():
    assert ExtractionResult.model_validate_json('{"items": []}').items == []


def test_理由と適用条件は_null_を許し_文字列も受け取る():
    item = ExtractedRequest.model_validate({**VALID, "stated_reason": "読む人は忙しいから"})
    assert item.stated_reason == "読む人は忙しいから"
    assert ExtractedRequest.model_validate(VALID).applies_when is None


@pytest.mark.parametrize("missing", ["quote", "request_summary", "reusable", "seq_from", "stated_reason"])
def test_必須の項目が欠けていれば拒否する(missing):
    broken = {k: v for k, v in VALID.items() if k != missing}
    with pytest.raises(ValidationError):
        ExtractedRequest.model_validate(broken)


def test_トップレベルが配列だと拒否する():
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate_json(json.dumps([VALID]))


def test_壊れた_JSON_は拒否する():
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate_json("申し訳ありませんが…")


def test_JSON_スキーマのトップレベルは_items_を持つオブジェクト():
    schema = ExtractionResult.model_json_schema()
    assert schema["type"] == "object"
    assert schema["properties"]["items"]["type"] == "array"
    item_props = schema["$defs"]["ExtractedRequest"]["properties"]
    assert set(item_props) == set(VALID)  # extracted_requests の列に対応する項目だけ


# --- プロンプト -------------------------------------------------------------------------


def a_chunk():
    return split_into_chunks(make_utterances(5), 25, 2)[0]


def test_build_messages_は_system_と_user_を返す():
    system, user = build_messages(a_chunk(), Config())
    assert system == SYSTEM_PROMPT
    assert "[0] SPEAKER_00: 発話0" in user
    assert "[4] SPEAKER_00: 発話4" in user
    assert "0 から 4 まで" in user


def test_発話番号はチャンク内の番号で振る():
    chunk = split_into_chunks(make_utterances(100), 25, 2)[1]  # グローバル seq 23 から
    _, user = build_messages(chunk, Config())
    assert "[0] " in user and "発話23" in user.splitlines()[2]  # 先頭の発話がチャンク内の [0]


def test_話者ラベルが無い発話は_UNKNOWN():
    chunk = a_chunk()
    from dataclasses import replace

    chunk = replace(chunk, utterances=[replace(chunk.utterances[0], speaker_label=None)])
    assert "[0] UNKNOWN: " in build_messages(chunk, Config())[1]


def test_同じ入力なら同じ文面になり_system_はチャンクによらず固定():
    c = split_into_chunks(make_utterances(100), 25, 2)
    assert build_messages(c[0], Config()) == build_messages(c[0], Config())
    assert build_messages(c[0], Config())[0] == build_messages(c[1], Config())[0]


def test_プロンプトは_理由と適用条件を推測で補わないことを強く求める():
    s = SYSTEM_PROMPT
    assert "stated_reason" in s and "applies_when" in s
    assert "語られた内容だけ" in s
    assert "推測" in s
    assert "null" in s
    # 理由（目的）も書いてある
    assert "言葉にする機会が失われ" in s
    # 悪い例が示されている
    assert "悪い例" in s


def test_プロンプトは_pass_criterion_を_Yes_No_で判定できる形にすることを求める():
    s = SYSTEM_PROMPT
    assert "Yes か No で判定できる" in s
    assert "冒頭2行以内に、読み手にしてほしい行動が書かれているか" in s  # CLAUDE.md の例


def test_プロンプトは_JSON_だけを返し_items_のオブジェクトで返すことを書いている():
    s = SYSTEM_PROMPT
    assert "JSON だけを返して" in s
    assert "前置き" in s
    assert '{"items"' in s
    assert "トップレベルを配列にしません" in s


def test_プロンプトは_extracted_requests_の項目を全部説明している():
    for name in VALID:
        assert name in SYSTEM_PROMPT, name


def test_推測で補わない規則は_6項目すべてを名指ししている():
    # 名指しが stated_reason と applies_when の2項目だけだったとき、reasoning_effort を下げた
    # 実機の実行で、規則が名指ししていない request_summary と pass_criterion に発話に無い語が
    # 入った（2026-11-03）。規則の射程を文面で確認する。
    s = SYSTEM_PROMPT
    head = s.split("# 最も重要な規則", 1)[1].split("# pass_criterion", 1)[0]
    for name in [
        "stated_reason",
        "applies_when",
        "request_summary",
        "pass_criterion",
        "counterpart_role",
        "artifact_type",
    ]:
        assert name in head, name
    assert "どの項目にも書かない" in s


def test_プロンプトは_文字起こしが壊れた発話の扱いを指示している():
    s = SYSTEM_PROMPT
    assert "文字起こしが壊れている発話の扱い" in s
    # 入力が機械の文字起こしであり、誤認識を含むことを述べている
    assert "聞き取りの誤り" in s
    # 音や文脈が近い語を当てはめることを禁じている
    assert "当てはめて書くことは" in s
    # 壊れていても項目を落とさない（本人が読めば直せるため）
    assert "items から省いてはいけません" in s
    # 試験対象の語（「感嘆符」→「簡単フォーイス」）を文面に書いていない。
    # 書くと、その語の正解を教えることになり、実機での比較が成立しない。
    assert "簡単フォーイス" not in s and "感嘆符" not in s


def test_プロンプトは_request_summary_に注釈を付けさせない():
    # pass_criterion が無いとき request_summary がそのまま check_text になる（match/link.py）。
    # 「（意味が取れない）」のような注釈を書かせると、それがチェック項目の文言として Discord に出る。
    # 実機で「（「過剰書き」は意味が取れない）」が request_summary に入ったので指示を外した（2026-11-03）。
    s = SYSTEM_PROMPT
    assert "のような注釈は付けない" in s
    assert "チェック項目として読めなくなる" in s


def test_プロンプトは_stated_reason_と_applies_when_を原文のまま写すことを求める():
    s = SYSTEM_PROMPT
    assert "一字一句そのまま写" in s
    assert "プログラムが" in s and "捨てられ" in s
    assert "言い換えない" in s
    # 別の分野の例（料理）を使う。過去に、例の文（読む人は忙しいから）をモデルがそのまま写した。
    assert "甘みが出るから" in s
    assert "読む人は忙しいから" not in s
    # 言い換えが誤りである悪い例が残っている
    assert "甘みを引き出すため" in s
