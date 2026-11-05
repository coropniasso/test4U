"""壊れた JSON の修復（llm/repair.py）。

repair.py は外部に一切接続しないので、壊れ方の見本ごとに完全にテストできる。
確認できない背面（Ollama）の挙動を吸収する部分を、確認できる場所に寄せたのがこのモジュールである。
"""

from __future__ import annotations

import json

import pytest

from kijun.llm.repair import (
    ACTION_NO_JSON_FOUND,
    ACTION_REMOVED_CODE_FENCE,
    ACTION_REMOVED_PROSE_AFTER,
    ACTION_REMOVED_PROSE_BEFORE,
    ACTION_STRIPPED_WHITESPACE,
    UnrepairableJsonError,
    parse_model_with_repair,
    repair_json_text,
)
from kijun.minutes.models import ChunkMinutes, Overview

PAYLOAD = {
    "topics": [{"title": "日程", "points": ["来週の水曜日に決めた"]}],
    "decisions": ["定例は水曜日の午後"],
    "todos": [{"task": "資料を用意する", "owner": None, "due": "金曜日"}],
    "open_issues": [],
}
GOOD = json.dumps(PAYLOAD, ensure_ascii=False)


def test_既に正常な_JSON_は何も変えない():
    outcome = repair_json_text(GOOD)
    assert outcome.text == GOOD and outcome.actions == []
    parsed, _ = parse_model_with_repair(GOOD, ChunkMinutes)
    assert parsed.todos[0].owner is None and parsed.todos[0].due == "金曜日"


def test_前後の空白を除く():
    outcome = repair_json_text(f"\n  {GOOD}  \n")
    assert outcome.text == GOOD and ACTION_STRIPPED_WHITESPACE in outcome.actions


def test_前置きの文章つき():
    outcome = repair_json_text(f"はい、まとめました。\n{GOOD}")
    assert outcome.text == GOOD and ACTION_REMOVED_PROSE_BEFORE in outcome.actions


def test_後書きの文章つき():
    outcome = repair_json_text(f"{GOOD}\n以上です。{{補足}}")
    assert outcome.text == GOOD and ACTION_REMOVED_PROSE_AFTER in outcome.actions


def test_コードフェンスつき_言語名あり():
    outcome = repair_json_text(f"```json\n{GOOD}\n```")
    assert outcome.text == GOOD and ACTION_REMOVED_CODE_FENCE in outcome.actions


def test_コードフェンスつき_言語名なし():
    assert repair_json_text(f"```\n{GOOD}\n```").text == GOOD


def test_閉じるフェンスが無い場合も中身を取り出す():
    assert repair_json_text(f"```json\n{GOOD}").text == GOOD


def test_文字列の中の波括弧はオブジェクトの終わりと数えない():
    payload = {"overview": "波括弧 } を含む {文}。"}
    text = json.dumps(payload, ensure_ascii=False) + "\n後書き"
    parsed, outcome = parse_model_with_repair(text, Overview)
    assert parsed.overview == payload["overview"] and ACTION_REMOVED_PROSE_AFTER in outcome.actions


def test_二重引用符のエスケープが含まれていても読める():
    payload = {"overview": 'A さんは "確認する" と言った。'}
    parsed, _ = parse_model_with_repair(json.dumps(payload, ensure_ascii=False), Overview)
    assert parsed.overview == payload["overview"]


def test_途中で切れた_JSON_は修復しない_後ろの節が黙って空になるのを避ける():
    cut = GOOD[: GOOD.index('"decisions"')]
    with pytest.raises(UnrepairableJsonError):
        parse_model_with_repair(cut, ChunkMinutes)


def test_JSON_が1文字も無ければ修復できない():
    outcome = repair_json_text("すみません、できません")
    assert ACTION_NO_JSON_FOUND in outcome.actions
    with pytest.raises(UnrepairableJsonError):
        parse_model_with_repair("すみません、できません", ChunkMinutes)


def test_空文字列は修復できない():
    with pytest.raises(UnrepairableJsonError):
        parse_model_with_repair("", Overview)


def test_スキーマに合わない_JSON_は修復できない():
    with pytest.raises(UnrepairableJsonError) as e:
        parse_model_with_repair('{"overview": 3}', Overview)
    assert "Overview" in str(e.value)


def test_actions_は日本語の説明に変換できる():
    outcome = repair_json_text(f"前置き\n{GOOD}")
    assert outcome.descriptions == ["JSON より前の散文を捨てた"]
