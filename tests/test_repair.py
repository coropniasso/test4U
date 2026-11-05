"""壊れた JSON の修復（extract/repair.py）。

repair.py は外部に一切接続しないので、壊れ方の見本ごとに完全にテストできる。
確認できない背面（Ollama）の挙動を吸収する部分を、確認できる場所に寄せたのがこのモジュールである。
"""

from __future__ import annotations

import json

import pytest

from kijun.extract.repair import (
    ACTION_NO_JSON_FOUND,
    ACTION_REMOVED_CODE_FENCE,
    ACTION_REMOVED_PROSE_AFTER,
    ACTION_REMOVED_PROSE_BEFORE,
    ACTION_STRIPPED_WHITESPACE,
    ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM,
    ACTION_WRAPPED_ARRAY_IN_ITEMS,
    UnrepairableJsonError,
    parse_with_repair,
    repair_json_text,
)

# extracted_requests の全項目を持つ要素1件（models.py の ExtractedRequest に合わせる）
ITEM = {
    "seq_from": 3,
    "seq_to": 4,
    "speaker_from_label": "SPEAKER_00",
    "speaker_to_label": "SPEAKER_01",
    "counterpart_role": None,
    "artifact_type": "announcement",
    "quote": "募集要項の最初に、何をしてほしいかを書いて",
    "request_summary": "告知文の冒頭に、読み手にしてほしい行動を書くことを求めた",
    "stated_reason": None,
    "applies_when": None,
    "reusable": True,
    "pass_criterion": "冒頭2行以内に、読み手にしてほしい行動が書かれているか",
}


def item(seq: int) -> dict:
    return {**ITEM, "seq_from": seq, "seq_to": seq}


def obj(*items: dict) -> str:
    return json.dumps({"items": list(items)}, ensure_ascii=False)


# (7) 既に正常な JSON は何も変えない ------------------------------------------------------


def test_既に正常な_JSON_は何も変えない():
    text = obj(item(1), item(5))
    outcome = repair_json_text(text)
    assert outcome.text == text
    assert outcome.actions == []
    assert outcome.truncated_items == 0
    parsed = parse_with_repair(text)
    assert [i.seq_from for i in parsed.result.items] == [1, 5]
    assert parsed.repair.actions == []


# (5) 要素が0件の正常な JSON ---------------------------------------------------------------


def test_要素が0件の正常な_JSON():
    parsed = parse_with_repair('{"items": []}')
    assert parsed.result.items == []
    assert parsed.repair.actions == []
    assert parsed.repair.truncated_items == 0


def test_要素が0件で前後に空白がある():
    outcome = repair_json_text('  {"items": []}\n\n')
    assert outcome.actions == [ACTION_STRIPPED_WHITESPACE]
    assert outcome.text == '{"items": []}'


# (1) 前置きの文章つき -------------------------------------------------------------------


def test_前置きの文章つき():
    text = "はい、抽出しました。以下が結果です。\n" + obj(item(2))
    outcome = repair_json_text(text)
    assert outcome.actions == [ACTION_REMOVED_PROSE_BEFORE]
    assert json.loads(outcome.text)["items"][0]["seq_from"] == 2
    assert parse_with_repair(text).result.items[0].seq_from == 2


def test_後書きの文章つき():
    text = obj(item(2)) + "\n\n以上です。他に該当する発言はありませんでした。"
    outcome = repair_json_text(text)
    assert outcome.actions == [ACTION_REMOVED_PROSE_AFTER]
    assert parse_with_repair(text).result.items[0].seq_from == 2


def test_前置きと後書きの両方():
    text = "結果:\n" + obj(item(2)) + "\nご確認ください。"
    outcome = repair_json_text(text)
    assert outcome.actions == [ACTION_REMOVED_PROSE_BEFORE, ACTION_REMOVED_PROSE_AFTER]
    assert parse_with_repair(text).result.items[0].seq_from == 2


def test_散文に波括弧が含まれていても_最初の波括弧から切る():
    # 「{」を含む散文が前に付いた場合、最初の「{」から JSON として読もうとして失敗する。
    # 修復の限界であり、検証で落ちることを明示しておく。
    text = "出力は {このように} なります。\n" + obj(item(1))
    with pytest.raises(UnrepairableJsonError):
        parse_with_repair(text)


# (2) コードフェンスつき -----------------------------------------------------------------


def test_コードフェンスつき_json_の言語名あり():
    text = "```json\n" + obj(item(7)) + "\n```"
    outcome = repair_json_text(text)
    assert ACTION_REMOVED_CODE_FENCE in outcome.actions
    assert parse_with_repair(text).result.items[0].seq_from == 7


def test_コードフェンスつき_言語名なし():
    text = "```\n" + obj(item(7)) + "\n```"
    assert parse_with_repair(text).result.items[0].seq_from == 7


def test_コードフェンスの前に散文がある場合は中身を取り出せない():
    # 散文が先にあるとフェンスの除去は働かないが、波括弧より前を捨てる処理で読める
    text = "結果です。\n```json\n" + obj(item(7)) + "\n```"
    outcome = repair_json_text(text)
    assert ACTION_REMOVED_PROSE_BEFORE in outcome.actions
    assert parse_with_repair(text).result.items[0].seq_from == 7


def test_閉じるフェンスが無い場合も中身を取り出す():
    text = "```json\n" + obj(item(7))
    assert parse_with_repair(text).result.items[0].seq_from == 7


# (3) トップレベルが配列 ------------------------------------------------------------------


def test_トップレベルが配列なら_items_に包む():
    text = json.dumps([item(1), item(2)], ensure_ascii=False)
    outcome = repair_json_text(text)
    assert outcome.actions == [ACTION_WRAPPED_ARRAY_IN_ITEMS]
    assert json.loads(outcome.text)["items"][1]["seq_from"] == 2
    assert [i.seq_from for i in parse_with_repair(text).result.items] == [1, 2]


def test_トップレベルが空の配列():
    assert parse_with_repair("[]").result.items == []


def test_トップレベルが配列で_コードフェンスと前置きつき():
    text = "抽出結果:\n```json\n" + json.dumps([item(9)], ensure_ascii=False) + "\n```"
    parsed = parse_with_repair(text)
    assert parsed.result.items[0].seq_from == 9
    assert ACTION_WRAPPED_ARRAY_IN_ITEMS in parsed.repair.actions


# (4) 末尾が切れている -------------------------------------------------------------------


def test_末尾が切れている_最後の完全な要素までで閉じる():
    full = obj(item(1), item(2))
    text = full[: full.index('{"seq_from": 2') + 40]  # 2件目の途中で切る
    outcome = repair_json_text(text)
    assert ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM in outcome.actions
    assert outcome.truncated_items == 1
    parsed = parse_with_repair(text)
    assert [i.seq_from for i in parsed.result.items] == [1]  # 1件目だけ残る
    assert parsed.repair.truncated_items == 1


def test_末尾が切れている_カンマまでで切れていれば捨てる要素は0件():
    full = obj(item(1), item(2))
    text = full[: full.index('{"seq_from": 2')]  # 「..., 」までで切る
    outcome = repair_json_text(text)
    assert ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM in outcome.actions
    assert outcome.truncated_items == 0
    assert [i.seq_from for i in parse_with_repair(text).result.items] == [1]


def test_末尾が切れている_配列のまま切れた場合も_items_に包む():
    full = json.dumps([item(1), item(2)], ensure_ascii=False)
    text = full[: full.index('{"seq_from": 2') + 40]
    parsed = parse_with_repair(text)
    assert [i.seq_from for i in parsed.result.items] == [1]
    assert ACTION_WRAPPED_ARRAY_IN_ITEMS in parsed.repair.actions
    assert ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM in parsed.repair.actions


def test_末尾が切れている_3件のうち2件が完全なら2件残る():
    full = obj(item(1), item(2), item(3))
    text = full[: full.index('{"seq_from": 3') + 30]
    parsed = parse_with_repair(text)
    assert [i.seq_from for i in parsed.result.items] == [1, 2]
    assert parsed.repair.truncated_items == 1


def test_完全な要素が1件も無いまま切れていれば修復しない():
    # 空の items として閉じてしまうと「抽出0件」と区別が付かず、出力が切れて
    # データを失ったことが呼び出し側から見えなくなる。検証で落として失敗として記録させる。
    text = '{"items": [{"seq_from": 1, "quote": "途中で'
    outcome = repair_json_text(text)
    assert ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM not in outcome.actions
    with pytest.raises(UnrepairableJsonError):
        parse_with_repair(text)


def test_文字列の中の波括弧は要素の終わりと数えない():
    tricky = {**item(1), "quote": "括弧 } と ] を含む引用"}
    text = obj(tricky, item(2))
    assert repair_json_text(text).actions == []
    assert [i.seq_from for i in parse_with_repair(text).result.items] == [1, 2]


def test_引用に二重引用符のエスケープが含まれていても読める():
    tricky = {**item(1), "quote": 'これは「\\"引用\\"」です'}
    text = obj(tricky)
    assert repair_json_text(text).actions == []
    assert parse_with_repair(text).result.items[0].seq_from == 1


# (6) 修復しても読めない -----------------------------------------------------------------


def test_JSON_が1文字も無ければ修復できない():
    outcome = repair_json_text("申し訳ありませんが、抽出できませんでした。")
    assert outcome.actions == [ACTION_NO_JSON_FOUND]
    with pytest.raises(UnrepairableJsonError, match="読めませんでした"):
        parse_with_repair("申し訳ありませんが、抽出できませんでした。")


def test_空文字列は修復できない():
    with pytest.raises(UnrepairableJsonError):
        parse_with_repair("")


def test_スキーマに合わない_JSON_は修復できない():
    with pytest.raises(UnrepairableJsonError):
        parse_with_repair('{"items": [{"seq_from": 1}]}')  # 必須項目が足りない


def test_items_以外のキーだけの_JSON_は修復できない():
    with pytest.raises(UnrepairableJsonError):
        parse_with_repair('{"results": []}')


def test_修復できなかったときの例外に_行った修復と内容の先頭が入る():
    with pytest.raises(UnrepairableJsonError) as e:
        parse_with_repair("前置き\n```json\n{\"results\": []}\n```")
    msg = str(e.value)
    assert "コードフェンス" in msg or "散文" in msg  # 行った修復が日本語で入る
    assert "results" in msg  # 読めなかった内容の先頭が入る


def test_actions_は日本語の説明に変換できる():
    outcome = repair_json_text("前置き\n" + obj(item(1)))
    assert outcome.descriptions == ["JSON より前の散文を捨てた"]
