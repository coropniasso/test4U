"""投稿する文面(仕様 2-3、テスト 1)。"""

from __future__ import annotations

from datetime import datetime

from kijun.db.repo import KnowledgeItemRow, RequestRow
from kijun.discord import format as fmt
from kijun.discord.base import normalize_emoji
from tests.conftest import make_conversation


def item(**kw) -> KnowledgeItemRow:
    base = dict(
        item_id="item_0142", artifact_type="announcement",
        check_text="告知文の冒頭2行以内に、読み手にしてほしい行動が書かれているか",
        pass_criterion="冒頭2行以内に行動が書かれているか", reason=None, applies_when=None,
        not_applies_when=None, importance=2, status="candidate", version=1,
        updated_at=datetime(2026, 11, 3),
    )
    base.update(kw)
    return KnowledgeItemRow(**base)


def request(**kw) -> RequestRow:
    base = dict(
        request_id="r1", conversation_id="20260920_1930_f2f_teirei", seq_from=1, seq_to=1,
        speaker_from_label="SPEAKER_00", speaker_to_label="SPEAKER_01", direction=None,
        counterpart_role=None, artifact_type="announcement",
        quote="募集要項の最初に、何をしてほしいかを書いて", request_summary="冒頭に行動を書く",
        stated_reason=None, applies_when=None, reusable=True, pass_criterion=None,
        source_type="live_conversation", model_name="m", extracted_at=datetime(2026, 11, 3),
    )
    base.update(kw)
    return RequestRow(**base)


CONV = make_conversation()


def test_理由と適用条件が_NULL_のときは_記録なし_と書く():
    text = fmt.format_new_candidate(item(), request(), CONV, {})
    assert "理由：記録なし" in text and "適用条件：記録なし" in text
    assert "理由：\n" not in text  # 空欄にしない


def test_語られた理由と適用条件はそのまま出る():
    text = fmt.format_new_candidate(item(reason="読む人は忙しいから", applies_when="社外向け"), request(), CONV, {})
    assert "理由：読む人は忙しいから" in text and "適用条件：社外向け" in text


def test_CLAUDE_md_の例と同じ形になる():
    text = fmt.format_new_candidate(item(), request(), CONV, {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"})
    lines = text.split("\n")
    assert lines[0] == "【新規候補 #0142】重要度案：2"
    assert lines[1] == "場面：9/20 定例"
    assert lines[2] == "発言：「募集要項の最初に、何をしてほしいかを書いて」（あなた→kouhai_a）"
    assert lines[3].startswith("チェック項目案：告知文の冒頭2行以内に")
    assert lines[-1] == "→ ✅❌⏸でstatus、1️⃣2️⃣3️⃣で重要度。理由・条件・修正はスレッドに返信"


def test_speaker_map_が未確定なら話者ラベルのまま出る():
    text = fmt.format_new_candidate(item(), request(), CONV, {})
    assert "（SPEAKER_00→SPEAKER_01）" in text
    assert "あなた" not in text


def test_一部だけ確定していれば確定した側だけ人物になる():
    text = fmt.format_new_candidate(item(), request(), CONV, {"SPEAKER_00": "self"})
    assert "（あなた→SPEAKER_01）" in text


def test_言われた側が不明なら_不明_と出す():
    text = fmt.format_new_candidate(item(), request(speaker_to_label=None), CONV, {"SPEAKER_00": "self"})
    assert "（あなた→不明）" in text


def test_重要度が判定できなかったときは_なし_と出す():
    assert "重要度案：なし" in fmt.format_new_candidate(item(importance=None), request(), CONV, {})


def test_種類の日本語表示と未知の種類():
    assert "場面：9/20 作業" in fmt.format_new_candidate(item(), request(), make_conversation_kind("sagyou"), {})
    assert "場面：9/20 retro" in fmt.format_new_candidate(item(), request(), make_conversation_kind("retro"), {})


def make_conversation_kind(kind: str):
    from dataclasses import replace

    return replace(CONV, setting_kind=kind)


def test_2000文字を超えたら_quote_だけが末尾から切り詰められる():
    long_quote = "あ" * 3000
    check = "チェック項目" * 20
    text = fmt.format_new_candidate(
        item(check_text=check, reason="理由" * 30, applies_when="条件" * 30),
        request(quote=long_quote), CONV, {},
    )
    assert len(text) <= fmt.DISCORD_MESSAGE_LIMIT
    assert f"チェック項目案：{check}" in text  # 切り詰められない
    assert "理由：" + "理由" * 30 in text and "適用条件：" + "条件" * 30 in text
    quote_line = next(line for line in text.split("\n") if line.startswith("発言："))
    assert "…" in quote_line and quote_line.startswith("発言：「あああ")
    assert len(text) == fmt.DISCORD_MESSAGE_LIMIT  # 限界まで残す


def test_2000文字以内なら切り詰めない():
    text = fmt.format_new_candidate(item(), request(quote="あ" * 100), CONV, {})
    assert "…" not in text and "あ" * 100 in text


def test_quote_を消しても収まらない場合は収まらないまま返す():
    # 黙ってチェック項目案を削るより、呼び出し側が検出して投稿しない方が安全
    text = fmt.format_new_candidate(item(check_text="あ" * 2500), request(), CONV, {})
    assert len(text) > fmt.DISCORD_MESSAGE_LIMIT
    assert "あ" * 2500 in text


def test_話者サンプルは文字数の多い順に選ぶ():
    utterances = [
        ("SPEAKER_00", "はい"), ("SPEAKER_01", "そうですね"), ("SPEAKER_00", "募集要項の最初に、何をしてほしいかを書いて"),
        ("SPEAKER_00", "それは"), ("SPEAKER_00", "冒頭の挨拶を短くして、結論を先に書いてください"), (None, "ラベルなし"),
        ("SPEAKER_00", "ちょっと長めの発言です"),
    ]
    got = fmt.pick_speaker_samples(utterances, 3)
    assert list(got) == ["SPEAKER_00", "SPEAKER_01"]
    assert got["SPEAKER_00"] == [
        "冒頭の挨拶を短くして、結論を先に書いてください", "募集要項の最初に、何をしてほしいかを書いて", "ちょっと長めの発言です",
    ]
    assert got["SPEAKER_01"] == ["そうですね"]


def test_話者サンプルの文面は構造化形式の例を必ず含む():
    text = fmt.format_speaker_sample("20260920_1930_f2f_teirei", {"SPEAKER_00": ["発言A"], "SPEAKER_01": ["発言B"]})
    assert text.startswith("【話者の対応付け】20260920_1930_f2f_teirei")
    assert "00=self 01=kouhai_a" in text
    assert "自分を必ず self と書いてください" in text


def test_話者サンプルは長い発言を縮めて2000文字に収める():
    samples = {f"SPEAKER_0{i}": ["あ" * 800] * 3 for i in range(3)}
    text = fmt.format_speaker_sample("c", samples)
    assert len(text) <= fmt.DISCORD_MESSAGE_LIMIT
    assert "00=self" in text and text.count("発言サンプル") == 3


def test_矛盾の文面は両方の項目を切り詰めずに出す():
    a = item(item_id="item_0003", reason="理由A")
    b = item(item_id="item_0007", check_text="別の文言", status="approved")
    text = fmt.format_contradiction(a, b, 0.78)
    assert "#0003 と #0007" in text and "類似度 0.78" in text
    assert "理由：理由A" in text and "理由：記録なし" in text
    assert a.check_text in text and "別の文言" in text
    assert "A か B" in text


def test_束ね結果の文面():
    conv = make_conversation(source_files=["a.m4a", "b.m4a"], duration_sec=2700.0)
    text = fmt.format_bundle_result(conv)
    assert "2 個のファイル" in text and "45.0 分" in text and "- a.m4a" in text and "gap_minutes" in text


def test_異体字セレクタを外して絵文字を比較できる():
    assert normalize_emoji("⏸️") == "⏸" and normalize_emoji("1️⃣") == "1⃣"
    assert normalize_emoji("✅") == "✅"
