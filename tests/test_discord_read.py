"""リアクションと返信の読み取り(仕様 1-5、2-4、2-6、テスト 2・3)。

2番: 誤って承認したチェック項目は、間違いであることが後から分かりにくい。判定できないリアクションを
     片方優先で決めてはいけない。
3番: LLM の振り分けが間違っても、人が原文から直せなければならない。
"""

from __future__ import annotations

import json

import pytest

from kijun.db import repo
from kijun.discord.base import Reaction
from kijun.discord.read import count_unprocessed, run_read
from kijun.review.reply import RoutedEntry
from tests.fakes import FakeDiscordClient, FakeReplyRouter
from tests.review_helpers import (
    APPROVE, CID, HOLD, NOW, ONE, REJECT, THREE, TWO, add_item, candidate, r, reply, setup_conversation,
)


@pytest.fixture
def posted(con, cfg):
    """候補 item_0001 を投稿済み(メッセージ m1)にする。"""
    cfg.discord.review_channel_id = "111"
    candidate(con, 1, importance=2)
    repo.insert_discord_post(
        con, repo.DiscordPostRow("m1", "new_candidate", "item_0001", CID, NOW, None, "111", None)
    )
    return "m1"


def post_row(con, message_id="m1"):
    return next(p for p in repo.list_discord_posts(con) if p.message_id == message_id)


def item(con):
    return repo.get_knowledge_item(con, "item_0001")


def read(con, cfg, client, router=None):
    return run_read(con, cfg, client, router, None, NOW)


# --- 2番: 判定できないリアクション --------------------------------------------------------


def test_承認と却下が両方付いていたら_status_を変えず_processed_at_を埋めず_記録する(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE, REJECT)

    report = read(con, cfg, client)

    assert item(con).status == "candidate"  # 片方を優先して決めない
    assert post_row(con).processed_at is None
    log = repo.list_review_log(con, "item_0001", "ambiguous_reaction")
    assert len(log) == 1 and "✅" in log[0].after and "❌" in log[0].after
    assert repo.list_review_log(con, "item_0001", "status_changed") == []
    # 会議IDと項目IDを標準出力用の報告に出す
    assert len(report.ambiguous) == 1 and CID in report.ambiguous[0] and "item_0001" in report.ambiguous[0]
    assert report.processed_posts == 0


def test_判定できなかった投稿は次回の_read_で再び対象になる(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE, REJECT)
    read(con, cfg, client)
    assert count_unprocessed(con) == 1

    # 2回目の読み取りでも同じ状態。対象になり、報告され、ログは重複して増えない
    second = read(con, cfg, client)
    assert len(second.ambiguous) == 1
    assert len(repo.list_review_log(con, "item_0001", "ambiguous_reaction")) == 1
    assert item(con).status == "candidate" and post_row(con).processed_at is None

    # ユーザーが ❌ を外した。今度は承認として決まる
    client.reactions["m1"] = r(APPROVE)
    third = read(con, cfg, client)
    assert item(con).status == "approved" and post_row(con).processed_at == NOW
    assert third.ambiguous == [] and third.status_changes == 1
    assert count_unprocessed(con) == 0


def test_承認と保留_却下と保留の組み合わせも判定しない(con, cfg, posted):
    client = FakeDiscordClient()
    for combo in [(APPROVE, HOLD), (REJECT, HOLD), (APPROVE, REJECT, HOLD)]:
        client.reactions["m1"] = r(*combo)
        read(con, cfg, client)
        assert item(con).status == "candidate" and post_row(con).processed_at is None


def test_判定できないリアクションが_ある間に異なる組み合わせになれば_それも記録する(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE, REJECT)
    read(con, cfg, client)
    client.reactions["m1"] = r(APPROVE, HOLD)
    read(con, cfg, client)
    assert len(repo.list_review_log(con, "item_0001", "ambiguous_reaction")) == 2


@pytest.mark.parametrize(
    "emoji, expected", [(APPROVE, "approved"), (REJECT, "rejected"), (HOLD, "held")]
)
def test_status_のリアクションが1つだけならそのとおりに更新する(con, cfg, posted, emoji, expected):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(emoji)

    read(con, cfg, client)

    assert item(con).status == expected
    assert post_row(con).processed_at == NOW
    log = repo.list_review_log(con, "item_0001", "status_changed")
    assert [(x.before, x.after) for x in log] == [("candidate", expected)]


def test_同じ絵文字を複数人が付けても1つの判定として扱う(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = [Reaction("✅", 3)]
    read(con, cfg, client)
    assert item(con).status == "approved"


def test_重要度のリアクションが2つ以上なら_importance_を変えない(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(ONE, THREE)

    report = read(con, cfg, client)

    assert item(con).importance == 2
    assert repo.list_review_log(con, "item_0001", "importance_overridden") == []
    assert len(repo.list_review_log(con, "item_0001", "ambiguous_reaction")) == 1
    assert len(report.ambiguous) == 1 and "item_0001" in report.ambiguous[0]


def test_重要度が複数でも_status_は別に判定する(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE, ONE, THREE)
    read(con, cfg, client)
    assert item(con).status == "approved" and item(con).importance == 2


def test_status_が無く重要度だけなら重要度のみ更新し_processed_at_は埋めない(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(THREE)

    read(con, cfg, client)

    assert item(con).importance == 3 and item(con).status == "candidate"
    assert post_row(con).processed_at is None  # 後から status を付ける可能性があるため


def test_重要度の上書きは_LLM_の判定値とユーザーの値の両方が1行から取れる(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(THREE)
    read(con, cfg, client)
    log = repo.list_review_log(con, "item_0001", "importance_overridden")
    assert [(x.before, x.after) for x in log] == [("2", "3")]


def test_重要度の上書きは読み直しても重複して記録しない(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(THREE)
    read(con, cfg, client)
    read(con, cfg, client)
    assert len(repo.list_review_log(con, "item_0001", "importance_overridden")) == 1
    # 後から別の値に変えたら記録する
    client.reactions["m1"] = r(ONE)
    read(con, cfg, client)
    log = repo.list_review_log(con, "item_0001", "importance_overridden")
    assert [(x.before, x.after) for x in log] == [("2", "3"), ("3", "1")]


def test_判定値と同じ値を付けたら_同意として_before_と_after_が同じ行を残す(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(TWO)
    read(con, cfg, client)
    log = repo.list_review_log(con, "item_0001", "importance_overridden")
    assert [(x.before, x.after) for x in log] == [("2", "2")]
    assert item(con).importance == 2


def test_LLM_が判定できなかった項目の重要度をユーザーが付けたら_before_は_NULL(con, cfg):
    cfg.discord.review_channel_id = "111"
    candidate(con, 1, importance=None)
    repo.insert_discord_post(con, repo.DiscordPostRow("m1", "new_candidate", "item_0001", CID, NOW, None, "111", None))
    client = FakeDiscordClient()
    client.reactions["m1"] = r(TWO)
    read(con, cfg, client)
    log = repo.list_review_log(con, "item_0001", "importance_overridden")
    assert [(x.before, x.after) for x in log] == [(None, "2")]
    assert item(con).importance == 2


def test_リアクションが無ければ何もしない_processed_at_も埋めない(con, cfg, posted):
    client = FakeDiscordClient()
    report = read(con, cfg, client)
    assert item(con).status == "candidate" and post_row(con).processed_at is None
    assert repo.list_review_log(con) == [] and report.processed_posts == 0


def test_Discord_から取得できなかった投稿は未処理のまま残す(con, cfg, posted):
    client = FakeDiscordClient()
    client.missing.add("m1")
    report = read(con, cfg, client)
    assert post_row(con).processed_at is None
    assert any("取得できませんでした" in m for m in report.messages)


def test_処理済みの投稿は読み直さない(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    read(con, cfg, client)
    client.fetch_calls.clear()
    client.reactions["m1"] = r(REJECT)  # 処理後に付け替えても反映しない
    read(con, cfg, client)
    assert client.fetch_calls == [] and item(con).status == "approved"


def test_チャンネルIDが空なら何もせず設定が必要だと出す(con, cfg):
    report = run_read(con, cfg, FakeDiscordClient(), None, None, NOW)
    assert report.skipped_for_config and "設定が必要" in report.messages[0]


# --- 3番: 返信の原文の保存 ------------------------------------------------------------------


def test_成功したとき_原文と振り分け結果が別の行で残る(con, cfg, posted):
    text = "冒頭に書かないと読み飛ばされるから。ただし社内向けの連絡では不要"
    router = FakeReplyRouter({text: [
        RoutedEntry(column="reason", text="冒頭に書かないと読み飛ばされる"),
        RoutedEntry(column="not_applies_when", text="社内向けの連絡"),
    ]})
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    client.replies["m1"] = [reply(text, 5)]

    report = read(con, cfg, client, router)

    received = repo.list_review_log(con, "item_0001", "reply_received")
    routed = repo.list_review_log(con, "item_0001", "reply_routed")
    assert len(received) == 1 and len(routed) == 1 and received[0].log_id != routed[0].log_id
    assert received[0].after == text  # 原文の全文
    detail = json.loads(routed[0].after)
    assert {(a["column"], a["new"]) for a in detail["applied"]} == {
        ("reason", "冒頭に書かないと読み飛ばされる"), ("not_applies_when", "社内向けの連絡"),
    }
    assert received[0].before == routed[0].before  # 同じ返信であることを結ぶ鍵
    got = item(con)
    assert got.reason == "冒頭に書かないと読み飛ばされる" and got.not_applies_when == "社内向けの連絡"
    assert got.version == 2
    assert report.replies_routed == 1 and post_row(con).processed_at == NOW


def test_振り分けが失敗しても原文は_review_log_に残り_項目は変わらない(con, cfg, posted):
    router = FakeReplyRouter(fail=True)
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    client.replies["m1"] = [reply("これは大事な説明です", 5)]
    before = item(con)

    report = read(con, cfg, client, router)

    received = repo.list_review_log(con, "item_0001", "reply_received")
    assert [x.after for x in received] == ["これは大事な説明です"]  # 原文は残る
    assert repo.list_review_log(con, "item_0001", "reply_routed") == []
    after = item(con)
    assert (after.reason, after.applies_when, after.check_text, after.version) == (
        before.reason, before.applies_when, before.check_text, before.version
    )
    assert post_row(con).processed_at is None  # 次回の read で再試行される
    assert report.replies_failed == 1


def test_失敗した返信は次回に再試行され_原文の記録は重複しない(con, cfg, posted):
    text = "適用条件は社外向けの告知のとき"
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    client.replies["m1"] = [reply(text, 5)]
    read(con, cfg, client, FakeReplyRouter(fail=True))

    router = FakeReplyRouter({text: [RoutedEntry(column="applies_when", text="社外向けの告知のとき")]})
    second = read(con, cfg, client, router)

    assert len(repo.list_review_log(con, "item_0001", "reply_received")) == 1
    assert len(repo.list_review_log(con, "item_0001", "reply_routed")) == 1
    assert item(con).applies_when == "社外向けの告知のとき"
    assert post_row(con).processed_at == NOW and second.processed_posts == 1


def test_LLM_を使えない場合も原文は残り_次回に振り分けられる(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    client.replies["m1"] = [reply("理由はこれ", 5)]
    report = read(con, cfg, client, router=None)
    assert [x.after for x in repo.list_review_log(con, "item_0001", "reply_received")] == ["理由はこれ"]
    assert post_row(con).processed_at is None and report.replies_failed == 1


def test_処理済みの返信は再度振り分けない(con, cfg, posted):
    text = "理由はこれ"
    router = FakeReplyRouter({text: [RoutedEntry(column="reason", text="これ")]})
    client = FakeDiscordClient()
    client.replies["m1"] = [reply(text, 5)]
    read(con, cfg, client, router)  # status が無いので processed にならず、次回も読まれる
    read(con, cfg, client, router)
    assert len(router.calls) == 1  # 2回目は LLM を呼ばない
    assert len(repo.list_review_log(con, "item_0001", "reply_received")) == 1


def test_どの欄にも当たらない返信も_原文と空の振り分け結果が残る(con, cfg, posted):
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    client.replies["m1"] = [reply("了解です", 5)]
    read(con, cfg, client, FakeReplyRouter())
    assert [x.after for x in repo.list_review_log(con, "item_0001", "reply_received")] == ["了解です"]
    assert json.loads(repo.list_review_log(con, "item_0001", "reply_routed")[0].after)["applied"] == []
    assert post_row(con).processed_at == NOW


def test_check_text_の修正は_check_text_edited_に変更前後が残る(con, cfg, posted):
    text = "その項目は「冒頭1行以内に行動が書かれているか」に直して"
    router = FakeReplyRouter({text: [RoutedEntry(column="check_text", text="冒頭1行以内に行動が書かれているか")]})
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    client.replies["m1"] = [reply(text, 5)]
    read(con, cfg, client, router)
    log = repo.list_review_log(con, "item_0001", "check_text_edited")
    assert [(x.before, x.after) for x in log] == [("チェック項目1", "冒頭1行以内に行動が書かれているか")]
    assert item(con).check_text == "冒頭1行以内に行動が書かれているか"


def test_ユーザー本人以外の返信は取り込まない(con, cfg, posted):
    router = FakeReplyRouter({"他人の発言": [RoutedEntry(column="reason", text="他人")]})
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    client.replies["m1"] = [reply("他人の発言", 5, owner=False)]
    read(con, cfg, client, router)
    assert router.calls == [] and repo.list_review_log(con, "item_0001", "reply_received") == []
    assert item(con).reason is None


def test_既にある理由は置き換え_古い値は振り分け結果の記録に残る(con, cfg):
    cfg.discord.review_channel_id = "111"
    candidate(con, 1, importance=2, reason="抽出時の理由")
    repo.insert_discord_post(con, repo.DiscordPostRow("m1", "new_candidate", "item_0001", CID, NOW, None, "111", None))
    router = FakeReplyRouter({"理由はこう": [RoutedEntry(column="reason", text="ユーザーの理由")]})
    client = FakeDiscordClient()
    client.replies["m1"] = [reply("理由はこう", 5)]
    read(con, cfg, client, router)
    assert item(con).reason == "ユーザーの理由"
    applied = json.loads(repo.list_review_log(con, "item_0001", "reply_routed")[0].after)["applied"]
    assert applied[0]["old"] == "抽出時の理由" and applied[0]["new"] == "ユーザーの理由"


# --- 矛盾の投稿への返信 -------------------------------------------------------------------------


@pytest.fixture
def contradiction(con, cfg):
    cfg.discord.contradictions_channel_id = "222"
    add_item(con, "item_0001", check_text="冒頭に書く", pass_criterion="冒頭か")
    add_item(con, "item_0002", check_text="末尾に書く", pass_criterion="末尾か")
    repo.insert_discord_post(con, repo.DiscordPostRow("c1", "contradiction", "item_0001", None, NOW, None, "222", "item_0002"))


def test_矛盾の返信は_A_B_の対象に応じて項目へ反映される(con, cfg, contradiction):
    text = "Aは社外向けの告知のとき。Bは社内連絡のとき"
    router = FakeReplyRouter({text: [
        RoutedEntry(column="applies_when", text="社外向けの告知のとき", target="A"),
        RoutedEntry(column="applies_when", text="社内連絡のとき", target="B"),
    ]})
    client = FakeDiscordClient()
    client.reactions["c1"] = r(APPROVE)  # リアクションでは何も変えない
    client.replies["c1"] = [reply(text, 5)]

    read(con, cfg, client, router)

    a, b = repo.get_knowledge_item(con, "item_0001"), repo.get_knowledge_item(con, "item_0002")
    assert (a.applies_when, b.applies_when) == ("社外向けの告知のとき", "社内連絡のとき")
    assert a.status == "candidate" and b.status == "candidate"
    assert router.calls[0][2].item_id == "item_0002"  # もう一方の項目も LLM に渡す
    assert [x.after for x in repo.list_review_log(con, "item_0001", "reply_received")] == [text]
    assert len(repo.list_review_log(con, "item_0002", "reply_routed")) == 1  # B の履歴にも残る
    assert post_row(con, "c1").processed_at == NOW


def test_矛盾の返信で対象が特定できない部分は反映せず_原文は残る(con, cfg, contradiction):
    text = "条件の違いです"
    router = FakeReplyRouter({text: [RoutedEntry(column="applies_when", text="どちらか不明", target=None)]})
    client = FakeDiscordClient()
    client.replies["c1"] = [reply(text, 5)]
    report = read(con, cfg, client, router)
    assert repo.get_knowledge_item(con, "item_0001").applies_when is None
    assert repo.get_knowledge_item(con, "item_0002").applies_when is None
    assert [x.after for x in repo.list_review_log(con, "item_0001", "reply_received")] == [text]
    detail = json.loads(repo.list_review_log(con, "item_0001", "reply_routed")[0].after)
    assert detail["applied"] == [] and detail["unapplied"][0]["reason"] == "どちらの項目か特定できない"
    assert any("反映できませんでした" in m for m in report.messages)


def test_矛盾の投稿は返信が無ければ未処理のまま(con, cfg, contradiction):
    client = FakeDiscordClient()
    client.reactions["c1"] = r(APPROVE)
    read(con, cfg, client, FakeReplyRouter())
    assert post_row(con, "c1").processed_at is None


# --- 束ね結果 ----------------------------------------------------------------------------------


def test_束ね結果に_X_が付いたら束ね直しが必要だと報告し_自動では分割し直さない(con, cfg):
    from tests.conftest import make_conversation

    cfg.discord.review_channel_id = "111"
    repo.insert_conversation(con, make_conversation("b1", source_files=["a.m4a", "b.m4a"]))
    repo.insert_discord_post(con, repo.DiscordPostRow("b1m", "bundle_result", None, "b1", NOW, None, "111", None))
    client = FakeDiscordClient()
    client.reactions["b1m"] = r(REJECT)

    report = read(con, cfg, client)

    assert report.rebundle_needed == ["b1"] and any("gap_minutes" in m for m in report.messages)
    assert repo.get_conversation(con, "b1").source_files == ["a.m4a", "b.m4a"]  # 分割し直していない
    assert post_row(con, "b1m").processed_at == NOW


def test_束ね結果にリアクションが無ければ未処理のまま(con, cfg):
    from tests.conftest import make_conversation

    cfg.discord.review_channel_id = "111"
    repo.insert_conversation(con, make_conversation("b1", source_files=["a.m4a", "b.m4a"]))
    repo.insert_discord_post(con, repo.DiscordPostRow("b1m", "bundle_result", None, "b1", NOW, None, "111", None))
    report = read(con, cfg, FakeDiscordClient())
    assert report.rebundle_needed == [] and post_row(con, "b1m").processed_at is None
