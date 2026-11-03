"""話者の対応付け(仕様 2-7、テスト 4)。"""

from __future__ import annotations

import pytest

from kijun.db import repo
from kijun.discord.read import run_read
from kijun.review.speaker import AmbiguousSpeakerReply, apply_speaker_replies, parse_structured
from tests.fakes import FakeDiscordClient, FakeSpeakerInterpreter
from tests.review_helpers import CID, NOW, add_request, reply, setup_conversation

LABELS = ["SPEAKER_00", "SPEAKER_01"]


# --- 構造化した形式は LLM を使わずに解釈される -------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("00=self 01=kouhai_a", {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}),
        ("00=self, 01=kouhai_a", {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}),
        ("00=self、01=kouhai_a", {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}),
        ("00 = self\n01 = kouhai_a", {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}),
        ("SPEAKER_00=Self speaker_01=Kouhai_A", {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}),
        ("  00=self  ", {"SPEAKER_00": "self"}),
    ],
)
def test_構造化形式の解釈(text, expected):
    assert parse_structured(text) == expected


@pytest.mark.parametrize("text", ["00は自分、01は後輩A", "了解です", "", "00=self ありがとう"])
def test_構造化形式でなければ_None(text):
    assert parse_structured(text) is None


def test_同じ話者に別の人物が書かれていたら解釈しない():
    with pytest.raises(AmbiguousSpeakerReply):
        parse_structured("00=self 00=kouhai_a")


def test_同じ話者に同じ人物を2回書くのは許す():
    assert parse_structured("00=self 00=self") == {"SPEAKER_00": "self"}


def test_構造化形式の返信は_LLM_に渡さずに_speaker_map_に反映される(con):
    setup_conversation(con)
    interpreter = FakeSpeakerInterpreter()

    outcome = apply_speaker_replies(con, CID, LABELS, [reply("00=self 01=kouhai_a")], interpreter)

    assert interpreter.calls == []  # LLM を使っていない
    assert repo.get_speaker_map(con, CID) == {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}
    assert outcome.changed and outcome.has_self and not outcome.failed


def test_自然な文は_LLM_に渡る(con):
    setup_conversation(con)
    text = "00は自分、01は後輩A"
    interpreter = FakeSpeakerInterpreter({text: {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}})

    outcome = apply_speaker_replies(con, CID, LABELS, [reply(text)], interpreter)

    assert interpreter.calls == [text]
    assert repo.get_speaker_map(con, CID) == {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}
    assert outcome.has_self


def test_自然な文の解釈に失敗したら反映せず_failed(con):
    setup_conversation(con)
    outcome = apply_speaker_replies(con, CID, LABELS, [reply("00は自分")], FakeSpeakerInterpreter(fail=True))
    assert outcome.failed and repo.get_speaker_map(con, CID) == {}


def test_LLM_を使えないときも構造化形式は反映できる(con):
    setup_conversation(con)
    ok = apply_speaker_replies(con, CID, LABELS, [reply("00=self")], None)
    assert ok.has_self and not ok.failed
    ng = apply_speaker_replies(con, CID, LABELS, [reply("01は後輩A", 5)], None)
    assert ng.failed


def test_会議に無い話者ラベルは反映しない(con):
    setup_conversation(con)
    outcome = apply_speaker_replies(con, CID, LABELS, [reply("00=self 07=friend")], None)
    assert repo.get_speaker_map(con, CID) == {"SPEAKER_00": "self"}
    assert any("SPEAKER_07" in m for m in outcome.messages)


def test_人物ラベルが英数字でなければ反映しない(con):
    setup_conversation(con)
    outcome = apply_speaker_replies(
        con, CID, LABELS, [reply("自然な文", 1)], FakeSpeakerInterpreter({"自然な文": {"SPEAKER_00": "self", "SPEAKER_01": "後輩A"}})
    )
    assert repo.get_speaker_map(con, CID) == {"SPEAKER_00": "self"}
    assert any("後輩A" in m for m in outcome.messages)


def test_ユーザー本人以外の返信は取り込まない(con):
    setup_conversation(con)
    apply_speaker_replies(con, CID, LABELS, [reply("00=self", owner=False)], None)
    assert repo.get_speaker_map(con, CID) == {}


def test_self_が決まらない会議はその旨を出す(con):
    setup_conversation(con)
    outcome = apply_speaker_replies(con, CID, LABELS, [reply("01=kouhai_a")], None)
    assert repo.get_speaker_map(con, CID) == {"SPEAKER_01": "kouhai_a"}
    assert not outcome.has_self and any("self が決まっていません" in m for m in outcome.messages)


def test_返信は古い順に適用され_後の返信が訂正になる(con):
    setup_conversation(con)
    outcome = apply_speaker_replies(
        con, CID, LABELS, [reply("00=kouhai_a 01=self", 10), reply("00=self 01=kouhai_a", 5)], None
    )
    assert repo.get_speaker_map(con, CID) == {"SPEAKER_00": "kouhai_a", "SPEAKER_01": "self"}
    assert outcome.has_self


# --- read 経由: self が決まらない会議では direction を導出しない -----------------------------------


@pytest.fixture
def sample_post(con, cfg):
    cfg.discord.review_channel_id = "111"
    setup_conversation(con)
    add_request(con, "r1", from_label="SPEAKER_00", to_label="SPEAKER_01")
    add_request(con, "r2", from_label="SPEAKER_01", to_label="SPEAKER_00")
    repo.insert_discord_post(con, repo.DiscordPostRow("s1", "speaker_sample", None, CID, NOW, None, "111", None))


def directions(con):
    return {x.request_id: x.direction for x in repo.get_requests(con, CID)}


def test_self_が決まれば_direction_が導出され_投稿は処理済みになる(con, cfg, sample_post):
    client = FakeDiscordClient()
    client.replies["s1"] = [reply("00=self 01=kouhai_a")]

    report = run_read(con, cfg, client, None, None, NOW)

    assert directions(con) == {"r1": "self_to_other", "r2": "other_to_self"}
    assert repo.list_discord_posts(con)[0].processed_at == NOW
    assert CID in report.derived_conversations and any("direction を導出しました" in m for m in report.messages)


def test_self_が決まらない会議では_direction_を導出せず_投稿も未処理のまま(con, cfg, sample_post):
    client = FakeDiscordClient()
    client.replies["s1"] = [reply("01=kouhai_a")]

    report = run_read(con, cfg, client, None, None, NOW)

    assert directions(con) == {"r1": None, "r2": None}
    assert repo.list_discord_posts(con)[0].processed_at is None
    assert any("self が決まっていません" in m for m in report.messages)
    assert not any("direction を導出しました" in m for m in report.messages)

    # 続きの返信で self が決まれば、次回の read で導出される
    client.replies["s1"] = [reply("01=kouhai_a", 1), reply("00=self", 2)]
    run_read(con, cfg, client, None, None, NOW)
    assert directions(con) == {"r1": "self_to_other", "r2": "other_to_self"}
    assert repo.list_discord_posts(con)[0].processed_at == NOW


def test_返信が無ければ何も反映しない(con, cfg, sample_post):
    run_read(con, cfg, FakeDiscordClient(), None, None, NOW)
    assert repo.get_speaker_map(con, CID) == {} and directions(con) == {"r1": None, "r2": None}
