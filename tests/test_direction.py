"""direction の導出(仕様 2-8、テスト 5)。

direction が決まらないと、CLAUDE.md 5章の「相手→自分 の抽出結果は、相手別の要求一覧として扱う」ができず、
自分が言ったことと言われたことを分けられない。そのためこのプロジェクトの目的に直結する。
"""

from __future__ import annotations

import pytest

from kijun.db import repo
from kijun.review.direction import compute_direction, derive_directions
from tests.review_helpers import CID, add_request, setup_conversation

SELF_AND_KOUHAI = {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a"}


# --- 4通り(仕様の必須) ------------------------------------------------------------


def test_自分から相手なら_self_to_other():
    assert compute_direction("SPEAKER_00", "SPEAKER_01", SELF_AND_KOUHAI) == "self_to_other"


def test_相手から自分なら_other_to_self():
    assert compute_direction("SPEAKER_01", "SPEAKER_00", SELF_AND_KOUHAI) == "other_to_self"


def test_第三者間のやり取りは_NULL_のまま():
    speaker_map = {"SPEAKER_00": "self", "SPEAKER_01": "kouhai_a", "SPEAKER_02": "friend_b"}
    assert compute_direction("SPEAKER_01", "SPEAKER_02", speaker_map) is None


def test_両方が_self_なら_NULL_のまま():
    # 独り言、または話者分離の誤り(同じ人が2つの話者ラベルに割れた)
    speaker_map = {"SPEAKER_00": "self", "SPEAKER_01": "self"}
    assert compute_direction("SPEAKER_00", "SPEAKER_01", speaker_map) is None
    assert compute_direction("SPEAKER_00", "SPEAKER_00", speaker_map) is None


# --- 境界 -------------------------------------------------------------------------


def test_言われた側が不明でも_自分が言ったなら_self_to_other():
    assert compute_direction("SPEAKER_00", None, SELF_AND_KOUHAI) == "self_to_other"


def test_言われた側が不明で_言ったのが相手なら_NULL():
    # 言われた側が自分かどうか分からないので決めない
    assert compute_direction("SPEAKER_01", None, SELF_AND_KOUHAI) is None


def test_speaker_map_に無い話者は自分でないものとして扱う():
    assert compute_direction("SPEAKER_00", "SPEAKER_09", SELF_AND_KOUHAI) == "self_to_other"
    assert compute_direction("SPEAKER_09", "SPEAKER_00", SELF_AND_KOUHAI) == "other_to_self"
    assert compute_direction(None, "SPEAKER_00", SELF_AND_KOUHAI) == "other_to_self"
    assert compute_direction("SPEAKER_08", "SPEAKER_09", SELF_AND_KOUHAI) is None


# --- DB への反映 ---------------------------------------------------------------------


@pytest.fixture
def conv(con):
    setup_conversation(con)
    return CID


def directions(con) -> dict[str, str | None]:
    return {r.request_id: r.direction for r in repo.get_requests(con, CID)}


def test_4通りが_DB_に反映される(con, conv):
    repo.set_speaker_map(con, CID, "SPEAKER_00", "self")
    repo.set_speaker_map(con, CID, "SPEAKER_01", "kouhai_a")
    repo.set_speaker_map(con, CID, "SPEAKER_02", "friend_b")
    add_request(con, "r1", from_label="SPEAKER_00", to_label="SPEAKER_01")  # 自分→相手
    add_request(con, "r2", from_label="SPEAKER_01", to_label="SPEAKER_00")  # 相手→自分
    add_request(con, "r3", from_label="SPEAKER_01", to_label="SPEAKER_02")  # 第三者間
    add_request(con, "r4", from_label="SPEAKER_00", to_label="SPEAKER_00")  # 両方 self

    result = derive_directions(con, CID)

    assert directions(con) == {"r1": "self_to_other", "r2": "other_to_self", "r3": None, "r4": None}
    assert (result.derived, result.total, result.self_to_other, result.other_to_self, result.undetermined) == (True, 4, 1, 1, 2)


def test_self_が決まっていない会議は導出せず_既存の値にも触らない(con, conv):
    repo.set_speaker_map(con, CID, "SPEAKER_01", "kouhai_a")  # self が無い
    add_request(con, "r1", direction="self_to_other")

    result = derive_directions(con, CID)

    assert result.derived is False and "self" in result.note
    assert directions(con) == {"r1": "self_to_other"}


def test_speaker_map_が空の会議も導出しない(con, conv):
    add_request(con, "r1")
    assert derive_directions(con, CID).derived is False
    assert directions(con) == {"r1": None}


def test_speaker_map_を直して再実行すると導出し直される_NULL_に戻ることもある(con, conv):
    add_request(con, "r1", from_label="SPEAKER_00", to_label="SPEAKER_01")
    repo.set_speaker_map(con, CID, "SPEAKER_00", "self")
    derive_directions(con, CID)
    assert directions(con) == {"r1": "self_to_other"}

    # 誤りに気付いて、00 を self から相手に直し、01 を self にした
    repo.set_speaker_map(con, CID, "SPEAKER_00", "kouhai_a")
    repo.set_speaker_map(con, CID, "SPEAKER_01", "self")
    derive_directions(con, CID)
    assert directions(con) == {"r1": "other_to_self"}

    # 両方 self に直すと、以前の値が残らず NULL になる
    repo.set_speaker_map(con, CID, "SPEAKER_00", "self")
    derive_directions(con, CID)
    assert directions(con) == {"r1": None}


def test_複数モデルの抽出結果すべてに適用する(con, conv):
    repo.set_speaker_map(con, CID, "SPEAKER_00", "self")
    add_request(con, "a#1", model="q4")
    add_request(con, "b#1", model="q8", from_label="SPEAKER_01", to_label="SPEAKER_00")
    derive_directions(con, CID)
    assert directions(con) == {"a#1": "self_to_other", "b#1": "other_to_self"}


def test_導出は_review_log_に記録しない(con, conv):
    repo.set_speaker_map(con, CID, "SPEAKER_00", "self")
    add_request(con, "r1")
    derive_directions(con, CID)
    assert repo.list_review_log(con) == []
