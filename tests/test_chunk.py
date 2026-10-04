from __future__ import annotations

import random

import pytest

from kijun.extract.chunk import (
    Chunk,
    merge_chunk_results,
    normalize_quote,
    restore_global_seq,
    split_into_chunks,
)
from kijun.models import Utterance
from tests.conftest import make_utterances
from tests.fakes import make_item


def seqs(chunk: Chunk) -> list[int]:
    return [u.seq for u in chunk.utterances]


# --- 分割 -----------------------------------------------------------------------------


def test_100分の会議を25分_重複2分で分ける():
    # 1分の発話が 100 件。チャンク1は先頭から25分以内に終わる発話 0〜24。
    chunks = split_into_chunks(make_utterances(100), 25, 2)
    assert seqs(chunks[0]) == list(range(0, 25))
    # 次は、チャンク1の終了（25分）の2分前以降に始まる発話 23 から
    assert seqs(chunks[1])[0] == 23
    assert seqs(chunks[1]) == list(range(23, 48))
    # 重複区間は 2分 = 発話 2 件
    assert sorted(set(seqs(chunks[0])) & set(seqs(chunks[1]))) == [23, 24]
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].seq_to == 99


def test_チャンクの_seq_from_seq_to_が発話と一致する():
    for c in split_into_chunks(make_utterances(100), 25, 2):
        assert c.seq_from == c.utterances[0].seq
        assert c.seq_to == c.utterances[-1].seq
        assert c.start_sec == c.utterances[0].start_sec
        assert c.end_sec == c.utterances[-1].end_sec


def test_境界は発話を割らない_ランダムな長さの発話():
    rng = random.Random(1)
    utterances = []
    t = 0.0
    for i in range(400):
        length = rng.uniform(1, 90)
        utterances.append(Utterance(i, "SPEAKER_00", t, t + length, f"u{i}"))
        t += length + rng.uniform(0, 3)
    chunks = split_into_chunks(utterances, 25, 2)

    # 発話はそのままの形で、どれか1つ以上のチャンクに入っている（欠けも分割も無い）
    covered = set()
    for c in chunks:
        assert c.utterances == utterances[c.seq_from : c.seq_to + 1]  # 連続した切り出し
        covered |= set(seqs(c))
        if len(c.utterances) > 1:
            assert c.end_sec - c.start_sec <= 25 * 60 + 1e-9
    assert covered == set(range(400))
    # チャンクの先頭は単調に増え、最後は最後の発話に届く
    starts = [c.seq_from for c in chunks]
    assert starts == sorted(set(starts))
    assert chunks[-1].seq_to == 399


def test_隣り合うチャンクは重なる():
    chunks = split_into_chunks(make_utterances(100), 25, 2)
    for a, b in zip(chunks, chunks[1:]):
        assert b.seq_from <= a.seq_to  # 重複区間がある
        assert b.seq_from > a.seq_from  # 先頭は進む


def test_重複0分なら重ならない():
    chunks = split_into_chunks(make_utterances(100), 25, 0)
    assert seqs(chunks[0]) == list(range(0, 25))
    assert seqs(chunks[1])[0] == 25


def test_短い会議は1チャンク():
    chunks = split_into_chunks(make_utterances(10), 25, 2)
    assert len(chunks) == 1 and seqs(chunks[0]) == list(range(10))


def test_発話が1件でも動く():
    chunks = split_into_chunks(make_utterances(1), 25, 2)
    assert len(chunks) == 1


def test_発話が無ければ空():
    assert split_into_chunks([], 25, 2) == []


@pytest.mark.parametrize("chunk,overlap", [(0, 0), (25, 25), (25, -1), (10, 30)])
def test_不正な設定は拒否する(chunk, overlap):
    with pytest.raises(ValueError):
        split_into_chunks(make_utterances(5), chunk, overlap)


def test_1発話が_chunk_minutes_を超える場合はその発話だけで1チャンクにする():
    # 発話 0〜9 は1分ずつ、発話10 は 40分の独話（chunk_minutes = 25 を超える）、続けて発話 11〜20 が1分ずつ
    utterances = make_utterances(10)
    utterances.append(Utterance(10, "SPEAKER_00", 600.0, 3000.0, "長い独話"))
    utterances += [
        Utterance(11 + i, "SPEAKER_01", 3000.0 + 60 * i, 3060.0 + 60 * i, f"後{i}") for i in range(10)
    ]
    chunks = split_into_chunks(utterances, 25, 2)

    monologue_chunks = [c for c in chunks if 10 in seqs(c)]
    assert len(monologue_chunks) == 1
    assert seqs(monologue_chunks[0]) == [10]  # 独話は分割せず、単独のチャンクにする
    assert monologue_chunks[0].utterances[0].end_sec - monologue_chunks[0].utterances[0].start_sec == 2400
    # すべての発話がどこかのチャンクに入っている
    assert {s for c in chunks for s in seqs(c)} == set(range(21))
    # 他のチャンクの内容が、別のチャンクに完全に含まれてしまうことは無い
    for a in chunks:
        for b in chunks:
            if a is not b:
                assert not set(seqs(a)) <= set(seqs(b))


def test_先頭が長い独話でも進む():
    utterances = [Utterance(0, "SPEAKER_00", 0.0, 4000.0, "独話")] + [
        Utterance(1 + i, "SPEAKER_01", 4000.0 + 60 * i, 4060.0 + 60 * i, "x") for i in range(5)
    ]
    chunks = split_into_chunks(utterances, 25, 2)
    assert seqs(chunks[0]) == [0]
    assert {s for c in chunks for s in seqs(c)} == set(range(6))


# --- 発話番号の復元 -------------------------------------------------------------------


def make_chunk(seq_from: int, size: int, index: int = 1) -> Chunk:
    utterances = make_utterances(seq_from + size)[seq_from:]
    return Chunk(index=index, seq_from=seq_from, seq_to=seq_from + size - 1, utterances=utterances)


def test_チャンク内の番号をグローバルな_seq_に復元する():
    chunk = make_chunk(seq_from=23, size=25)
    restored = restore_global_seq(make_item(0, 3), chunk)
    assert (restored.seq_from, restored.seq_to) == (23, 26)
    restored = restore_global_seq(make_item(24), chunk)
    assert (restored.seq_from, restored.seq_to) == (47, 47)


def test_先頭のチャンクでは番号が変わらない():
    chunk = make_chunk(seq_from=0, size=10, index=0)
    assert restore_global_seq(make_item(4, 5), chunk).seq_from == 4


def test_範囲外や逆順の番号は_None():
    chunk = make_chunk(seq_from=23, size=25)
    assert restore_global_seq(make_item(25), chunk) is None  # 25 件目（0〜24 の外）
    assert restore_global_seq(make_item(-1), chunk) is None
    assert restore_global_seq(make_item(3, 2), chunk) is None
    # 元のオブジェクトは変更されない
    original = make_item(1)
    restore_global_seq(original, chunk)
    assert original.seq_from == 1


# --- 重複排除 -------------------------------------------------------------------------


def two_chunks():
    """チャンク0 = seq 0〜24、チャンク1 = seq 23〜47（重複区間は seq 23〜24）。"""
    chunks = split_into_chunks(make_utterances(100), 25, 2)
    return chunks[0], chunks[1], chunks[2]


def test_重複区間をまたぐ抽出結果が1件に落ちる():
    c0, c1, _ = two_chunks()
    # 同じ指摘（グローバルな seq 23）が、チャンク0 では番号 23、チャンク1 では番号 0 として2回出る
    outcome = merge_chunk_results(
        [
            (c0, [make_item(23, quote="結論を先に書いて")]),
            (c1, [make_item(0, quote="結論を先に書いて")]),
        ]
    )
    assert len(outcome.items) == 1
    assert outcome.items[0].seq_from == 23
    assert outcome.overlap_dropped_count == 1


def test_重複区間では後続チャンクの結果を捨て_前のチャンクの結果を残す():
    c0, c1, _ = two_chunks()
    outcome = merge_chunk_results(
        [
            (c0, [make_item(24, quote="前のチャンクの文言")]),
            (c1, [make_item(1, quote="後のチャンクの文言")]),  # グローバル seq 24（重複区間）
        ]
    )
    assert [i.quote for i in outcome.items] == ["前のチャンクの文言"]


def test_重複区間の外の結果は両方残る():
    c0, c1, _ = two_chunks()
    outcome = merge_chunk_results(
        [
            (c0, [make_item(5, quote="a")]),
            (c1, [make_item(10, quote="b")]),  # グローバル seq 33
        ]
    )
    assert [(i.seq_from, i.quote) for i in outcome.items] == [(5, "a"), (33, "b")]
    assert outcome.overlap_dropped_count == 0


def test_seq_from_と_seq_to_が全チャンクでグローバルな番号になる():
    c0, c1, c2 = two_chunks()
    outcome = merge_chunk_results(
        [
            (c0, [make_item(2, 4, quote="q0")]),
            (c1, [make_item(5, 7, quote="q1")]),  # 23 + 5 = 28, 30
            (c2, [make_item(5, 7, quote="q2")]),  # チャンク2 の先頭は重複区間なので、その外の番号を使う
        ]
    )
    got = [(i.quote, i.seq_from, i.seq_to) for i in outcome.items]
    assert got[0] == ("q0", 2, 4)
    assert got[1] == ("q1", 28, 30)
    assert got[2][0] == "q2" and got[2][1] == c2.seq_from + 5 and got[2][2] == c2.seq_from + 7
    # 発話の本文と突き合わせて、番号が正しい発話を指していることを確かめる
    all_utterances = {u.seq: u for u in make_utterances(100)}
    assert all_utterances[got[1][1]].text == "発話28"


def test_結果は_seq_from_の昇順に並ぶ():
    c0, c1, _ = two_chunks()
    outcome = merge_chunk_results(
        [(c0, [make_item(9, quote="b"), make_item(3, quote="a")]), (c1, [make_item(20, quote="c")])]
    )
    assert [i.seq_from for i in outcome.items] == [3, 9, 43]


def test_保険の重複判定_quote_が一致し_seq_from_の差が2以内なら同一とみなす():
    c0, c1, _ = two_chunks()
    # チャンク1 の番号 3 = グローバル 26。重複区間（23〜24）の外だが、チャンク0 の 24 と quote が同じで差が 2
    outcome = merge_chunk_results(
        [
            (c0, [make_item(24, quote="結論を、先に書いて。")]),
            (c1, [make_item(3, quote="結論を先に書いて")]),  # 句読点と空白の違いは無視する
        ]
    )
    assert len(outcome.items) == 1
    assert outcome.duplicate_count == 1


def test_保険の重複判定_seq_from_の差が3以上なら別の指摘():
    c0, c1, _ = two_chunks()
    outcome = merge_chunk_results(
        [
            (c0, [make_item(24, quote="同じ文言")]),
            (c1, [make_item(4, quote="同じ文言")]),  # グローバル 27、差 3
        ]
    )
    assert len(outcome.items) == 2


def test_保険の重複判定_quote_が違えば別の指摘():
    c0, c1, _ = two_chunks()
    outcome = merge_chunk_results(
        [(c0, [make_item(24, quote="文言A")]), (c1, [make_item(3, quote="文言B")])]
    )
    assert len(outcome.items) == 2


def test_直前のチャンクの抽出が失敗していたら重複区間の結果を捨てない():
    c0, c1, _ = two_chunks()
    outcome = merge_chunk_results([(c0, None), (c1, [make_item(0, quote="q")])])  # グローバル seq 23
    assert [i.seq_from for i in outcome.items] == [23]
    assert outcome.overlap_dropped_count == 0


def test_範囲外の番号は捨てて件数を数える():
    c0, _, _ = two_chunks()
    outcome = merge_chunk_results([(c0, [make_item(99, quote="q"), make_item(1, quote="r")])])
    assert [i.quote for i in outcome.items] == ["r"]
    assert outcome.invalid_count == 1


def test_すべて失敗なら空():
    c0, c1, _ = two_chunks()
    assert merge_chunk_results([(c0, None), (c1, None)]).items == []


def test_normalize_quote():
    assert normalize_quote("結論を、先に 書いて。") == "結論を先に書いて"
    assert normalize_quote("A b！") == "Ab"
