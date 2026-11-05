from __future__ import annotations

import random

import pytest

from kijun.minutes.chunk import Chunk, split_into_chunks
from kijun.models import Utterance
from tests.conftest import make_utterances


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


# --- 議事録の分け方(重なり 0) ----------------------------------------------------------


def test_重なり0で60分の会議を15分ごとに4つに分ける_発話は重複も欠落もしない():
    chunks = split_into_chunks(make_utterances(60), 15, 0)
    assert [seqs(c) for c in chunks] == [list(range(0, 15)), list(range(15, 30)), list(range(30, 45)), list(range(45, 60))]


def test_重なり0ではランダムな長さの発話でも全発話がちょうど1回ずつ入る():
    rng = random.Random(7)
    t = 0.0
    utterances = []
    for i in range(300):
        length = rng.uniform(1, 90)
        utterances.append(Utterance(i, "SPEAKER_00", t, t + length, "x"))
        t += length + rng.uniform(0, 3)
    chunks = split_into_chunks(utterances, 15, 0)
    flat = [s for c in chunks for s in seqs(c)]
    assert flat == list(range(300))
