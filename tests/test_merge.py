from __future__ import annotations

from kijun.transcribe.base import SpeakerTurn, TranscribedSegment, shift_items
from kijun.transcribe.merge import assign_speaker, merge_segments, overlap_sec


def seg(start: float, end: float, text: str = "あ") -> TranscribedSegment:
    return TranscribedSegment(start, end, text)


def turn(start: float, end: float, label: str) -> SpeakerTurn:
    return SpeakerTurn(start, end, label)


def test_重なりが最大の話者が割り当たる():
    # 区間 0〜10 のうち、SPEAKER_00 と 3秒、SPEAKER_01 と 6秒重なる
    turns = [turn(0, 3, "SPEAKER_00"), turn(4, 10, "SPEAKER_01")]
    assert assign_speaker(seg(0, 10), turns) == "SPEAKER_01"


def test_話者の区間が区間の一部しか覆っていなくても割り当たる():
    assert assign_speaker(seg(5, 6), [turn(0, 5.5, "SPEAKER_00")]) == "SPEAKER_00"


def test_重なりが無い場合は_None():
    turns = [turn(0, 5, "SPEAKER_00"), turn(10, 15, "SPEAKER_01")]
    assert assign_speaker(seg(6, 9), turns) is None


def test_接しているだけ_重なり0_は_None():
    assert assign_speaker(seg(5, 6), [turn(0, 5, "SPEAKER_00")]) is None


def test_話者区間が空なら_None():
    assert assign_speaker(seg(0, 1), []) is None


def test_重なりが同じ長さなら先にある話者():
    turns = [turn(0, 2, "SPEAKER_00"), turn(2, 4, "SPEAKER_01")]
    assert assign_speaker(seg(1, 3), turns) == "SPEAKER_00"


def test_overlap_sec():
    assert overlap_sec(0, 5, 3, 8) == 2
    assert overlap_sec(0, 5, 5, 8) == 0
    assert overlap_sec(0, 5, 6, 8) == 0


def test_同一話者の連続発話は_min_utterance_sec_未満の間隔で結合される():
    turns = [turn(0, 10, "SPEAKER_00")]
    segs = [seg(0, 2, "おはよう"), seg(2.3, 4, "ございます")]  # 間隔 0.3 秒 < 0.4
    out = merge_segments(segs, turns, min_utterance_sec=0.4)
    assert len(out) == 1
    assert out[0].text == "おはようございます"
    assert (out[0].start_sec, out[0].end_sec) == (0, 4)
    assert out[0].speaker_label == "SPEAKER_00"
    assert out[0].seq == 0


def test_間隔がちょうど_min_utterance_sec_なら結合しない():
    turns = [turn(0, 10, "SPEAKER_00")]
    segs = [seg(0, 2, "a"), seg(2.5, 4, "b")]  # 間隔 0.5
    out = merge_segments(segs, turns, min_utterance_sec=0.5)
    assert [u.text for u in out] == ["a", "b"]


def test_間隔が_min_utterance_sec_以上なら結合しない():
    turns = [turn(0, 10, "SPEAKER_00")]
    out = merge_segments([seg(0, 2, "a"), seg(3, 4, "b")], turns, min_utterance_sec=0.4)
    assert [u.text for u in out] == ["a", "b"]


def test_話者が違えば間隔が短くても結合しない():
    turns = [turn(0, 2, "SPEAKER_00"), turn(2, 4, "SPEAKER_01")]
    out = merge_segments([seg(0, 2, "a"), seg(2.1, 4, "b")], turns, min_utterance_sec=0.4)
    assert [(u.speaker_label, u.text) for u in out] == [("SPEAKER_00", "a"), ("SPEAKER_01", "b")]


def test_話者が不明_None_の区間どうしは結合しない():
    out = merge_segments([seg(0, 1, "a"), seg(1.1, 2, "b")], [], min_utterance_sec=0.4)
    assert [(u.speaker_label, u.text) for u in out] == [(None, "a"), (None, "b")]


def test_3つ以上の連続も結合され_seq_は0から振られる():
    turns = [turn(0, 5, "SPEAKER_00"), turn(5, 10, "SPEAKER_01")]
    segs = [seg(0, 1, "a"), seg(1.1, 2, "b"), seg(2.2, 3, "c"), seg(5, 6, "d"), seg(6.2, 7, "e")]
    out = merge_segments(segs, turns, min_utterance_sec=0.4)
    assert [(u.seq, u.speaker_label, u.text) for u in out] == [
        (0, "SPEAKER_00", "abc"),
        (1, "SPEAKER_01", "de"),
    ]


def test_入力の順序が乱れていても開始時刻の昇順に処理する():
    turns = [turn(0, 10, "SPEAKER_00")]
    out = merge_segments([seg(5, 6, "b"), seg(0, 1, "a")], turns, min_utterance_sec=0.4)
    assert [u.text for u in out] == ["a", "b"]


def test_空白だけの区間は捨てる():
    turns = [turn(0, 10, "SPEAKER_00")]
    out = merge_segments([seg(0, 1, "a"), seg(1, 2, "  "), seg(2, 3, "b")], turns, min_utterance_sec=0.4)
    assert [u.text for u in out] == ["a", "b"]


def test_入力が空なら空():
    assert merge_segments([], [], 0.4) == []


def test_複数ファイルの時刻を後ろにずらせる():
    shifted = shift_items([seg(1, 2, "a")], 100.0)
    assert (shifted[0].start_sec, shifted[0].end_sec, shifted[0].text) == (101.0, 102.0, "a")
    t = shift_items([turn(0, 1, "SPEAKER_00")], 50.0)
    assert (t[0].start_sec, t[0].end_sec, t[0].speaker_label) == (50.0, 51.0, "SPEAKER_00")
