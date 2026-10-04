"""文字起こしの区間と話者の区間の突き合わせ（計画書 6-4）。

純粋な計算だけで、GPU にも外部ファイルにも触らない。
"""

from __future__ import annotations

from kijun.models import Utterance
from kijun.transcribe.base import SpeakerTurn, TranscribedSegment


def overlap_sec(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """2つの区間が重なっている秒数。重なりが無ければ 0。"""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(segment: TranscribedSegment, turns: list[SpeakerTurn]) -> str | None:
    """segment と重なりが最も長い話者の speaker_label を返す。

    重なりが全く無ければ None を返す。重なりが同じ長さの話者が複数いる場合は、
    turns の並びで先にある方を返す。
    """
    best_label: str | None = None
    best_overlap = 0.0
    for t in turns:
        ov = overlap_sec(segment.start_sec, segment.end_sec, t.start_sec, t.end_sec)
        if ov > best_overlap:
            best_overlap = ov
            best_label = t.speaker_label
    return best_label


def merge_segments(
    segments: list[TranscribedSegment],
    turns: list[SpeakerTurn],
    min_utterance_sec: float,
) -> list[Utterance]:
    """文字起こしの区間に話者を割り当て、同一話者の連続する区間を結合して Utterance にする。

    手順:
    1. 本文が空白だけの区間を捨てる。開始時刻の昇順に並べる。
    2. 各区間に、重なりが最長の話者を割り当てる（重なりが無ければ None）。
    3. 直前の区間と speaker_label が同じで、かつ「前の区間の終了から次の区間の開始までの間隔」が
       min_utterance_sec 未満なら、1つの発話に結合する。結合した本文は、日本語の文字起こしなので
       区切りを入れずにつなぐ。speaker_label が None の区間は、同じ話者だと判断できないので結合しない。
    4. seq を 0 から順に振る。
    """
    ordered = sorted(
        (s for s in segments if s.text.strip()), key=lambda s: (s.start_sec, s.end_sec)
    )
    merged: list[list] = []  # [speaker_label, start_sec, end_sec, text]
    for seg in ordered:
        label = assign_speaker(seg, turns)
        text = seg.text.strip()
        if merged:
            prev = merged[-1]
            gap = seg.start_sec - prev[2]
            if label is not None and label == prev[0] and gap < min_utterance_sec:
                prev[2] = max(prev[2], seg.end_sec)
                prev[3] = prev[3] + text
                continue
        merged.append([label, seg.start_sec, seg.end_sec, text])
    return [
        Utterance(seq=i, speaker_label=m[0], start_sec=m[1], end_sec=m[2], text=m[3])
        for i, m in enumerate(merged)
    ]
