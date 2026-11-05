"""文字起こしの区間と話者の区間の突き合わせ（計画書 6-4）。

純粋な計算だけで、GPU にも外部ファイルにも触らない。
"""

from __future__ import annotations

from dataclasses import dataclass

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
    max_utterance_sec: float = 0.0,
) -> list[Utterance]:
    """文字起こしの区間に話者を割り当て、同一話者の連続する区間を結合して Utterance にする。

    手順:
    1. 本文が空白だけの区間を捨てる。開始時刻の昇順に並べる。
    2. 各区間に、重なりが最長の話者を割り当てる（重なりが無ければ None）。
    3. 直前の区間と speaker_label が同じで、かつ「前の区間の終了から次の区間の開始までの間隔」が
       min_utterance_sec 未満なら、1つの発話に結合する。結合した本文は、日本語の文字起こしなので
       区切りを入れずにつなぐ。speaker_label が None の区間は、同じ話者だと判断できないので結合しない。
       max_utterance_sec が 0 より大きい場合は、結合後の長さ（新しい終了時刻 - 開始時刻）がそれを
       超えるなら結合せず、次の区間から新しい発話を始める。0 は上限なし。
       1つの文字起こし区間そのものが max_utterance_sec より長い場合は、分割せずそのまま1つの発話にする。
       word_timestamps=False で動かしているため区間の内部の時刻が分からず、分割すると文字と時刻の
       対応が壊れるからである。
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
            new_end = max(prev[2], seg.end_sec)
            within_max = max_utterance_sec <= 0 or (new_end - prev[1]) <= max_utterance_sec
            if label is not None and label == prev[0] and gap < min_utterance_sec and within_max:
                prev[2] = new_end
                prev[3] = prev[3] + text
                continue
        merged.append([label, seg.start_sec, seg.end_sec, text])
    return [
        Utterance(seq=i, speaker_label=m[0], start_sec=m[1], end_sec=m[2], text=m[3])
        for i, m in enumerate(merged)
    ]


@dataclass(frozen=True)
class DroppedSegment:
    """幻聴の疑いで除いた文字起こしの区間。文字起こしの JSON の dropped_segments に残す。"""

    start_sec: float
    end_sec: float
    text: str
    avg_logprob: float | None
    speaker_overlap: float
    reason: str


def speaker_overlap_ratio(segment: TranscribedSegment, turns: list[SpeakerTurn]) -> float:
    """区間のうち、いずれかの話者区間に覆われている部分の割合（0〜1）。

    話者区間どうしが重なることがあるので、話者区間の和集合で数える（重なりの秒数を足し合わせると
    1 を超える）。区間の長さが 0 以下なら 0 を返す。
    """
    duration = segment.end_sec - segment.start_sec
    if duration <= 0:
        return 0.0
    clipped = sorted(
        (max(t.start_sec, segment.start_sec), min(t.end_sec, segment.end_sec))
        for t in turns
        if overlap_sec(segment.start_sec, segment.end_sec, t.start_sec, t.end_sec) > 0
    )
    covered = 0.0
    cur_start: float | None = None
    cur_end = 0.0
    for s, e in clipped:
        if cur_start is None:
            cur_start, cur_end = s, e
        elif s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            covered += cur_end - cur_start
            cur_start, cur_end = s, e
    if cur_start is not None:
        covered += cur_end - cur_start
    return min(1.0, covered / duration)


def filter_hallucinated_segments(
    segments: list[TranscribedSegment],
    turns: list[SpeakerTurn],
    max_speaker_overlap: float,
    max_avg_logprob: float,
) -> tuple[list[TranscribedSegment], list[DroppedSegment]]:
    """幻聴の疑いがある区間を (残す区間, 除いた区間) に分ける。

    除く条件：話者区間との重なりの割合が max_speaker_overlap 未満、かつ avg_logprob が
    max_avg_logprob 未満（どちらも「未満」。ちょうど閾値の値は残す）。
    判断できない場合は除かない：turns が空（話者分離の結果が無い）、avg_logprob が None、
    max_speaker_overlap が 0 以下（除去を使わない設定）。
    """
    if not turns or max_speaker_overlap <= 0:
        return list(segments), []
    kept: list[TranscribedSegment] = []
    dropped: list[DroppedSegment] = []
    for seg in segments:
        if seg.avg_logprob is None:
            kept.append(seg)
            continue
        ratio = speaker_overlap_ratio(seg, turns)
        if ratio < max_speaker_overlap and seg.avg_logprob < max_avg_logprob:
            dropped.append(
                DroppedSegment(
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    text=seg.text,
                    avg_logprob=seg.avg_logprob,
                    speaker_overlap=ratio,
                    reason=(
                        f"話者区間との重なり {ratio:.2f} < {max_speaker_overlap:g} "
                        f"かつ avg_logprob {seg.avg_logprob:.2f} < {max_avg_logprob:g}"
                    ),
                )
            )
        else:
            kept.append(seg)
    return kept, dropped
