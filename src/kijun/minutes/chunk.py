"""発話のチャンク分割（議事録の生成）。

純粋な計算で、LLM にも DB にも触らない。議事録は重なり 0 で分ける（minutes/generate.py）。
重ねる指定もできるが、重ねた区間の結果の重複排除はここに無い（議事録では使わない）。
"""

from __future__ import annotations

from dataclasses import dataclass

from kijun.models import Utterance


@dataclass(frozen=True)
class Chunk:
    """LLM への1回分の入力。utterances は seq の連続した発話。"""

    index: int  # 0 から始まるチャンクの番号
    seq_from: int  # 先頭の発話の seq（会議全体の番号）
    seq_to: int  # 末尾の発話の seq（会議全体の番号）
    utterances: list[Utterance]

    @property
    def start_sec(self) -> float:
        return self.utterances[0].start_sec

    @property
    def end_sec(self) -> float:
        return self.utterances[-1].end_sec


def split_into_chunks(
    utterances: list[Utterance], chunk_minutes: float, overlap_minutes: float
) -> list[Chunk]:
    """発話を、およそ chunk_minutes 分ごとのチャンクに分ける。隣り合うチャンクは overlap_minutes 分ほど重ねる。

    - チャンクの境界は発話の境界に丸める。発話を途中で切らない。
    - チャンクの先頭の発話の開始時刻から chunk_minutes 分以内に終わる発話を、できるだけ多く入れる。
    - 次のチャンクは、直前のチャンクの終了時刻の overlap_minutes 分前以降に始まる最初の発話から始める。
    - 1つの発話が chunk_minutes を超える長さ（長い独話）の場合は、その発話だけで1チャンクにする。
      発話は分割しない。
    - 次のチャンクが直前のチャンクに完全に含まれてしまう場合（直後に長い独話が来るとき）は、
      重ねずに、直前のチャンクの次の発話から始める。
    """
    if not utterances:
        return []
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes は 0 より大きくしてください")
    if overlap_minutes < 0 or overlap_minutes >= chunk_minutes:
        raise ValueError("overlap_minutes は 0 以上、chunk_minutes 未満にしてください")

    chunk_sec = chunk_minutes * 60.0
    overlap_sec = overlap_minutes * 60.0
    n = len(utterances)

    def last_index_for(start: int) -> int:
        """start から始めたとき、チャンクに入る最後の発話の添字。先頭の発話は必ず入れる。"""
        limit = utterances[start].start_sec + chunk_sec
        last = start
        while last + 1 < n and utterances[last + 1].end_sec <= limit:
            last += 1
        return last

    chunks: list[Chunk] = []
    start = 0
    while True:
        end = last_index_for(start)
        chunks.append(
            Chunk(
                index=len(chunks),
                seq_from=utterances[start].seq,
                seq_to=utterances[end].seq,
                utterances=utterances[start : end + 1],
            )
        )
        if end == n - 1:
            break
        # 次の開始位置: 直前のチャンクの終了時刻の overlap_sec 前以降に始まる最初の発話。
        threshold = utterances[end].end_sec - overlap_sec
        next_start = start + 1
        while next_start < n and utterances[next_start].start_sec < threshold:
            next_start += 1
        next_start = min(next_start, end + 1)
        # 次のチャンクが直前のチャンクに完全に含まれる場合は、重ねずに続きから始める。
        if last_index_for(next_start) <= end:
            next_start = end + 1
        start = next_start
    return chunks
