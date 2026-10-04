"""チャンク分割と、発話番号の復元、重複排除（計画書 7-3、CLAUDE.md 4-4）。

すべて純粋な計算で、LLM にも DB にも触らない。

発話番号の扱い:
- LLM に渡すとき、各チャンクの発話は「チャンク内の番号（0 から始まる）」で番号を振る。
- LLM が返す seq_from / seq_to はチャンク内の番号なので、会議全体の seq に復元する
  （chunk.seq_from を足す）。復元は restore_global_seq が行う。
- 発話の seq は会議内で連続した整数（0, 1, 2, ...）という前提で、チャンク内の番号 n は
  会議全体の seq では chunk.seq_from + n になる。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from kijun.extract.models import ExtractedRequest
from kijun.models import Utterance

# 保険の重複判定: quote の正規化が一致し、seq_from の差がこの値以内なら同一の指摘とみなす
DUPLICATE_SEQ_DISTANCE = 2


@dataclass(frozen=True)
class Chunk:
    """抽出の1回分の入力。utterances は seq の連続した発話。"""

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


def restore_global_seq(item: ExtractedRequest, chunk: Chunk) -> ExtractedRequest | None:
    """LLM が返したチャンク内の番号を、会議全体の seq に復元する。

    番号がチャンクの発話の範囲外、または seq_from が seq_to より大きい場合は None を返す
    （LLM が存在しない番号を書いた、または順序を逆にしたもの。呼び出し側が捨てた件数を数える）。
    """
    size = len(chunk.utterances)
    if not (0 <= item.seq_from < size and 0 <= item.seq_to < size):
        return None
    if item.seq_from > item.seq_to:
        return None
    return item.model_copy(
        update={
            "seq_from": item.seq_from + chunk.seq_from,
            "seq_to": item.seq_to + chunk.seq_from,
        }
    )


def normalize_quote(text: str) -> str:
    """quote の比較用に、空白と句読点（記号・区切り・制御文字）を取り除く。"""
    return "".join(
        c for c in text if unicodedata.category(c)[0] not in ("P", "Z", "C")
    )


@dataclass
class MergeOutcome:
    """複数チャンクの抽出結果を1つにまとめた結果。"""

    items: list[ExtractedRequest]  # 会議全体の seq に復元済み。seq_from の昇順
    invalid_count: int = 0  # 番号が範囲外などで捨てた件数
    overlap_dropped_count: int = 0  # 重複区間で後続チャンクのものとして捨てた件数
    duplicate_count: int = 0  # 保険の重複判定（quote の一致）で捨てた件数
    notes: list[str] = field(default_factory=list)


def merge_chunk_results(
    results: list[tuple[Chunk, list[ExtractedRequest] | None]],
) -> MergeOutcome:
    """チャンクごとの抽出結果を、会議全体の1つのリストにまとめる。

    results は (チャンク, そのチャンクの抽出結果) の並び。チャンクの抽出に失敗した場合は、
    抽出結果の代わりに None を入れる。

    手順:
    1. 各結果の番号を restore_global_seq で会議全体の seq に復元する。範囲外のものは捨てる。
    2. 重複排除（主）: チャンク N とチャンク N+1 の重複区間（N+1 の先頭から N の末尾まで）に
       seq_from が入る結果のうち、後続の N+1 から出たものを捨てる。
       ただしチャンク N の抽出が失敗している場合は、重複区間の結果を捨てない
       （捨てると、その区間の結果がどちらからも得られなくなるため）。
    3. 重複排除（保険）: quote の正規化が一致し、seq_from の差が DUPLICATE_SEQ_DISTANCE 以内の
       ものを同一とみなし、先に出たもの（前のチャンクから出たもの）を残す。
    """
    out = MergeOutcome(items=[])
    kept: list[ExtractedRequest] = []
    previous: tuple[Chunk, bool] | None = None  # (直前のチャンク, 直前のチャンクの抽出が成功したか)

    for chunk, items in results:
        if items is not None:
            for raw in items:
                item = restore_global_seq(raw, chunk)
                if item is None:
                    out.invalid_count += 1
                    continue
                if previous is not None and previous[1]:
                    prev_chunk = previous[0]
                    if chunk.seq_from <= item.seq_from <= prev_chunk.seq_to:
                        out.overlap_dropped_count += 1
                        continue
                norm = normalize_quote(item.quote)
                if any(
                    normalize_quote(k.quote) == norm
                    and abs(k.seq_from - item.seq_from) <= DUPLICATE_SEQ_DISTANCE
                    for k in kept
                ):
                    out.duplicate_count += 1
                    continue
                kept.append(item)
        previous = (chunk, items is not None)

    out.items = sorted(kept, key=lambda i: (i.seq_from, i.seq_to))
    return out
