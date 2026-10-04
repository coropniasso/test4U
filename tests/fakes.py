"""テスト用の偽の実装。GPU、ネットワーク、外部プロセスに触らずに、本物と同じインターフェースを満たす。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from kijun.extract.base import ExtractionError
from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractedRequest, ExtractionResult
from kijun.transcribe.base import SpeakerTurn, TranscribedSegment


class FakeTranscriber:
    """渡された区間をそのまま返す。呼ばれた audio_paths を記録する。"""

    def __init__(self, segments: list[TranscribedSegment]) -> None:
        self._segments = segments
        self.calls: list[list[Path]] = []

    def transcribe(self, audio_paths: list[Path]) -> list[TranscribedSegment]:
        self.calls.append(list(audio_paths))
        return list(self._segments)


class FakeDiarizer:
    """渡された話者区間をそのまま返す。"""

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self._turns = turns
        self.calls: list[list[Path]] = []

    def diarize(self, audio_paths: list[Path]) -> list[SpeakerTurn]:
        self.calls.append(list(audio_paths))
        return list(self._turns)


class FakeExtractor:
    """チャンクごとに決めた抽出結果を返す。

    responses は、チャンクの番号（chunk.index）から「そのチャンク内の番号で書いた抽出結果」への辞書。
    無いチャンクは空の結果を返す。fail_chunks に入れた番号のチャンクは ExtractionError を出す。
    handler を渡すと、それを優先して使う。
    """

    def __init__(
        self,
        responses: dict[int, list[ExtractedRequest]] | None = None,
        fail_chunks: set[int] | None = None,
        model_name: str = "fake-model",
        handler: Callable[[Chunk], ExtractionResult] | None = None,
    ) -> None:
        self.model_name = model_name
        self._responses = responses or {}
        self._fail_chunks = fail_chunks or set()
        self._handler = handler
        self.chunks_seen: list[Chunk] = []

    def extract(self, chunk: Chunk) -> ExtractionResult:
        self.chunks_seen.append(chunk)
        if chunk.index in self._fail_chunks:
            raise ExtractionError(f"偽の失敗（チャンク {chunk.index}）")
        if self._handler is not None:
            return self._handler(chunk)
        return ExtractionResult(items=list(self._responses.get(chunk.index, [])))


class FakeEmbedder:
    """文字列のハッシュから決定的にベクトルを作る。

    - 同じ文字列には、必ず同じベクトルを返す（hashlib.md5 を使う。Python の hash() は
      実行ごとに値が変わるので使わない）。
    - 似た文字列には、似たベクトルを返す。文字の2-gram（連続する2文字）を、ハッシュで
      決めた次元に符号付きで加算して作るので、共通する2-gram が多いほど内積が大きくなる。
    - queries と passages は同じ計算にする。呼ばれた文字列を記録するので、再計算の有無を検査できる。
    """

    def __init__(self, dim: int = 256, model_name: str = "fake-embedder") -> None:
        self.dim = dim
        self.model_name = model_name
        self.query_texts: list[str] = []
        self.passage_texts: list[str] = []

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        padded = f"^{text}$"
        for i in range(len(padded) - 1):
            digest = hashlib.md5(padded[i : i + 2].encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            v[index] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self.query_texts.extend(texts)
        return self._encode(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        self.passage_texts.extend(texts)
        return self._encode(texts)

    @property
    def encoded_count(self) -> int:
        return len(self.query_texts) + len(self.passage_texts)


def make_item(
    seq_from: int,
    seq_to: int | None = None,
    quote: str = "引用",
    summary: str = "要約",
    reason: str | None = None,
    applies_when: str | None = None,
    reusable: bool = True,
    criterion: str | None = None,
    speaker_from: str = "SPEAKER_00",
    speaker_to: str | None = "SPEAKER_01",
    artifact_type: str | None = None,
) -> ExtractedRequest:
    """テスト用の ExtractedRequest を作る。"""
    return ExtractedRequest(
        seq_from=seq_from,
        seq_to=seq_from if seq_to is None else seq_to,
        speaker_from_label=speaker_from,
        speaker_to_label=speaker_to,
        counterpart_role=None,
        artifact_type=artifact_type,
        quote=quote,
        request_summary=summary,
        stated_reason=reason,
        applies_when=applies_when,
        reusable=reusable,
        pass_criterion=criterion,
    )
