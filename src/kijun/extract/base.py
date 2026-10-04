"""抽出のインターフェース。

Ollama と Claude API は、この Protocol の背後に置く。テストでは tests/fakes.py の FakeExtractor を使う。
"""

from __future__ import annotations

from typing import Protocol

from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractionResult


class ExtractionError(Exception):
    """1チャンクの抽出に失敗したときの例外（JSON が壊れていた、拒否された、通信に失敗した等）。

    呼び出し側（extract/run.py）はこの例外を捕まえ、そのチャンクを失敗として記録して
    次のチャンクに進む。会議全体は落とさない。
    """


class Extractor(Protocol):
    # request_id と extracted_requests.model_name に使うモデル名（例: qwen3:14b）
    model_name: str

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """チャンクから要求・指摘を抽出する。

        返す seq_from / seq_to は、チャンク内の番号（0 から始まる）。
        会議全体の seq への復元は chunk.py が行う。失敗したら ExtractionError を出す。
        """
        ...
