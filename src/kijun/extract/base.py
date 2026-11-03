"""抽出のインターフェース。

推論サーバー（Strata）に触る実装は、この Protocol の背後に置く。
テストでは tests/fakes.py の FakeExtractor を使う。

Protocol に enforces_json_schema と max_input_tokens を持たせる理由：
背面がスキーマへの適合を保証するかどうか、1リクエストに何トークン入るかは、背面によって違う。
Strata がどちらを満たすかは確認できていない（`uv run kijun doctor` で実測する）。
背面ごとの挙動を呼び出し側にハードコードせず、実装に宣言させる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractionResult


class ExtractionError(Exception):
    """1チャンクの抽出に失敗したときの例外（JSON が壊れていた、通信に失敗した等）。

    呼び出し側（extract/run.py）はこの例外を捕まえ、そのチャンクを失敗として記録して
    次のチャンクに進む。会議全体は落とさない。
    """


@dataclass(frozen=True)
class RepairEvent:
    """1チャンクの出力に対して行った修復の記録。

    run.py がこれを extract_logs の JSON に書く。背面が壊れた JSON を返していることに
    ユーザーが気付けるようにするため。
    """

    chunk_index: int
    seq_from: int
    seq_to: int
    actions: list[str]  # repair.ACTION_* の識別子
    truncated_items: int
    attempt: int  # 何回目の試行で読めたか（1 または 2）


class Extractor(Protocol):
    # request_id と extracted_requests.model_name に使うモデル名（例: qwen3.8-flash-next）
    model_name: str
    # サーバー側が JSON スキーマへの適合を保証するか。
    # False のとき、呼び出し側は repair.py を通してから検証する。
    enforces_json_schema: bool
    # 1リクエストで受け付ける入力トークンの上限。未確認なら None。
    # None のときは chunk_minutes の妥当性を検査しない。
    max_input_tokens: int | None
    # 出力を修復した記録。実装は空のリストで初期化し、修復するたびに追加する。
    # run.py がこれを読んで extract_logs に書く。
    repair_events: list[RepairEvent]

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """チャンクから要求・指摘を抽出する。

        返す seq_from / seq_to は、チャンク内の番号（0 から始まる）。
        会議全体の seq への復元は chunk.py が行う。失敗したら ExtractionError を出す。
        """
        ...
