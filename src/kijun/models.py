"""複数のパッケージが共有するデータ型。

transcribe と extract は互いを import しない（VRAM の境界をプロセスの境界にするため）。
両方が使う Utterance はここに置く。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Utterance:
    """発話1件。seq は会議内で 0 から始まる通し番号。"""

    seq: int
    speaker_label: str | None
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class Conversation:
    """conversations テーブルの1行。"""

    conversation_id: str
    recorded_at: datetime
    setting_format: str
    setting_kind: str
    source_files: list[str]
    transcript_path: str | None
    duration_sec: float | None
    transcribed_at: datetime | None
    created_at: datetime
