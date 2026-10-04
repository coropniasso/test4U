"""文字起こしと話者分離のインターフェース（計画書 6-1）。

GPU を使う実装は、この Protocol の背後に置く。テストでは tests/fakes.py の偽の実装を使う。
複数ファイルを渡せるのは、会議単位に束ねた結果が複数ファイルから成るため。
2本目以降の時刻は、それより前のファイルの長さを足してつなげる。
文字起こしと話者分離で時刻の基準をそろえるため、長さは音声をデコードした結果
（faster-whisper の info.duration と、デコードしたサンプル数）から求める。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeVar


@dataclass(frozen=True)
class TranscribedSegment:
    """文字起こしの1区間。時刻は、束ねた会議の先頭からの秒数。"""

    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class SpeakerTurn:
    """話者分離の1区間。時刻は、束ねた会議の先頭からの秒数。"""

    start_sec: float
    end_sec: float
    speaker_label: str


class Transcriber(Protocol):
    def transcribe(self, audio_paths: list[Path]) -> list[TranscribedSegment]: ...


class Diarizer(Protocol):
    def diarize(self, audio_paths: list[Path]) -> list[SpeakerTurn]: ...


T = TypeVar("T", TranscribedSegment, SpeakerTurn)


def shift_items(items: list[T], offset_sec: float) -> list[T]:
    """区間の時刻を offset_sec だけ後ろにずらした新しいリストを返す。"""
    return [replace(i, start_sec=i.start_sec + offset_sec, end_sec=i.end_sec + offset_sec) for i in items]
