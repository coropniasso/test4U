"""会議単位への束ね（CLAUDE.md 4-7、計画書 5-3）。

隣り合う2ファイルが次の条件をすべて満たすとき、同じ会議（conversation_id）にする。
1. 2つの recorded_at の日付が同じ。
2. require_same_setting が true の場合、形式と種類が同じ。
3. 前のファイルの終了時刻（recorded_at + 長さ）から次のファイルの recorded_at までが
   gap_minutes 分以内（ちょうど gap_minutes 分も含む）。重なっている場合も含む。

長さが取れなかったファイルは、終了時刻を recorded_at と同じとみなし、
その会議に has_unknown_duration の印を付ける。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from kijun.config import BundleConfig
from kijun.ingest.filename import ParsedName


@dataclass(frozen=True)
class AudioFile:
    """束ねの入力。解析済みのファイル名と、音声の長さ（取れなければ None）。"""

    parsed: ParsedName
    duration_sec: float | None


@dataclass
class Bundle:
    """1つの会議にまとめたファイルの束。"""

    conversation_id: str
    recorded_at: datetime
    setting_format: str
    setting_kind: str
    files: list[AudioFile] = field(default_factory=list)

    @property
    def has_unknown_duration(self) -> bool:
        """長さが取れなかったファイルを含むか。"""
        return any(f.duration_sec is None for f in self.files)

    @property
    def total_duration_sec(self) -> float | None:
        """長さが取れたファイルの合計。1つも取れなければ None。"""
        known = [f.duration_sec for f in self.files if f.duration_sec is not None]
        return float(sum(known)) if known else None

    @property
    def file_names(self) -> list[str]:
        return [f.parsed.path.name for f in self.files]


def make_conversation_id(parsed: ParsedName) -> str:
    """束の先頭ファイルから `yyyyMMdd_HHmm_<形式>_<種類>` を作る。連番は含めない。"""
    return (
        f"{parsed.recorded_at:%Y%m%d_%H%M}_{parsed.setting_format}_{parsed.setting_kind}"
    )


def _end_time(f: AudioFile) -> datetime:
    """ファイルの終了時刻。長さが無ければ開始時刻と同じとみなす。"""
    return f.parsed.recorded_at + timedelta(seconds=f.duration_sec or 0.0)


def _should_join(prev: AudioFile, nxt: AudioFile, cfg: BundleConfig) -> bool:
    if prev.parsed.recorded_at.date() != nxt.parsed.recorded_at.date():
        return False
    if cfg.require_same_setting and (
        prev.parsed.setting_format != nxt.parsed.setting_format
        or prev.parsed.setting_kind != nxt.parsed.setting_kind
    ):
        return False
    gap = nxt.parsed.recorded_at - _end_time(prev)
    return gap <= timedelta(minutes=cfg.gap_minutes)


def bundle_files(files: list[AudioFile], cfg: BundleConfig) -> list[Bundle]:
    """ファイルを recorded_at の昇順に並べ、会議単位の束にして返す。"""
    ordered = sorted(files, key=lambda f: (f.parsed.recorded_at, f.parsed.part or 0, f.parsed.path.name))
    bundles: list[Bundle] = []
    for f in ordered:
        if bundles and _should_join(bundles[-1].files[-1], f, cfg):
            bundles[-1].files.append(f)
            continue
        bundles.append(
            Bundle(
                conversation_id=make_conversation_id(f.parsed),
                recorded_at=f.parsed.recorded_at,
                setting_format=f.parsed.setting_format,
                setting_kind=f.parsed.setting_kind,
                files=[f],
            )
        )
    return bundles
