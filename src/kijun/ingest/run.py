"""ingest サブコマンドの本体。inbox の走査、束ね、conversations への登録。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.ingest.audio import get_duration_sec
from kijun.ingest.bundle import AudioFile, Bundle, bundle_files
from kijun.ingest.filename import warn_stderr, scan_inbox
from kijun.models import Conversation
from kijun.table import format_table


@dataclass
class IngestResult:
    bundles: list[Bundle]
    failed: list[tuple[str, str]]  # (ファイル名, 解析できなかった理由)
    registered: list[str] = field(default_factory=list)  # 新しく登録した conversation_id
    updated: list[str] = field(default_factory=list)  # 文字起こし前の会議のファイル一覧を更新した ID
    skipped: list[str] = field(default_factory=list)  # 文字起こし済みのため触らなかった ID


def collect_bundles(
    cfg: Config,
    warn: Callable[[str], None] = warn_stderr,
    duration_fn: Callable = get_duration_sec,
) -> IngestResult:
    """inbox を走査して束ねる。DB には書かない。

    duration_fn はテストで差し替えるための引数。
    """
    parsed, failed = scan_inbox(cfg.paths.inbox, cfg.settings, warn)
    files: list[AudioFile] = []
    for p in parsed:
        duration = duration_fn(p.path)
        if duration is None:
            warn(f"音声の長さを取得できませんでした（束ねの判定では長さ 0 秒として扱います）: {p.path.name}")
        files.append(AudioFile(parsed=p, duration_sec=duration))
    bundles = bundle_files(files, cfg.bundle)
    return IngestResult(bundles=bundles, failed=[(str(p.name), reason) for p, reason in failed])


def register_bundles(
    con: duckdb.DuckDBPyConnection, result: IngestResult, now: datetime | None = None
) -> None:
    """束ねた結果を conversations に登録する。

    - 未登録の会議は新しく登録する。
    - 登録済みで文字起こし前の会議は、ファイル一覧と合計の長さを更新する
      （後から同じ会議の続きのファイルが届いた場合に備える）。
    - 文字起こし済みの会議は触らない。
    """
    now = now or datetime.now()
    for b in result.bundles:
        existing = repo.get_conversation(con, b.conversation_id)
        if existing is None:
            repo.insert_conversation(
                con,
                Conversation(
                    conversation_id=b.conversation_id,
                    recorded_at=b.recorded_at,
                    setting_format=b.setting_format,
                    setting_kind=b.setting_kind,
                    source_files=b.file_names,
                    transcript_path=None,
                    duration_sec=b.total_duration_sec,
                    transcribed_at=None,
                    created_at=now,
                ),
            )
            result.registered.append(b.conversation_id)
        elif existing.transcribed_at is None:
            if existing.source_files != b.file_names or existing.duration_sec != b.total_duration_sec:
                repo.update_conversation_sources(
                    con, b.conversation_id, b.file_names, b.total_duration_sec
                )
                result.updated.append(b.conversation_id)
        else:
            result.skipped.append(b.conversation_id)


def format_bundles(bundles: list[Bundle]) -> str:
    """束ねた結果を表にする（ingest --dry-run の出力）。"""
    rows = []
    for b in bundles:
        total = b.total_duration_sec
        rows.append(
            [
                b.conversation_id,
                f"{b.recorded_at:%Y-%m-%d %H:%M}",
                f"{b.setting_format}/{b.setting_kind}",
                str(len(b.files)),
                f"{total / 60:.1f}分" if total is not None else "不明",
                ", ".join(b.file_names),
                "長さ不明のファイルあり" if b.has_unknown_duration else "",
            ]
        )
    return format_table(
        ["会議ID", "開始", "形式/種類", "ファイル数", "合計の長さ", "ファイル", "注意"], rows
    )
