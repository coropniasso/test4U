"""音声の30日削除（作業表 #9、計画書 第10節）。

CLAUDE.md 2章の参加者への説明「音声は文字起こし完了後30日で削除」を守るための実装。

- conversations.transcribed_at から [retention].audio_days 日が経った会議を対象にする。
- 対象の会議の source_files のうち、[paths].processed に実在するファイルを削除対象にする。
- 既定では削除せず一覧だけを出す。削除は execute=True（CLI の --execute）のときだけ行う。
  録音は録り直せないため。
- 削除するのは音声だけ。文字起こしの JSON（[paths].transcripts）は削除しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.table import format_table


@dataclass(frozen=True)
class PurgeCandidate:
    conversation_id: str
    file_name: str
    path: Path
    transcribed_at: datetime
    elapsed_days: int  # 文字起こし完了から経過した日数（切り捨て）


def find_purge_candidates(
    con: duckdb.DuckDBPyConnection, cfg: Config, now: datetime | None = None
) -> list[PurgeCandidate]:
    """削除の対象になる音声ファイルを返す。ファイルには何もしない。

    経過日数が audio_days 日以上（ちょうど audio_days 日を含む）の会議が対象。
    processed に実在しないファイル（まだ inbox にある、すでに消してある）は含めない。
    source_files にディレクトリ区切りを含む名前があっても、processed の外は対象にしない。
    """
    now = now or datetime.now()
    cutoff = now - timedelta(days=cfg.retention.audio_days)
    candidates: list[PurgeCandidate] = []
    for conv in repo.list_transcribed_before(con, cutoff):
        assert conv.transcribed_at is not None  # list_transcribed_before の条件で保証される
        elapsed = (now - conv.transcribed_at).days
        for name in conv.source_files:
            if Path(name).name != name:
                continue
            path = cfg.paths.processed / name
            if path.is_file():
                candidates.append(
                    PurgeCandidate(conv.conversation_id, name, path, conv.transcribed_at, elapsed)
                )
    return candidates


def format_candidates(candidates: list[PurgeCandidate]) -> str:
    """削除対象を表にする。列は、ファイル名・会議ID・文字起こし完了日・経過日数。"""
    rows = [
        [c.file_name, c.conversation_id, f"{c.transcribed_at:%Y-%m-%d}", str(c.elapsed_days)]
        for c in candidates
    ]
    return format_table(["ファイル名", "会議ID", "文字起こし完了日", "経過日数"], rows)


def purge_candidates(candidates: list[PurgeCandidate]) -> tuple[list[PurgeCandidate], list[tuple[PurgeCandidate, str]]]:
    """候補のファイルを実際に削除する。戻り値は (削除できたもの, [(削除できなかったもの, 理由)])。"""
    deleted: list[PurgeCandidate] = []
    failed: list[tuple[PurgeCandidate, str]] = []
    for c in candidates:
        try:
            c.path.unlink()
            deleted.append(c)
        except OSError as e:
            failed.append((c, str(e)))
    return deleted, failed
