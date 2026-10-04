from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from kijun.config import Config
from kijun.db import migrate, repo
from kijun.models import Conversation, Utterance


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """パスを一時ディレクトリにした設定。それ以外は既定値。"""
    c = Config()
    c.paths.inbox = tmp_path / "inbox"
    c.paths.processed = tmp_path / "processed"
    c.paths.transcripts = tmp_path / "transcripts"
    c.paths.db = tmp_path / "kijun.duckdb"
    return c


@pytest.fixture
def con(cfg: Config):
    """スキーマ適用済みの一時的な DuckDB 接続。"""
    connection = repo.connect(cfg.paths.db)
    migrate.apply_schema(connection)
    yield connection
    connection.close()


def make_conversation(
    conversation_id: str = "20260920_1930_f2f_teirei",
    source_files: list[str] | None = None,
    transcribed_at: datetime | None = None,
    duration_sec: float | None = 600.0,
) -> Conversation:
    return Conversation(
        conversation_id=conversation_id,
        recorded_at=datetime(2026, 9, 20, 19, 30),
        setting_format="f2f",
        setting_kind="teirei",
        source_files=source_files or [f"{conversation_id}.m4a"],
        transcript_path=None,
        duration_sec=duration_sec,
        transcribed_at=transcribed_at,
        created_at=datetime(2026, 9, 20, 23, 30),
    )


def make_utterances(count: int, seconds_each: float = 60.0, speakers: int = 2) -> list[Utterance]:
    """連続した count 件の発話。話者は交互。"""
    return [
        Utterance(
            seq=i,
            speaker_label=f"SPEAKER_{i % speakers:02d}",
            start_sec=i * seconds_each,
            end_sec=(i + 1) * seconds_each,
            text=f"発話{i}",
        )
        for i in range(count)
    ]
