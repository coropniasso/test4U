"""各テーブルへの書き込み・読み出し関数（conversations、utterances、minutes）。

SQL は schema.sql のテーブル定義に合わせる。接続は自動コミットで使う。
DuckDB は、同じトランザクション内で同じ主キーの行を DELETE してから INSERT すると
主キー違反になる。そのため「消してから入れ直す」処理は、トランザクションを張らず、
DELETE と INSERT を別々に実行する（自動コミット）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from kijun.models import Conversation, Utterance


@dataclass(frozen=True)
class MinutesRow:
    """minutes テーブルの1行。"""

    conversation_id: str
    path: str
    model_name: str
    created_at: datetime
    discord_message_id: str | None
    posted_at: datetime | None


_CONVERSATION_COLUMNS = (
    "conversation_id, recorded_at, setting_format, setting_kind, source_files, "
    "transcript_path, duration_sec, transcribed_at, created_at"
)


def connect(db_path: Path | str, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """DuckDB に接続する。保存先のディレクトリが無ければ作る（読み取り専用では作らない）。"""
    path = Path(db_path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


# --- conversations -----------------------------------------------------------


def insert_conversation(con: duckdb.DuckDBPyConnection, conv: Conversation) -> None:
    con.execute(
        f"INSERT INTO conversations ({_CONVERSATION_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            conv.conversation_id,
            conv.recorded_at,
            conv.setting_format,
            conv.setting_kind,
            conv.source_files,
            conv.transcript_path,
            conv.duration_sec,
            conv.transcribed_at,
            conv.created_at,
        ],
    )


def replace_conversation(con: duckdb.DuckDBPyConnection, conv: Conversation) -> None:
    """同じ conversation_id の行があれば消してから入れ直す（restore-transcript --force 用）。"""
    con.execute("DELETE FROM conversations WHERE conversation_id = ?", [conv.conversation_id])
    insert_conversation(con, conv)


def _to_conversation(row: tuple) -> Conversation:
    return Conversation(
        conversation_id=row[0],
        recorded_at=row[1],
        setting_format=row[2],
        setting_kind=row[3],
        source_files=list(row[4]),
        transcript_path=row[5],
        duration_sec=row[6],
        transcribed_at=row[7],
        created_at=row[8],
    )


def get_conversation(con: duckdb.DuckDBPyConnection, conversation_id: str) -> Conversation | None:
    row = con.execute(
        f"SELECT {_CONVERSATION_COLUMNS} FROM conversations WHERE conversation_id = ?",
        [conversation_id],
    ).fetchone()
    return _to_conversation(row) if row else None


def list_conversations(con: duckdb.DuckDBPyConnection) -> list[Conversation]:
    rows = con.execute(
        f"SELECT {_CONVERSATION_COLUMNS} FROM conversations ORDER BY recorded_at"
    ).fetchall()
    return [_to_conversation(r) for r in rows]


def update_conversation_sources(
    con: duckdb.DuckDBPyConnection,
    conversation_id: str,
    source_files: list[str],
    duration_sec: float | None,
) -> None:
    """文字起こし前の会議について、束ねたファイル一覧と合計の長さを更新する。"""
    con.execute(
        "UPDATE conversations SET source_files = ?, duration_sec = ? WHERE conversation_id = ?",
        [source_files, duration_sec, conversation_id],
    )


def mark_transcribed(
    con: duckdb.DuckDBPyConnection,
    conversation_id: str,
    transcript_path: str,
    duration_sec: float | None,
    transcribed_at: datetime,
) -> None:
    """文字起こしの完了を記録する。transcribed_at は音声の30日削除の判定に使う。"""
    con.execute(
        "UPDATE conversations SET transcript_path = ?, duration_sec = ?, transcribed_at = ? "
        "WHERE conversation_id = ?",
        [transcript_path, duration_sec, transcribed_at, conversation_id],
    )


def list_transcribed_before(
    con: duckdb.DuckDBPyConnection, cutoff: datetime
) -> list[Conversation]:
    """transcribed_at が cutoff 以前の会議を返す。"""
    rows = con.execute(
        f"SELECT {_CONVERSATION_COLUMNS} FROM conversations "
        "WHERE transcribed_at IS NOT NULL AND transcribed_at <= ? ORDER BY transcribed_at",
        [cutoff],
    ).fetchall()
    return [_to_conversation(r) for r in rows]


# --- utterances --------------------------------------------------------------


def replace_utterances(
    con: duckdb.DuckDBPyConnection, conversation_id: str, utterances: list[Utterance]
) -> None:
    """会議の発話を入れ直す。文字起こしをやり直したときに古い行を残さないため。"""
    con.execute("DELETE FROM utterances WHERE conversation_id = ?", [conversation_id])
    if utterances:
        con.executemany(
            "INSERT INTO utterances (conversation_id, seq, speaker_label, start_sec, end_sec, text) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                [conversation_id, u.seq, u.speaker_label, u.start_sec, u.end_sec, u.text]
                for u in utterances
            ],
        )


def get_utterances(con: duckdb.DuckDBPyConnection, conversation_id: str) -> list[Utterance]:
    rows = con.execute(
        "SELECT seq, speaker_label, start_sec, end_sec, text FROM utterances "
        "WHERE conversation_id = ? ORDER BY seq",
        [conversation_id],
    ).fetchall()
    return [Utterance(*r) for r in rows]


# --- minutes -----------------------------------------------------------------

_MINUTES_COLUMNS = "conversation_id, path, model_name, created_at, discord_message_id, posted_at"


def upsert_minutes(
    con: duckdb.DuckDBPyConnection, conversation_id: str, path: str, model_name: str, created_at: datetime
) -> None:
    """議事録を記録する。既にあれば path・model_name・created_at を更新する。

    Discord への投稿の記録（discord_message_id、posted_at）は変えない。そのため、作り直した議事録は
    自動では再投稿されない。
    """
    exists = con.execute(
        "SELECT count(*) FROM minutes WHERE conversation_id = ?", [conversation_id]
    ).fetchone()[0]
    if exists:
        con.execute(
            "UPDATE minutes SET path = ?, model_name = ?, created_at = ? WHERE conversation_id = ?",
            [path, model_name, created_at, conversation_id],
        )
    else:
        con.execute(
            f"INSERT INTO minutes ({_MINUTES_COLUMNS}) VALUES (?, ?, ?, ?, NULL, NULL)",
            [conversation_id, path, model_name, created_at],
        )


def get_minutes(con: duckdb.DuckDBPyConnection, conversation_id: str) -> MinutesRow | None:
    row = con.execute(
        f"SELECT {_MINUTES_COLUMNS} FROM minutes WHERE conversation_id = ?", [conversation_id]
    ).fetchone()
    return MinutesRow(*row) if row else None


def list_unposted_minutes(con: duckdb.DuckDBPyConnection) -> list[MinutesRow]:
    """Discord に未投稿（posted_at が NULL）の議事録。作成の古い順。"""
    rows = con.execute(
        f"SELECT {_MINUTES_COLUMNS} FROM minutes WHERE posted_at IS NULL ORDER BY created_at, conversation_id"
    ).fetchall()
    return [MinutesRow(*r) for r in rows]


def mark_minutes_posted(
    con: duckdb.DuckDBPyConnection, conversation_id: str, message_id: str, at: datetime
) -> None:
    con.execute(
        "UPDATE minutes SET discord_message_id = ?, posted_at = ? WHERE conversation_id = ?",
        [message_id, at, conversation_id],
    )


def list_conversations_without_minutes(con: duckdb.DuckDBPyConnection) -> list[Conversation]:
    """文字起こしが済んでいて、議事録がまだ無い会議。録音の古い順。"""
    rows = con.execute(
        f"SELECT {_CONVERSATION_COLUMNS} "
        "FROM conversations c WHERE c.transcribed_at IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM minutes m WHERE m.conversation_id = c.conversation_id) "
        "ORDER BY c.recorded_at"
    ).fetchall()
    return [_to_conversation(r) for r in rows]
