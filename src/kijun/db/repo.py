"""各テーブルへの書き込み・読み出し関数。

SQL は schema.sql のテーブル定義に合わせる。接続は自動コミットで使う。
DuckDB は、同じトランザクション内で同じ主キーの行を DELETE してから INSERT すると
主キー違反になる。そのため「消してから入れ直す」処理は、トランザクションを張らず、
DELETE と INSERT を別々に実行する（自動コミット）。
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np

from kijun.models import Conversation, Utterance

# status の保存値（schema.sql の knowledge_items.status と一致させる）
STATUS_CANDIDATE = "candidate"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_HELD = "held"

# embeddings.owner_type の保存値
OWNER_KNOWLEDGE_ITEM = "knowledge_item"
OWNER_EXTRACTED_REQUEST = "extracted_request"

SOURCE_LIVE_CONVERSATION = "live_conversation"
SOURCE_DICTATED_REVIEW = "dictated_review"


@dataclass(frozen=True)
class RequestRow:
    """extracted_requests テーブルの1行。"""

    request_id: str
    conversation_id: str
    seq_from: int
    seq_to: int
    speaker_from_label: str | None
    speaker_to_label: str | None
    direction: str | None
    counterpart_role: str | None
    artifact_type: str | None
    quote: str
    request_summary: str
    stated_reason: str | None
    applies_when: str | None
    reusable: bool
    pass_criterion: str | None
    source_type: str
    model_name: str
    extracted_at: datetime


@dataclass(frozen=True)
class KnowledgeItemRow:
    """knowledge_items テーブルの1行。"""

    item_id: str
    artifact_type: str | None
    check_text: str
    pass_criterion: str | None
    reason: str | None
    applies_when: str | None
    not_applies_when: str | None
    importance: int | None
    status: str
    version: int
    updated_at: datetime


_REQUEST_COLUMNS = (
    "request_id, conversation_id, seq_from, seq_to, speaker_from_label, speaker_to_label, "
    "direction, counterpart_role, artifact_type, quote, request_summary, stated_reason, "
    "applies_when, reusable, pass_criterion, source_type, model_name, extracted_at"
)

_ITEM_COLUMNS = (
    "item_id, artifact_type, check_text, pass_criterion, reason, applies_when, "
    "not_applies_when, importance, status, version, updated_at"
)

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


# --- extracted_requests ------------------------------------------------------


def delete_requests(
    con: duckdb.DuckDBPyConnection, conversation_id: str, model_name: str
) -> int:
    """会議とモデルを指定して抽出結果を消し、消した件数を返す（再抽出の前に使う）。

    item_evidence に紐付いている抽出結果は、消すと根拠の参照先が無くなる。
    その場合は呼び出し側が先に確認する。
    """
    n = con.execute(
        "SELECT count(*) FROM extracted_requests WHERE conversation_id = ? AND model_name = ?",
        [conversation_id, model_name],
    ).fetchone()[0]
    con.execute(
        "DELETE FROM extracted_requests WHERE conversation_id = ? AND model_name = ?",
        [conversation_id, model_name],
    )
    return n


def count_linked_requests(
    con: duckdb.DuckDBPyConnection, conversation_id: str, model_name: str
) -> int:
    """会議とモデルを指定した抽出結果のうち、item_evidence に紐付いている件数を返す。"""
    return con.execute(
        "SELECT count(*) FROM extracted_requests r "
        "WHERE r.conversation_id = ? AND r.model_name = ? "
        "AND EXISTS (SELECT 1 FROM item_evidence e WHERE e.request_id = r.request_id)",
        [conversation_id, model_name],
    ).fetchone()[0]


def insert_requests(con: duckdb.DuckDBPyConnection, rows: list[RequestRow]) -> None:
    if not rows:
        return
    con.executemany(
        f"INSERT INTO extracted_requests ({_REQUEST_COLUMNS}) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            [
                r.request_id,
                r.conversation_id,
                r.seq_from,
                r.seq_to,
                r.speaker_from_label,
                r.speaker_to_label,
                r.direction,
                r.counterpart_role,
                r.artifact_type,
                r.quote,
                r.request_summary,
                r.stated_reason,
                r.applies_when,
                r.reusable,
                r.pass_criterion,
                r.source_type,
                r.model_name,
                r.extracted_at,
            ]
            for r in rows
        ],
    )


def get_requests(
    con: duckdb.DuckDBPyConnection, conversation_id: str, model_name: str | None = None
) -> list[RequestRow]:
    sql = f"SELECT {_REQUEST_COLUMNS} FROM extracted_requests WHERE conversation_id = ?"
    params: list = [conversation_id]
    if model_name is not None:
        sql += " AND model_name = ?"
        params.append(model_name)
    sql += " ORDER BY model_name, seq_from, request_id"
    return [RequestRow(*r) for r in con.execute(sql, params).fetchall()]


def list_request_models(con: duckdb.DuckDBPyConnection, conversation_id: str) -> list[str]:
    """会議に抽出結果を持つモデル名の一覧を、名前順で返す。"""
    rows = con.execute(
        "SELECT DISTINCT model_name FROM extracted_requests WHERE conversation_id = ? "
        "ORDER BY model_name",
        [conversation_id],
    ).fetchall()
    return [r[0] for r in rows]


# --- knowledge_items / item_evidence -----------------------------------------


def insert_knowledge_item(con: duckdb.DuckDBPyConnection, item: KnowledgeItemRow) -> None:
    con.execute(
        f"INSERT INTO knowledge_items ({_ITEM_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            item.item_id,
            item.artifact_type,
            item.check_text,
            item.pass_criterion,
            item.reason,
            item.applies_when,
            item.not_applies_when,
            item.importance,
            item.status,
            item.version,
            item.updated_at,
        ],
    )


def list_knowledge_items(
    con: duckdb.DuckDBPyConnection, statuses: list[str] | None = None
) -> list[KnowledgeItemRow]:
    sql = f"SELECT {_ITEM_COLUMNS} FROM knowledge_items"
    params: list = []
    if statuses:
        sql += " WHERE status IN (" + ", ".join("?" for _ in statuses) + ")"
        params = list(statuses)
    sql += " ORDER BY item_id"
    return [KnowledgeItemRow(*r) for r in con.execute(sql, params).fetchall()]


def next_item_id(con: duckdb.DuckDBPyConnection) -> str:
    """item_id を `item_0001` の形式で採番する。既存の最大番号 + 1 にする。"""
    rows = con.execute("SELECT item_id FROM knowledge_items").fetchall()
    numbers = []
    for (item_id,) in rows:
        if item_id.startswith("item_") and item_id[5:].isdigit():
            numbers.append(int(item_id[5:]))
    return f"item_{(max(numbers) + 1) if numbers else 1:04d}"


def add_evidence(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    request_id: str,
    similarity: float | None,
    linked_at: datetime,
) -> None:
    """項目と根拠発言を紐付ける。同じ組が既にあれば何もしない（照合の再実行を許すため）。"""
    exists = con.execute(
        "SELECT count(*) FROM item_evidence WHERE item_id = ? AND request_id = ?",
        [item_id, request_id],
    ).fetchone()[0]
    if exists:
        return
    con.execute(
        "INSERT INTO item_evidence (item_id, request_id, similarity, linked_at) VALUES (?, ?, ?, ?)",
        [item_id, request_id, similarity, linked_at],
    )


def get_linked_request_ids(con: duckdb.DuckDBPyConnection, request_ids: list[str]) -> set[str]:
    """request_ids のうち、すでに item_evidence に紐付いているものを返す。"""
    if not request_ids:
        return set()
    placeholders = ", ".join("?" for _ in request_ids)
    rows = con.execute(
        f"SELECT DISTINCT request_id FROM item_evidence WHERE request_id IN ({placeholders})",
        request_ids,
    ).fetchall()
    return {r[0] for r in rows}


# --- embeddings --------------------------------------------------------------


def text_hash(text: str) -> str:
    """ベクトル化した元テキストのハッシュ（NFKC で正規化した文字列の SHA-256、16進）。

    embeddings.text_hash に保存し、テキストが編集されたかどうかの判定に使う。
    NFKC で正規化する理由は、全角と半角の違いだけでベクトルを作り直さないため。
    ハッシュの計算はこの関数に一本化する（get_embeddings、put_embeddings、match/link.py が使う）。
    """
    return hashlib.sha256(unicodedata.normalize("NFKC", text).encode("utf-8")).hexdigest()


def get_embeddings(
    con: duckdb.DuckDBPyConnection, owner_type: str, texts: dict[str, str], model_name: str
) -> dict[str, np.ndarray]:
    """保存済みのベクトルを owner_id をキーにして返す。

    texts は {owner_id: 現在のテキスト}。次の行だけを返す。
    - model_name が一致する。
    - text_hash が、現在のテキストから計算したハッシュと一致する（テキストが編集されていない）。
    一致しない行は返さない。呼び出し側が再計算して put_embeddings で上書きする。
    """
    if not texts:
        return {}
    owner_ids = list(texts)
    placeholders = ", ".join("?" for _ in owner_ids)
    rows = con.execute(
        "SELECT owner_id, text_hash, vector FROM embeddings "
        f"WHERE owner_type = ? AND model_name = ? AND owner_id IN ({placeholders})",
        [owner_type, model_name, *owner_ids],
    ).fetchall()
    return {
        owner_id: np.asarray(vec, dtype=np.float32)
        for owner_id, stored_hash, vec in rows
        if stored_hash == text_hash(texts[owner_id])
    }


def put_embeddings(
    con: duckdb.DuckDBPyConnection,
    owner_type: str,
    model_name: str,
    entries: dict[str, tuple[str, np.ndarray]],
    created_at: datetime,
) -> None:
    """ベクトルを保存する。

    entries は {owner_id: (ベクトル化した元テキスト, ベクトル)}。text_hash も書き込む。
    同じ (owner_type, owner_id, model_name) の行があれば入れ替える。
    """
    for owner_id, (text, vec) in entries.items():
        arr = np.asarray(vec, dtype=np.float32)
        con.execute(
            "DELETE FROM embeddings WHERE owner_type = ? AND owner_id = ? AND model_name = ?",
            [owner_type, owner_id, model_name],
        )
        con.execute(
            "INSERT INTO embeddings (owner_type, owner_id, model_name, text_hash, dim, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [owner_type, owner_id, model_name, text_hash(text), int(arr.shape[0]), arr.tolist(), created_at],
        )
