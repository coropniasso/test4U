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

# extracted_requests.direction の保存値
DIRECTION_SELF_TO_OTHER = "self_to_other"
DIRECTION_OTHER_TO_SELF = "other_to_self"

# speaker_map.person_label のうち、ユーザー本人を表す値。direction の導出がこれに依存する。
PERSON_SELF = "self"

# discord_posts.post_type の保存値
POST_NEW_CANDIDATE = "new_candidate"
POST_CONTRADICTION = "contradiction"
POST_SPEAKER_SAMPLE = "speaker_sample"
POST_BUNDLE_RESULT = "bundle_result"
# 文字起こしが壊れていて判定できなかった抽出結果。reusable = FALSE なので照合の対象から外れ、
# knowledge_items にならない。利用者が壊れた引用を見て直せるように、別の種別で投稿する。
POST_TRANSCRIPTION_UNCLEAR = "transcription_unclear"

# review_log.action の保存値。値はここ1か所で定義し、他のファイルでは文字列を直接書かない。
# 後で上書き履歴を回帰の目的変数にする（CLAUDE.md 4-6）ため、値を変えると過去の履歴が読めなくなる。
ACTION_STATUS_CHANGED = "status_changed"  # before = 変更前の status、after = 変更後の status
ACTION_IMPORTANCE_PROPOSED = "importance_proposed"  # after = LLM が添えた根拠の1文
ACTION_IMPORTANCE_OVERRIDDEN = "importance_overridden"  # before = LLM の判定値、after = ユーザーの値
ACTION_REPLY_RECEIVED = "reply_received"  # before = 返信の送信時刻（ISO）、after = 返信の原文の全文
ACTION_REPLY_ROUTED = "reply_routed"  # before = 対応する返信の送信時刻（ISO）、after = 振り分け結果の JSON
ACTION_AMBIGUOUS_REACTION = "ambiguous_reaction"  # after = 判定できなかったリアクションの説明
ACTION_CHECK_TEXT_EDITED = "check_text_edited"  # before / after = check_text の変更前後

REVIEW_LOG_ACTIONS = (
    ACTION_STATUS_CHANGED,
    ACTION_IMPORTANCE_PROPOSED,
    ACTION_IMPORTANCE_OVERRIDDEN,
    ACTION_REPLY_RECEIVED,
    ACTION_REPLY_ROUTED,
    ACTION_AMBIGUOUS_REACTION,
    ACTION_CHECK_TEXT_EDITED,
)


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
    # 末尾に置いてあるのは、既定値を持たせるためである（データクラスは既定値のある項目を
    # 既定値の無い項目より後ろに置かなければならない）。この並びは _REQUEST_COLUMNS の並びと
    # 一致していなければならない。SELECT の結果から RequestRow(*r) で位置で組み立てるためである。
    # 既定が False なのは「文字起こしは読めた」が通常の場合であり、DB の列の既定値と同じである。
    transcription_unclear: bool = False
    # 原文と照合して一致しなかったときの、モデルが返した元の値（extract/verbatim.py）。
    # stated_reason / applies_when はその場合 None になる。
    stated_reason_unverified: str | None = None
    applies_when_unverified: str | None = None


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


@dataclass(frozen=True)
class DiscordPostRow:
    """discord_posts テーブルの1行。"""

    message_id: str
    post_type: str
    item_id: str | None
    conversation_id: str | None
    posted_at: datetime
    processed_at: datetime | None
    channel_id: str | None
    pair_item_id: str | None
    # transcription_unclear のとき、投稿した extracted_requests.request_id。
    # 既定値を持たせてあるのは、他の種別では使わないためである。並びは _POST_COLUMNS と
    # 一致していなければならない（SELECT の結果から DiscordPostRow(*r) で組み立てる）。
    request_id: str | None = None


@dataclass(frozen=True)
class ReviewLogRow:
    """review_log テーブルの1行。"""

    log_id: int
    item_id: str
    action: str
    before: str | None
    after: str | None
    reviewed_at: datetime


_REQUEST_COLUMNS = (
    "request_id, conversation_id, seq_from, seq_to, speaker_from_label, speaker_to_label, "
    "direction, counterpart_role, artifact_type, quote, request_summary, stated_reason, "
    "applies_when, reusable, pass_criterion, source_type, model_name, extracted_at, "
    "transcription_unclear, stated_reason_unverified, applies_when_unverified"
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
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                r.transcription_unclear,
                r.stated_reason_unverified,
                r.applies_when_unverified,
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


def get_transcription_unclear_requests(
    con: duckdb.DuckDBPyConnection, model_name: str
) -> list[RequestRow]:
    """文字起こしが読めず、要求の中心が決まらなかった抽出結果（会議の古い順）。

    これらは reusable = FALSE なので照合（match/link.py）の対象から外れ、knowledge_items に
    ならない。だが発話したのは利用者本人なので、壊れた引用を見れば自分が何と言ったか分かる。
    その機会を渡すために、Discord に別の種別で投稿する（CLAUDE.md 7章の案A）。
    """
    sql = (
        f"SELECT {_REQUEST_COLUMNS} FROM extracted_requests r "
        "WHERE r.transcription_unclear AND r.model_name = ? "
        "ORDER BY (SELECT c.recorded_at FROM conversations c "
        "          WHERE c.conversation_id = r.conversation_id), r.seq_from, r.request_id"
    )
    return [RequestRow(*r) for r in con.execute(sql, [model_name]).fetchall()]


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


# --- knowledge_items の更新 ---------------------------------------------------


def get_knowledge_item(con: duckdb.DuckDBPyConnection, item_id: str) -> KnowledgeItemRow | None:
    row = con.execute(
        f"SELECT {_ITEM_COLUMNS} FROM knowledge_items WHERE item_id = ?", [item_id]
    ).fetchone()
    return KnowledgeItemRow(*row) if row else None


def update_item_status(
    con: duckdb.DuckDBPyConnection, item_id: str, status: str, at: datetime
) -> None:
    con.execute(
        "UPDATE knowledge_items SET status = ?, updated_at = ? WHERE item_id = ?",
        [status, at, item_id],
    )


def update_item_importance(
    con: duckdb.DuckDBPyConnection, item_id: str, importance: int | None, at: datetime
) -> None:
    con.execute(
        "UPDATE knowledge_items SET importance = ?, updated_at = ? WHERE item_id = ?",
        [importance, at, item_id],
    )


# update_item_fields が書き換えてよい列。SQL に列名を埋め込むので、許可する名前をここで固定する。
EDITABLE_ITEM_FIELDS = ("check_text", "reason", "applies_when", "not_applies_when")


def update_item_fields(
    con: duckdb.DuckDBPyConnection, item_id: str, fields: dict[str, str], at: datetime
) -> None:
    """ユーザーが書き足した欄を更新し、version を1つ上げる。fields の空は何もしない。"""
    unknown = set(fields) - set(EDITABLE_ITEM_FIELDS)
    if unknown:
        raise ValueError(f"更新できない列です: {sorted(unknown)}")
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    con.execute(
        f"UPDATE knowledge_items SET {assignments}, version = version + 1, updated_at = ? "
        "WHERE item_id = ?",
        [*fields.values(), at, item_id],
    )


def get_item_evidence_requests(
    con: duckdb.DuckDBPyConnection, item_id: str, limit: int | None = None
) -> list[RequestRow]:
    """項目の根拠になった抽出結果を、紐付けた順（古いものから）に返す。"""
    sql = (
        f"SELECT {', '.join('r.' + c.strip() for c in _REQUEST_COLUMNS.split(','))} "
        "FROM item_evidence e JOIN extracted_requests r USING (request_id) "
        "WHERE e.item_id = ? ORDER BY e.linked_at, r.request_id"
    )
    params: list = [item_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [RequestRow(*r) for r in con.execute(sql, params).fetchall()]


# --- review_log --------------------------------------------------------------


def insert_review_log(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
    action: str,
    before: str | None,
    after: str | None,
    at: datetime,
) -> int:
    """review_log に1行足し、log_id を返す。action は REVIEW_LOG_ACTIONS のどれか。"""
    if action not in REVIEW_LOG_ACTIONS:
        raise ValueError(f"未定義の review_log.action です: {action}")
    log_id = con.execute("SELECT nextval('review_log_seq')").fetchone()[0]
    con.execute(
        "INSERT INTO review_log (log_id, item_id, action, before, after, reviewed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [log_id, item_id, action, before, after, at],
    )
    return log_id


def list_review_log(
    con: duckdb.DuckDBPyConnection, item_id: str | None = None, action: str | None = None
) -> list[ReviewLogRow]:
    """review_log を log_id の昇順で返す。item_id と action で絞り込める。"""
    sql = "SELECT log_id, item_id, action, before, after, reviewed_at FROM review_log"
    where, params = [], []
    if item_id is not None:
        where.append("item_id = ?")
        params.append(item_id)
    if action is not None:
        where.append("action = ?")
        params.append(action)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY log_id"
    return [ReviewLogRow(*r) for r in con.execute(sql, params).fetchall()]


# --- discord_posts -----------------------------------------------------------

_POST_COLUMNS = (
    "message_id, post_type, item_id, conversation_id, posted_at, processed_at, "
    "channel_id, pair_item_id, request_id"
)


def insert_discord_post(con: duckdb.DuckDBPyConnection, row: DiscordPostRow) -> None:
    con.execute(
        f"INSERT INTO discord_posts ({_POST_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            row.message_id,
            row.post_type,
            row.item_id,
            row.conversation_id,
            row.posted_at,
            row.processed_at,
            row.channel_id,
            row.pair_item_id,
            row.request_id,
        ],
    )


def list_discord_posts(
    con: duckdb.DuckDBPyConnection,
    post_type: str | None = None,
    unprocessed_only: bool = False,
) -> list[DiscordPostRow]:
    """投稿の記録を投稿時刻の昇順で返す。"""
    sql = f"SELECT {_POST_COLUMNS} FROM discord_posts"
    where, params = [], []
    if post_type is not None:
        where.append("post_type = ?")
        params.append(post_type)
    if unprocessed_only:
        where.append("processed_at IS NULL")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY posted_at, message_id"
    return [DiscordPostRow(*r) for r in con.execute(sql, params).fetchall()]


def mark_post_processed(con: duckdb.DuckDBPyConnection, message_id: str, at: datetime) -> None:
    con.execute("UPDATE discord_posts SET processed_at = ? WHERE message_id = ?", [at, message_id])


# --- speaker_map と direction ----------------------------------------------------


def get_speaker_map(con: duckdb.DuckDBPyConnection, conversation_id: str) -> dict[str, str]:
    """会議の {話者ラベル: 人物ラベル}。未登録なら空。"""
    rows = con.execute(
        "SELECT speaker_label, person_label FROM speaker_map WHERE conversation_id = ?",
        [conversation_id],
    ).fetchall()
    return {label: person for label, person in rows}


def set_speaker_map(
    con: duckdb.DuckDBPyConnection, conversation_id: str, speaker_label: str, person_label: str
) -> None:
    """話者ラベルと人物の対応を入れ直す（同じラベルがあれば置き換える）。"""
    con.execute(
        "DELETE FROM speaker_map WHERE conversation_id = ? AND speaker_label = ?",
        [conversation_id, speaker_label],
    )
    con.execute(
        "INSERT INTO speaker_map (conversation_id, speaker_label, person_label) VALUES (?, ?, ?)",
        [conversation_id, speaker_label, person_label],
    )


def update_request_direction(
    con: duckdb.DuckDBPyConnection, request_id: str, direction: str | None
) -> None:
    con.execute(
        "UPDATE extracted_requests SET direction = ? WHERE request_id = ?", [direction, request_id]
    )
