"""Discord のレビューのテストが共有する、DB の準備と部品。"""

from __future__ import annotations

from datetime import datetime

from kijun.db import repo
from kijun.discord.base import Reaction, Reply
from tests.conftest import make_conversation, make_utterances

CID = "20260920_1930_f2f_teirei"
NOW = datetime(2026, 11, 4, 1, 0)
APPROVE, REJECT, HOLD = "✅", "❌", "⏸️"  # Discord は ⏸ を異体字セレクタ付きで返すことがある
ONE, TWO, THREE = "1️⃣", "2️⃣", "3️⃣"


def r(*emojis: str) -> list[Reaction]:
    return [Reaction(e, 1) for e in emojis]


def reply(text: str, minute: int = 0, owner: bool = True) -> Reply:
    return Reply(author_is_owner=owner, content=text, created_at=datetime(2026, 11, 3, 22, minute))


def add_request(
    con, request_id: str, conversation_id: str = CID, quote: str = "引用", summary: str = "要約",
    from_label: str | None = "SPEAKER_00", to_label: str | None = "SPEAKER_01", model: str = "m",
    direction: str | None = None, **kw,
) -> repo.RequestRow:
    row = repo.RequestRow(
        request_id=request_id, conversation_id=conversation_id, seq_from=1, seq_to=1,
        speaker_from_label=from_label, speaker_to_label=to_label, direction=direction,
        counterpart_role=None, artifact_type=kw.get("artifact_type", "announcement"), quote=quote,
        request_summary=summary, stated_reason=kw.get("stated_reason"), applies_when=kw.get("applies_when"),
        reusable=True, pass_criterion=kw.get("pass_criterion"), source_type="live_conversation",
        model_name=model, extracted_at=datetime(2026, 11, 3),
    )
    repo.insert_requests(con, [row])
    return row


def add_item(
    con, item_id: str = "item_0001", check_text: str = "冒頭2行以内に行動が書かれているか",
    status: str = "candidate", importance: int | None = None, artifact_type: str | None = "announcement",
    pass_criterion: str | None = None, request_id: str | None = None, linked_at: datetime | None = None,
    **kw,
) -> repo.KnowledgeItemRow:
    item = repo.KnowledgeItemRow(
        item_id=item_id, artifact_type=artifact_type, check_text=check_text, pass_criterion=pass_criterion,
        reason=kw.get("reason"), applies_when=kw.get("applies_when"), not_applies_when=kw.get("not_applies_when"),
        importance=importance, status=status, version=1, updated_at=datetime(2026, 11, 3),
    )
    repo.insert_knowledge_item(con, item)
    if request_id:
        repo.add_evidence(con, item_id, request_id, None, linked_at or datetime(2026, 11, 3, 12, 0))
    return item


def setup_conversation(con, conversation_id: str = CID, utterance_count: int = 6) -> None:
    repo.insert_conversation(con, make_conversation(conversation_id))
    repo.replace_utterances(con, conversation_id, make_utterances(utterance_count))


def candidate(con, n: int = 1, importance: int | None = None, linked_at: datetime | None = None, **kw):
    """会議・抽出結果・新規候補を1組作る(item_000n)。"""
    if repo.get_conversation(con, CID) is None:
        setup_conversation(con)
    item_id = f"item_{n:04d}"
    add_request(con, f"{CID}#m#{n}", quote=f"発言{n}", summary=f"要約{n}")
    return add_item(
        con, item_id, check_text=kw.pop("check_text", f"チェック項目{n}"), importance=importance,
        request_id=f"{CID}#m#{n}", linked_at=linked_at, **kw,
    )
