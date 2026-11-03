"""スレッド返信の振り分け(仕様 2-6、CLAUDE.md 4-6)。

ユーザーが書いた自由な文章を、LLM が reason / applies_when / not_applies_when / check_text のどれに
入れるかを決める。CLAUDE.md 4-6 は3つ(reason / applies_when / check_text)を挙げているが、
knowledge_items には not_applies_when の列があり(CLAUDE.md 5章)、「どの状況なら同じ指摘をしないか」は
CLAUDE.md 1章が暗黙知の中身として挙げている2点のうちの1つなので、振り分け先に加えた。

返信の原文は必ず review_log に残す(reply_received)。そのうえで振り分けの結果を別の行(reply_routed)に残す。
LLM の振り分けが間違っても、原文から人が直せるようにするためで、振り分けの結果だけを保存してはいけない。
振り分けに失敗したときも、原文の記録は残る。

1つの返信が複数の欄に対応することがある(「冒頭に書かないと読み飛ばされるから。ただし社内向けの連絡では不要」は
reason と not_applies_when の両方)。そのため出力は欄ごとの配列にする。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

import duckdb
from pydantic import BaseModel, Field

from kijun.db import repo
from kijun.db.repo import KnowledgeItemRow
from kijun.discord.base import Reply
from kijun.review.llm import ReviewLlmError, StrataJsonChat

COLUMN_REASON = "reason"
COLUMN_APPLIES_WHEN = "applies_when"
COLUMN_NOT_APPLIES_WHEN = "not_applies_when"
COLUMN_CHECK_TEXT = "check_text"

# 矛盾の投稿(項目 A と B の2つを扱う)で、返信がどちらの項目についてかを表す印
TARGET_A = "A"
TARGET_B = "B"


class RoutedEntry(BaseModel):
    """返信の一部を、項目のどの欄に入れるか。"""

    column: Literal["reason", "applies_when", "not_applies_when", "check_text"] = Field(
        description="入れる欄。reason=理由、applies_when=適用条件、not_applies_when=適用しない条件、check_text=チェック項目の文言の修正。"
    )
    text: str = Field(min_length=1, description="その欄に入れる文。返信に書かれた内容だけ。")
    target: Literal["A", "B"] | None = Field(
        default=None,
        description="項目が A と B の2つあるとき、どちらの項目についてか。1つだけのときは null。",
    )


class RoutedReply(BaseModel):
    """返信1件の振り分け結果。どの欄にも当たらない返信は items が空。"""

    items: list[RoutedEntry]


class ReplyRouter(Protocol):
    def route(
        self, reply_text: str, item: KnowledgeItemRow, other: KnowledgeItemRow | None = None
    ) -> RoutedReply:
        """返信を欄ごとに振り分ける。other は矛盾の投稿のもう一方の項目。失敗したら ReviewLlmError を出す。"""
        ...


SYSTEM_PROMPT = """\
あなたは、ユーザーが書いた返信を、チェック項目の欄に振り分ける係です。
ユーザーは、会話から抜き出されたチェック項目に対して、理由や適用条件を自分の言葉で書き足しています。

# 欄
- reason: その指摘をした理由。
- applies_when: その指摘を適用する状況・条件。
- not_applies_when: 同じ指摘をしない状況・条件(「〜のときは不要」「〜なら言わない」)。
- check_text: チェック項目の文言そのものの修正(「その項目は〇〇と直して」)。

# 規則
- 1つの返信が複数の欄に当たることがあります。当たる欄ごとに1要素を作ってください。
- text には、返信に書かれた内容だけを入れてください。書かれていない理由や条件を推測して足してはいけません。
  この仕組みの目的は、ユーザー自身が理由と適用条件を言葉にすることです。あなたが補うと、その機会が失われます。
- 返信がどの欄にも当たらない(了解、感想、質問など)ときは、items を空の配列にしてください。
- 項目が A と B の2つ渡されたときは、各要素の target に、その内容がどちらの項目についてかを "A" または "B" で書いてください。
  どちらか判断できない内容は、要素にしないでください。項目が1つだけのときは target を null にしてください。

# 出力形式
JSON だけを返してください。前置き、説明、コードフェンス(```)は付けません。
形は {"items": [{"column": "...", "text": "...", "target": null}, ...]} のオブジェクトです(トップレベルを配列にしません)。
"""


def _describe(label: str | None, item: KnowledgeItemRow) -> str:
    head = f"項目{label}" if label else "項目"
    return "\n".join(
        [
            f"{head}(item_id: {item.item_id})",
            f"  check_text: {item.check_text}",
            f"  reason: {item.reason or '(記録なし)'}",
            f"  applies_when: {item.applies_when or '(記録なし)'}",
            f"  not_applies_when: {item.not_applies_when or '(記録なし)'}",
        ]
    )


def build_user_message(
    reply_text: str, item: KnowledgeItemRow, other: KnowledgeItemRow | None
) -> str:
    if other is None:
        items = _describe(None, item)
    else:
        items = _describe(TARGET_A, item) + "\n\n" + _describe(TARGET_B, other)
    return f"{items}\n\nユーザーの返信:\n{reply_text}"


class StrataReplyRouter:
    def __init__(self, chat: StrataJsonChat) -> None:
        self._chat = chat

    def route(
        self, reply_text: str, item: KnowledgeItemRow, other: KnowledgeItemRow | None = None
    ) -> RoutedReply:
        return self._chat.ask(
            SYSTEM_PROMPT, build_user_message(reply_text, item, other), RoutedReply, "reply_routing"
        )


# --- DB への反映 ----------------------------------------------------------------


@dataclass
class ReplyOutcome:
    """返信1件の処理結果。"""

    key: str  # 返信の送信時刻(ISO)。reply_received.before / reply_routed.before と同じ
    already_routed: bool = False  # 前の夜に振り分け済みだった
    routed: bool = False  # 今回振り分けた
    failed: bool = False  # 振り分けに失敗した(原文の記録だけ残り、次回再試行する)
    applied: list[dict] = field(default_factory=list)
    unapplied: list[dict] = field(default_factory=list)


def reply_key(reply: Reply) -> str:
    return reply.created_at.isoformat()


def _has_log(
    con: duckdb.DuckDBPyConnection, item_id: str, action: str, before: str, after: str | None = None
) -> bool:
    for row in repo.list_review_log(con, item_id, action):
        if row.before == before and (after is None or row.after == after):
            return True
    return False


def process_reply(
    con: duckdb.DuckDBPyConnection,
    router: ReplyRouter,
    post_item_id: str,
    reply: Reply,
    at: datetime,
    other_item_id: str | None = None,
) -> ReplyOutcome:
    """返信1件を記録し、振り分けて項目に反映する。

    1. 原文を review_log に reply_received として残す(既にあれば重複して残さない)。
       振り分けより先に行う。LLM が失敗しても原文が残るようにするため。
    2. 既に reply_routed があれば何もしない(前の夜に処理済み)。
    3. LLM に振り分けさせる。失敗したら failed を返し、knowledge_items は変えない。
       processed_at を埋めないので、次回の read で再試行される。
    4. 振り分け結果を knowledge_items に反映し、reply_routed を残す。
       同じ欄の既存の値は置き換える(古い値は reply_routed の JSON に残る)。
       check_text を変えたときは check_text_edited も残す。

    矛盾の投稿(other_item_id がある)では、項目 A = post_item_id、B = other_item_id として、
    各要素の target で反映先を決める。target が無い・不正な要素は反映せず、unapplied に残す
    (返信の原文は残っているので、人が直せる)。
    """
    key = reply_key(reply)
    outcome = ReplyOutcome(key=key)

    if not _has_log(con, post_item_id, repo.ACTION_REPLY_RECEIVED, key, reply.content):
        repo.insert_review_log(con, post_item_id, repo.ACTION_REPLY_RECEIVED, key, reply.content, at)
    if _has_log(con, post_item_id, repo.ACTION_REPLY_ROUTED, key):
        outcome.already_routed = True
        return outcome

    item = repo.get_knowledge_item(con, post_item_id)
    other = repo.get_knowledge_item(con, other_item_id) if other_item_id else None
    if item is None:
        outcome.failed = True
        return outcome
    try:
        routed = router.route(reply.content, item, other)
    except ReviewLlmError as e:
        print(
            f"警告: 項目 {post_item_id} への返信を振り分けられませんでした(原文は review_log に残してあります。"
            f"次回の read で再試行します): {e}",
            file=sys.stderr,
        )
        outcome.failed = True
        return outcome

    # 反映先の項目ごとに、欄の新しい値をまとめる(同じ欄が複数あれば後のものが勝つ)
    targets: dict[str, dict[str, str]] = {}
    for entry in routed.items:
        if other is None:
            target_id = post_item_id
        elif entry.target == TARGET_A:
            target_id = post_item_id
        elif entry.target == TARGET_B:
            target_id = other.item_id
        else:
            outcome.unapplied.append(
                {"column": entry.column, "text": entry.text, "reason": "どちらの項目か特定できない"}
            )
            continue
        targets.setdefault(target_id, {})[entry.column] = entry.text

    for target_id, fields in targets.items():
        current = repo.get_knowledge_item(con, target_id)
        assert current is not None
        changes = {c: t for c, t in fields.items() if getattr(current, c) != t}
        for column, text in fields.items():
            outcome.applied.append(
                {"item_id": target_id, "column": column, "old": getattr(current, column), "new": text}
            )
        if changes:
            repo.update_item_fields(con, target_id, changes, at)
        if COLUMN_CHECK_TEXT in changes:
            repo.insert_review_log(
                con, target_id, repo.ACTION_CHECK_TEXT_EDITED, current.check_text, changes[COLUMN_CHECK_TEXT], at
            )

    summary = json.dumps({"applied": outcome.applied, "unapplied": outcome.unapplied}, ensure_ascii=False)
    repo.insert_review_log(con, post_item_id, repo.ACTION_REPLY_ROUTED, key, summary, at)
    if other is not None and any(a["item_id"] == other.item_id for a in outcome.applied):
        repo.insert_review_log(con, other.item_id, repo.ACTION_REPLY_ROUTED, key, summary, at)
    outcome.routed = True
    return outcome
