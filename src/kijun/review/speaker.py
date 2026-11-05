"""話者の対応付けの返信の解釈と、speaker_map への反映(仕様 2-7、CLAUDE.md 4-6)。

会議ごとに各話者ラベルの発言サンプルを投稿し、ユーザーが「00 は自分、01 は後輩 A」のように返信する。
返信は2通りの形で受ける。

1. 構造化した形式(`00=self 01=kouhai_a`)。LLM を使わず正規表現で解釈する。確実で、費用も時間もかからない。
2. 自然な文章。1 で解釈できなかった場合だけ LLM に渡す。自然文から英数字の person_label への変換は LLM が行う。

`self`(ユーザー本人)だけは特別扱いする。direction の導出がこれに依存するためで、self が1つも決まらない会議は
その旨を出し、direction を導出しない。

推測しない(原則2-4): 会議に存在しない話者ラベルへの対応付けは反映しない。1つの返信の中で同じ話者に
別々の人物が書かれていたら、解釈できない返信として扱う。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Protocol

import duckdb
from pydantic import BaseModel, Field

from kijun.db import repo
from kijun.discord.base import Reply
from kijun.review.llm import JsonChat, ReviewLlmError

# `00=self`、`SPEAKER_01 = kouhai_a` のような1組
_PAIR = re.compile(r"(?:SPEAKER_)?(\d{2})\s*=\s*([A-Za-z0-9_]+)", re.IGNORECASE)
# 組と組の間に置いてよい文字(空白、読点、カンマ、セミコロン)
_SEPARATORS = re.compile(r"[\s,、，;；]*")
_PERSON_LABEL = re.compile(r"[a-z0-9_]+")


class AmbiguousSpeakerReply(ValueError):
    """返信を解釈できない(同じ話者に別々の人物が書かれている等)。推測せず、未処理のまま残す。"""


def normalize_speaker_label(raw: str) -> str:
    """`00`、`SPEAKER_00`、`speaker_00` を `SPEAKER_00` にそろえる。"""
    digits = raw.upper().removeprefix("SPEAKER_")
    return f"SPEAKER_{digits}"


def parse_structured(text: str) -> dict[str, str] | None:
    """`00=self 01=kouhai_a` の形式なら {話者ラベル: 人物ラベル} を返す。この形式でなければ None。

    返信の全体がこの形式の組と区切りだけでできているときだけ、構造化した形式とみなす。
    文章が混じっているときは None を返し、LLM に渡す側に任せる。
    同じ話者に別々の人物が書かれていたら AmbiguousSpeakerReply を出す。
    """
    pairs = list(_PAIR.finditer(text))
    if not pairs:
        return None
    remainder = _PAIR.sub(" ", text)
    if _SEPARATORS.fullmatch(remainder) is None:
        return None
    mapping: dict[str, str] = {}
    for m in pairs:
        label, person = normalize_speaker_label(m.group(1)), m.group(2).lower()
        if mapping.get(label, person) != person:
            raise AmbiguousSpeakerReply(
                f"{label} に {mapping[label]} と {person} の両方が書かれています"
            )
        mapping[label] = person
    return mapping


class SpeakerAssignment(BaseModel):
    speaker: str = Field(description="話者の番号またはラベル。例: 00、SPEAKER_01。")
    person: str = Field(
        description="人物を表す英小文字・数字・アンダースコアだけの識別子。ユーザー本人は self。例: self、kouhai_a。"
    )


class SpeakerAssignments(BaseModel):
    items: list[SpeakerAssignment]


class SpeakerReplyInterpreter(Protocol):
    def interpret(self, text: str, labels: list[str]) -> dict[str, str]:
        """自然な文の返信を {話者ラベル: 人物ラベル} にする。失敗したら ReviewLlmError を出す。"""
        ...


SYSTEM_PROMPT = """\
あなたは、会議の話者ラベルと人物の対応を書いたユーザーの返信を、機械が読める形にする係です。
話者ラベルは SPEAKER_00 のような機械的な名前で、ユーザーが「00」のように番号だけで書くこともあります。

# 規則
- ユーザー本人(「自分」「私」「僕」「俺」「わたし」など)は、person を self にしてください。
- それ以外の人物は、英小文字・数字・アンダースコアだけの識別子にしてください(例: 後輩A は kouhai_a、友人B は friend_b)。
- 返信に書かれている対応だけを出力してください。書かれていない話者や人物を推測して足してはいけません。
- 対応が1つも読み取れない返信は、items を空の配列にしてください。

# 出力形式
JSON だけを返してください。前置き、説明、コードフェンス(```)は付けません。
形は {"items": [{"speaker": "00", "person": "self"}, ...]} のオブジェクトです(トップレベルを配列にしません)。
"""


class LlmSpeakerInterpreter:
    def __init__(self, chat: JsonChat) -> None:
        self._chat = chat

    def interpret(self, text: str, labels: list[str]) -> dict[str, str]:
        user = f"この会議の話者ラベル: {', '.join(labels)}\n\nユーザーの返信:\n{text}"
        result = self._chat.ask(SYSTEM_PROMPT, user, SpeakerAssignments, "speaker_assignment")
        return {normalize_speaker_label(a.speaker): a.person.strip().lower() for a in result.items}


@dataclass
class SpeakerOutcome:
    """話者の対応付けの返信を反映した結果。"""

    changed: bool = False  # speaker_map の行が増えた・変わった
    has_self: bool = False  # 反映後の speaker_map に self がある
    failed: bool = False  # 解釈できなかった返信がある(未処理のまま残す)
    messages: list[str] = field(default_factory=list)


def apply_speaker_replies(
    con: duckdb.DuckDBPyConnection,
    conversation_id: str,
    labels: list[str],
    replies: list[Reply],
    interpreter: SpeakerReplyInterpreter | None,
) -> SpeakerOutcome:
    """返信を古い順に解釈して speaker_map に反映する。

    構造化した形式は LLM を使わずに解釈する。そうでなければ interpreter に渡す
    (interpreter が None、または失敗したときは、その返信を解釈できなかったものとして failed にする)。
    会議に存在しない話者ラベルへの対応付けは反映せず、messages に残す。
    """
    outcome = SpeakerOutcome()
    for reply in sorted(replies, key=lambda r: r.created_at):
        if not reply.author_is_owner or not reply.content.strip():
            continue
        try:
            mapping = parse_structured(reply.content)
            if mapping is None:
                if interpreter is None:
                    raise ReviewLlmError("LLM を使えないため、自然な文の返信を解釈できません")
                mapping = interpreter.interpret(reply.content, labels)
        except (AmbiguousSpeakerReply, ReviewLlmError) as e:
            outcome.failed = True
            outcome.messages.append(f"会議 {conversation_id}: 話者の対応付けの返信を解釈できませんでした: {e}")
            continue
        if not mapping:
            outcome.messages.append(f"会議 {conversation_id}: 返信から対応を読み取れませんでした: {reply.content[:40]}")
            outcome.failed = True
            continue
        existing = repo.get_speaker_map(con, conversation_id)
        for label, person in mapping.items():
            if label not in labels:
                outcome.messages.append(
                    f"会議 {conversation_id}: 話者ラベル {label} はこの会議にありません(反映しません)"
                )
                continue
            if _PERSON_LABEL.fullmatch(person) is None:
                outcome.messages.append(
                    f"会議 {conversation_id}: 人物ラベル '{person}' は英小文字・数字・アンダースコアではありません(反映しません)"
                )
                continue
            if existing.get(label) != person:
                repo.set_speaker_map(con, conversation_id, label, person)
                outcome.changed = True
    final = repo.get_speaker_map(con, conversation_id)
    outcome.has_self = repo.PERSON_SELF in final.values()
    if final and not outcome.has_self:
        outcome.messages.append(
            f"会議 {conversation_id}: self が決まっていません。direction を導出できません。"
            "返信に `00=self` のように自分を書いてください"
        )
    return outcome
