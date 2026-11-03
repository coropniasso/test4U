"""Discord に投稿する文面を作る(仕様 2-3)。

純粋な文字列処理で、外部に接続しない。投稿の形は CLAUDE.md 4-6 の「メッセージの例」に合わせる。

守る点:
- 語られなかった理由・適用条件は、空欄にせず「記録なし」と書く(CLAUDE.md 1章)。
  語られなかったことが見えている必要があり、LLM が推測で埋めていないことの確認にもなる。
- 話者と人物の対応が未確定なら、話者ラベル(SPEAKER_00)のまま出す。推測しない。
- Discord の1メッセージは 2,000 文字まで。超えるときは quote だけを末尾から切り詰める。
  チェック項目案と理由・適用条件はレビューの対象なので切り詰めない。
"""

from __future__ import annotations

import re

from kijun.db.repo import KnowledgeItemRow, RequestRow
from kijun.models import Conversation

# Discord の1メッセージの文字数の上限(Discord の仕様)
DISCORD_MESSAGE_LIMIT = 2000

NO_RECORD = "記録なし"
ELLIPSIS = "…"
SELF_DISPLAY = "あなた"

# setting_kind の日本語表示。未知の種類は英数字のまま出す。
SETTING_KIND_JA = {"teirei": "定例", "sagyou": "作業", "zatsudan": "雑談"}

REVIEW_GUIDE = "→ ✅❌⏸でstatus、1️⃣2️⃣3️⃣で重要度。理由・条件・修正はスレッドに返信"


def item_number(item_id: str) -> str:
    """item_0142 -> 0142。この形でなければ item_id をそのまま返す。"""
    m = re.fullmatch(r"item_(\d+)", item_id)
    return m.group(1) if m else item_id


def speaker_display(label: str | None, speaker_map: dict[str, str]) -> str:
    """話者ラベルの表示名。self は「あなた」、対応済みなら人物ラベル、未確定ならラベルのまま。"""
    if label is None:
        return "不明"
    person = speaker_map.get(label)
    if person is None:
        return label
    return SELF_DISPLAY if person == "self" else person


def scene(conversation: Conversation) -> str:
    kind = SETTING_KIND_JA.get(conversation.setting_kind, conversation.setting_kind)
    return f"{conversation.recorded_at.month}/{conversation.recorded_at.day} {kind}"


def _or_no_record(value: str | None) -> str:
    return value if value else NO_RECORD


def format_new_candidate(
    item: KnowledgeItemRow,
    request: RequestRow | None,
    conversation: Conversation | None,
    speaker_map: dict[str, str],
    limit: int = DISCORD_MESSAGE_LIMIT,
) -> str:
    """新規候補の文面。limit を超えるときは quote だけを切り詰める。

    quote を切り詰めても limit に収まらない場合は、収まらないまま返す(呼び出し側が検出して投稿しない)。
    チェック項目案などを黙って削ると、レビューの対象が見えなくなるため。
    """
    importance = f"重要度案：{item.importance}" if item.importance else "重要度案：なし"
    head = f"【新規候補 #{item_number(item.item_id)}】{importance}"
    scene_line = f"場面：{scene(conversation)}" if conversation else "場面：不明"
    tail = "\n".join(
        [
            f"チェック項目案：{item.check_text}",
            f"理由：{_or_no_record(item.reason)}",
            f"適用条件：{_or_no_record(item.applies_when)}",
            REVIEW_GUIDE,
        ]
    )

    def build(quote_line: str) -> str:
        return "\n".join([head, scene_line, quote_line, tail])

    if request is None:
        return build("発言：記録なし")
    who = (
        f"（{speaker_display(request.speaker_from_label, speaker_map)}"
        f"→{speaker_display(request.speaker_to_label, speaker_map)}）"
    )

    def quote_line(quote: str) -> str:
        return f"発言：「{quote}」{who}"

    text = build(quote_line(request.quote))
    if len(text) <= limit:
        return text
    overflow = len(text) - limit
    keep = len(request.quote) - overflow - len(ELLIPSIS)
    # 切り詰めの結果が空になるときも、省略されたことが分かるように印だけは残す
    shortened = (request.quote[:keep] if keep > 0 else "") + ELLIPSIS
    return build(quote_line(shortened))


def format_contradiction(
    a: KnowledgeItemRow, b: KnowledgeItemRow, similarity: float
) -> str:
    """矛盾の可能性がある組の文面。項目 A・B の中身は切り詰めない。"""

    def block(label: str, item: KnowledgeItemRow) -> str:
        return "\n".join(
            [
                f"{label} #{item_number(item.item_id)}(status: {item.status})",
                f"  チェック項目：{item.check_text}",
                f"  基準：{_or_no_record(item.pass_criterion)}",
                f"  理由：{_or_no_record(item.reason)}",
                f"  適用条件：{_or_no_record(item.applies_when)}",
                f"  適用しない条件：{_or_no_record(item.not_applies_when)}",
            ]
        )

    return "\n".join(
        [
            f"【矛盾の可能性】#{item_number(a.item_id)} と #{item_number(b.item_id)}"
            f"(類似度 {similarity:.2f}、成果物：{a.artifact_type or NO_RECORD})",
            "同じ話題で基準が食い違っています。多くは矛盾ではなく適用条件の違いです。",
            block("A", a),
            block("B", b),
            "→ どの状況ならどちらを使うかを、スレッドに返信してください。"
            "A か B のどちらについてか分かるように書いてください。",
            "  例：「Aは社外向けの告知のとき。Bは社内連絡のとき」",
        ]
    )


def pick_speaker_samples(
    utterances: list[tuple[str | None, str]], count: int
) -> dict[str, list[str]]:
    """話者ラベルごとに、文字数の多い発話を上位 count 件選ぶ。

    「はい」「そうですね」のような短い発話では誰か分からないため、文字数で選ぶ(仕様 2-7)。
    文字数が同じなら、先に話された方を優先する。話者ラベルが無い発話は対象にしない。
    """
    by_speaker: dict[str, list[tuple[int, str]]] = {}
    for index, (label, text) in enumerate(utterances):
        if label is None:
            continue
        by_speaker.setdefault(label, []).append((index, text))
    return {
        label: [text for _, text in sorted(entries, key=lambda e: (-len(e[1]), e[0]))[:count]]
        for label, entries in sorted(by_speaker.items())
    }


def format_speaker_sample(
    conversation_id: str, samples: dict[str, list[str]], limit: int = DISCORD_MESSAGE_LIMIT
) -> str:
    """話者の対応付けの依頼の文面。1件あたりの文字数は、全体が limit に収まるように決める。"""
    footer = [
        "→ このメッセージに返信してください。",
        "   書き方(確実)：00=self 01=kouhai_a",
        "   自然な文でも読み取ります：「00は自分、01は後輩A」",
        "   自分を必ず self と書いてください(自分→相手の判定に使います)",
    ]
    head = f"【話者の対応付け】{conversation_id}"

    def build(max_chars: int) -> str:
        lines = [head]
        for label, texts in samples.items():
            lines.append(f"{label} の発言サンプル：")
            for text in texts:
                shown = text.replace("\n", " ")
                if len(shown) > max_chars:
                    shown = shown[: max(max_chars - len(ELLIPSIS), 0)] + ELLIPSIS
                lines.append(f"  「{shown}」")
        return "\n".join([*lines, *footer])

    total = sum(len(t) for texts in samples.values() for t in texts)
    max_chars = max(total, 1)
    text = build(max_chars)
    # 全体が limit に収まるまで、1件あたりの上限を縮める
    while len(text) > limit and max_chars > len(ELLIPSIS) + 1:
        max_chars = max(max_chars * 3 // 4, len(ELLIPSIS) + 1)
        text = build(max_chars)
    return text


def format_bundle_result(conversation: Conversation) -> str:
    files = "\n".join(f"  - {name}" for name in conversation.source_files)
    minutes = (
        f"、合計 {conversation.duration_sec / 60:.1f} 分" if conversation.duration_sec else ""
    )
    return "\n".join(
        [
            f"【束ね結果】{conversation.conversation_id}",
            f"次の {len(conversation.source_files)} 個のファイルを1つの会議として束ねました{minutes}。",
            files,
            "→ 束ね方に誤りがあれば ❌ を付けてください。自動では分割し直しません。",
            "   config.toml の bundle.gap_minutes を調整して `kijun ingest` をやり直してください。",
        ]
    )
