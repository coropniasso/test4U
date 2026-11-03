"""リアクションと返信の読み取り(仕様 2-4、1-5、CLAUDE.md 4-8 の手順1)。

前回までに投稿したメッセージのうち processed_at が NULL のものを読み、DuckDB に反映する。
bot は常駐しないので、PC が落ちていた間のリアクションと返信も、この読み取りでまとめて取り込む。

判定できないリアクションは推測で決めない(仕様 1-5、原則2-4):
- status のリアクション(✅❌⏸)が2つ以上付いていたら、片方を優先せず、knowledge_items を変えない。
  review_log に ambiguous_reaction を残し、processed_at を埋めず、会議IDと項目IDを報告する。
  次回の read で再び対象になる。誤って承認すると、誤ったチェック項目が「承認済み」として残り、
  間違いであることが後から分かりにくいため。
- 重要度のリアクション(1️⃣2️⃣3️⃣)が2つ以上付いていたら、importance を変えない。
- どのリアクションも無ければ何もしない。processed_at を埋めない(後から付ける可能性があるため)。

processed_at を埋める条件(投稿の種類ごと):
- new_candidate: status が1つに決まり、かつ返信の振り分けがすべて済んだとき。
  status が決まるまでは何度でも読み直す(重要度の上書きと返信の記録は冪等にしてある)。
- contradiction: 返信が1件以上あり、すべて振り分けが済んだとき。リアクションでは何も変えない。
- speaker_sample: 反映後の speaker_map に self があり、解釈できない返信が無いとき。
- bundle_result: ❌ または ✅ が付いたとき。❌ なら束ね直しが必要であることを報告する。
  自動では分割し直さない。会議の束ね方を機械が勝手に変えると、どの音声がどの会議に属するかが分からなくなる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.db.repo import DiscordPostRow
from kijun.discord.base import (
    EMOJI_APPROVE,
    EMOJI_HOLD,
    EMOJI_IMPORTANCE,
    EMOJI_REJECT,
    DiscordClient,
    Reaction,
    Reply,
    normalize_emoji,
)
from kijun.review.direction import derive_directions
from kijun.review.reply import ReplyRouter, process_reply
from kijun.review.speaker import SpeakerReplyInterpreter, apply_speaker_replies

STATUS_BY_EMOJI = {
    EMOJI_APPROVE: repo.STATUS_APPROVED,
    EMOJI_REJECT: repo.STATUS_REJECTED,
    EMOJI_HOLD: repo.STATUS_HELD,
}


@dataclass
class ReadReport:
    processed_posts: int = 0
    status_changes: int = 0
    importance_overrides: int = 0
    replies_routed: int = 0
    replies_failed: int = 0
    ambiguous: list[str] = field(default_factory=list)  # 判定できなかったリアクション(会議ID・項目ID付き)
    rebundle_needed: list[str] = field(default_factory=list)  # 束ね直しが必要な会議ID
    messages: list[str] = field(default_factory=list)
    derived_conversations: list[str] = field(default_factory=list)
    skipped_for_config: bool = False


def count_unprocessed(con: duckdb.DuckDBPyConnection) -> int:
    """読み取り待ちの投稿の件数。夜間バッチが、返信の振り分け(LLM)のために Strata を起動するかの判断に使う。"""
    return len(repo.list_discord_posts(con, unprocessed_only=True))


def _present(reactions: list[Reaction], table: dict) -> list[str]:
    """table に載っている絵文字のうち、1人以上が付けているものを、付いた順に重複なしで返す。"""
    found: list[str] = []
    for r in reactions:
        emoji = normalize_emoji(r.emoji)
        if emoji in table and r.count > 0 and emoji not in found:
            found.append(emoji)
    return found


def _owner_replies(replies: list[Reply]) -> list[Reply]:
    """ユーザー本人の返信だけ。他のメンバーや bot 自身の返信は取り込まない。"""
    return sorted(
        (r for r in replies if r.author_is_owner and r.content.strip()), key=lambda r: r.created_at
    )


def _conversation_of_item(con: duckdb.DuckDBPyConnection, item_id: str) -> str:
    evidence = repo.get_item_evidence_requests(con, item_id, 1)
    return evidence[0].conversation_id if evidence else "-"


def _log_ambiguous_once(
    con: duckdb.DuckDBPyConnection, item_id: str, description: str, at: datetime
) -> None:
    """判定できなかったリアクションを review_log に残す。直前の記録と同じなら重複して残さない。"""
    previous = repo.list_review_log(con, item_id, repo.ACTION_AMBIGUOUS_REACTION)
    if previous and previous[-1].after == description:
        return
    repo.insert_review_log(con, item_id, repo.ACTION_AMBIGUOUS_REACTION, None, description, at)


def _apply_status(
    con, post: DiscordPostRow, reactions: list[Reaction], report: ReadReport, at: datetime
) -> bool:
    """status のリアクションを反映する。status が1つに決まったら True。"""
    found = _present(reactions, STATUS_BY_EMOJI)
    if not found:
        return False
    item = repo.get_knowledge_item(con, post.item_id)
    if len(found) >= 2:
        description = f"status のリアクションが複数: {' '.join(found)}"
        _log_ambiguous_once(con, post.item_id, description, at)
        report.ambiguous.append(
            f"会議 {_conversation_of_item(con, post.item_id)} 項目 {post.item_id}: {description}"
            "(status を変えていません。1つだけにしてください)"
        )
        return False
    new_status = STATUS_BY_EMOJI[found[0]]
    if item.status != new_status:
        repo.update_item_status(con, post.item_id, new_status, at)
        repo.insert_review_log(con, post.item_id, repo.ACTION_STATUS_CHANGED, item.status, new_status, at)
        report.status_changes += 1
    return True


def _apply_importance(
    con, post: DiscordPostRow, reactions: list[Reaction], report: ReadReport, at: datetime
) -> None:
    """重要度のリアクションを反映する。

    上書きは importance_overridden として、before = LLM の判定値(現在の値)、after = ユーザーの値で残す。
    ユーザーの値が判定値と同じときも残す(before = after。LLM の判定に同意した記録になる)。
    同じリアクションを読み直しても重複して残さないよう、直前の記録の after と比べる。
    """
    found = _present(reactions, EMOJI_IMPORTANCE)
    if not found:
        return
    if len(found) >= 2:
        description = f"重要度のリアクションが複数: {' '.join(found)}"
        _log_ambiguous_once(con, post.item_id, description, at)
        report.ambiguous.append(
            f"会議 {_conversation_of_item(con, post.item_id)} 項目 {post.item_id}: {description}"
            "(importance を変えていません。1つだけにしてください)"
        )
        return
    value = EMOJI_IMPORTANCE[found[0]]
    previous = repo.list_review_log(con, post.item_id, repo.ACTION_IMPORTANCE_OVERRIDDEN)
    if previous and previous[-1].after == str(value):
        return
    item = repo.get_knowledge_item(con, post.item_id)
    repo.insert_review_log(
        con, post.item_id, repo.ACTION_IMPORTANCE_OVERRIDDEN,
        None if item.importance is None else str(item.importance), str(value), at,
    )
    if item.importance != value:
        repo.update_item_importance(con, post.item_id, value, at)
    report.importance_overrides += 1


def _handle_replies(
    con, router: ReplyRouter | None, post: DiscordPostRow, replies: list[Reply], report: ReadReport,
    at: datetime, other_item_id: str | None = None,
) -> bool:
    """返信をすべて振り分ける。1件でも失敗したら False(その投稿は未処理のまま残す)。"""
    all_done = True
    for reply in _owner_replies(replies):
        if router is None:
            # LLM を使えない。原文だけは記録しておき、次回に振り分ける
            _record_raw_only(con, post, reply, at)
            report.replies_failed += 1
            all_done = False
            continue
        outcome = process_reply(con, router, post.item_id, reply, at, other_item_id)
        if outcome.failed:
            report.replies_failed += 1
            all_done = False
        elif outcome.routed:
            report.replies_routed += 1
            for u in outcome.unapplied:
                report.messages.append(
                    f"項目 {post.item_id}: 返信の一部を反映できませんでした({u['reason']}): {u['text']}"
                    "(原文は review_log にあります)"
                )
    return all_done


def _record_raw_only(con, post: DiscordPostRow, reply: Reply, at: datetime) -> None:
    key = reply.created_at.isoformat()
    for row in repo.list_review_log(con, post.item_id, repo.ACTION_REPLY_RECEIVED):
        if row.before == key and row.after == reply.content:
            return
    repo.insert_review_log(con, post.item_id, repo.ACTION_REPLY_RECEIVED, key, reply.content, at)


def _read_new_candidate(con, post, reactions, replies, router, report, at) -> bool:
    if repo.get_knowledge_item(con, post.item_id) is None:
        report.messages.append(f"項目 {post.item_id} が knowledge_items にありません(投稿 {post.message_id})")
        return False
    decided = _apply_status(con, post, reactions, report, at)
    _apply_importance(con, post, reactions, report, at)
    replies_done = _handle_replies(con, router, post, replies, report, at)
    return decided and replies_done


def _read_contradiction(con, post, replies, router, report, at) -> bool:
    owner_replies = _owner_replies(replies)
    if not owner_replies:
        return False
    return _handle_replies(con, router, post, replies, report, at, other_item_id=post.pair_item_id)


def _read_speaker_sample(con, post, replies, interpreter, report) -> bool:
    labels = sorted({u.speaker_label for u in repo.get_utterances(con, post.conversation_id) if u.speaker_label})
    outcome = apply_speaker_replies(con, post.conversation_id, labels, replies, interpreter)
    report.messages.extend(outcome.messages)
    if outcome.changed:
        report.derived_conversations.append(post.conversation_id)
    return outcome.has_self and not outcome.failed


def _read_bundle_result(post, reactions, report) -> bool:
    emojis = {normalize_emoji(r.emoji) for r in reactions if r.count > 0}
    if EMOJI_REJECT in emojis:
        report.rebundle_needed.append(post.conversation_id)
        report.messages.append(
            f"会議 {post.conversation_id}: 束ね結果に ❌ が付きました。自動では分割し直しません。"
            "config.toml の bundle.gap_minutes を調整して `kijun ingest` をやり直してください"
        )
        return True
    return EMOJI_APPROVE in emojis


def run_read(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    client: DiscordClient,
    router: ReplyRouter | None,
    interpreter: SpeakerReplyInterpreter | None,
    now: datetime | None = None,
) -> ReadReport:
    """読み取り待ちの投稿を読み、DB に反映する。router / interpreter が None なら LLM を使わない(返信は未処理で残る)。"""
    now = now or datetime.now()
    report = ReadReport()
    if not cfg.discord.review_channel_id and not cfg.discord.contradictions_channel_id:
        report.skipped_for_config = True
        report.messages.append(
            "設定が必要です: config.toml の [discord] review_channel_id と contradictions_channel_id を書いてください。"
        )
        return report

    posts = repo.list_discord_posts(con, unprocessed_only=True)
    by_channel: dict[str, list[DiscordPostRow]] = {}
    for p in posts:
        channel = p.channel_id or cfg.discord.review_channel_id
        by_channel.setdefault(channel, []).append(p)

    for channel, channel_posts in by_channel.items():
        ids = [p.message_id for p in channel_posts]
        all_reactions = client.fetch_reactions(channel, ids)
        all_replies = client.fetch_thread_replies(channel, ids)
        for post in channel_posts:
            if post.message_id not in all_reactions or post.message_id not in all_replies:
                report.messages.append(
                    f"投稿 {post.message_id}({post.post_type})を Discord から取得できませんでした"
                    "(削除された可能性があります)。未処理のまま残します"
                )
                continue
            reactions, replies = all_reactions[post.message_id], all_replies[post.message_id]
            if post.post_type == repo.POST_NEW_CANDIDATE:
                done = _read_new_candidate(con, post, reactions, replies, router, report, now)
            elif post.post_type == repo.POST_CONTRADICTION:
                done = _read_contradiction(con, post, replies, router, report, now)
            elif post.post_type == repo.POST_SPEAKER_SAMPLE:
                done = _read_speaker_sample(con, post, replies, interpreter, report)
            elif post.post_type == repo.POST_BUNDLE_RESULT:
                done = _read_bundle_result(post, reactions, report)
            else:
                report.messages.append(f"投稿 {post.message_id} の種類 {post.post_type} は未対応です")
                done = False
            if done:
                repo.mark_post_processed(con, post.message_id, now)
                report.processed_posts += 1

    # 話者の対応が変わった会議は、最後に direction を導出し直す
    for conversation_id in dict.fromkeys(report.derived_conversations):
        result = derive_directions(con, conversation_id)
        if result.derived:
            report.messages.append(
                f"会議 {conversation_id}: direction を導出しました"
                f"(自分→相手 {result.self_to_other}、相手→自分 {result.other_to_self}、未確定 {result.undetermined})"
            )
    return report
