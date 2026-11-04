"""Discord への4種類の投稿(仕様 2-1、CLAUDE.md 4-6、4-8 の手順8)。

- 新規候補(#review): 1件1メッセージ。1回の投稿件数に上限を置く(discord.max_candidates_per_run)。
- 矛盾の可能性がある組(#contradictions): 検出規則は review/contradiction.py。
- 話者の対応付けの依頼: 会議ごとに各話者の発言サンプル。
- 束ね結果: 複数のファイルを束ねた会議。

投稿するたびに、その場で discord_posts に記録する(途中で落ちても、投稿済みのものを再投稿しないため)。
1件の投稿の失敗では全体を止めない。失敗は報告に残し、終了コードで知らせる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.db.repo import KnowledgeItemRow
from kijun.discord import format as fmt
from kijun.discord.base import DiscordClient, DiscordError
from kijun.review.contradiction import find_contradictions
from kijun.review.importance import ImportanceJudge, propose_importance


@dataclass
class PostReport:
    posted: dict[str, int] = field(default_factory=dict)  # post_type -> 投稿した件数
    not_posted_candidates: int = 0  # 上限を超えて、次回に回した新規候補の件数
    failures: list[str] = field(default_factory=list)  # 投稿できなかったものの理由
    messages: list[str] = field(default_factory=list)
    skipped_for_config: bool = False  # チャンネルIDが空で何もしなかった

    @property
    def failed(self) -> bool:
        return bool(self.failures)


def pending_candidates(con: duckdb.DuckDBPyConnection) -> list[KnowledgeItemRow]:
    """status が candidate で、まだ new_candidate として投稿していない項目。"""
    posted = {p.item_id for p in repo.list_discord_posts(con, repo.POST_NEW_CANDIDATE)}
    return [i for i in repo.list_knowledge_items(con, [repo.STATUS_CANDIDATE]) if i.item_id not in posted]


def count_candidates_needing_importance(con: duckdb.DuckDBPyConnection) -> int:
    """投稿前に重要度の判定(LLM)が要る項目の件数。夜間バッチが Strata を起動するかの判断に使う。"""
    return sum(1 for i in pending_candidates(con) if i.importance is None)


def _first_linked_at(con: duckdb.DuckDBPyConnection) -> dict[str, datetime]:
    rows = con.execute("SELECT item_id, min(linked_at) FROM item_evidence GROUP BY item_id").fetchall()
    return {item_id: linked_at for item_id, linked_at in rows}


def order_candidates(
    items: list[KnowledgeItemRow], linked_at: dict[str, datetime]
) -> list[KnowledgeItemRow]:
    """重要度の降順、同じ重要度なら新しい順(仕様 1-4)。

    レビューの時間が足りないときに、重要なものが先に目に入るようにするため。
    重要度が NULL のものは最後に置く。「新しい」は、根拠の抽出結果を最初に紐付けた時刻(候補ができた時刻)で決め、
    同じ時刻なら item_id の大きい方を新しいとする。
    """
    ordered = sorted(items, key=lambda i: i.item_id, reverse=True)
    ordered = sorted(ordered, key=lambda i: linked_at.get(i.item_id, datetime.min), reverse=True)
    return sorted(ordered, key=lambda i: i.importance or 0, reverse=True)


def _send(
    con: duckdb.DuckDBPyConnection,
    client: DiscordClient,
    report: PostReport,
    channel_id: str,
    text: str,
    post_type: str,
    now: datetime,
    *,
    label: str,
    item_id: str | None = None,
    conversation_id: str | None = None,
    pair_item_id: str | None = None,
    request_id: str | None = None,
) -> bool:
    """1件投稿して discord_posts に記録する。投稿できたら True。"""
    if len(text) > fmt.DISCORD_MESSAGE_LIMIT:
        # 黙って切り詰めると、レビューの対象(チェック項目案など)が見えなくなる
        report.failures.append(
            f"{label}: 文面が {len(text)} 文字で、Discord の上限 {fmt.DISCORD_MESSAGE_LIMIT} 文字を超えるため投稿しません"
        )
        return False
    try:
        message_id = client.post(channel_id, text)
    except DiscordError as e:
        report.failures.append(f"{label}: 投稿できませんでした: {e}")
        return False
    repo.insert_discord_post(
        con,
        repo.DiscordPostRow(
            message_id=message_id, post_type=post_type, item_id=item_id,
            conversation_id=conversation_id, posted_at=now, processed_at=None,
            channel_id=channel_id, pair_item_id=pair_item_id, request_id=request_id,
        ),
    )
    report.posted[post_type] = report.posted.get(post_type, 0) + 1
    return True


def post_bundle_results(con, cfg: Config, client: DiscordClient, report: PostReport, now: datetime) -> None:
    """複数のファイルを束ねた会議のうち、まだ投稿していないものを投稿する。"""
    done = {p.conversation_id for p in repo.list_discord_posts(con, repo.POST_BUNDLE_RESULT)}
    for conv in repo.list_conversations(con):
        if len(conv.source_files) < 2 or conv.conversation_id in done:
            continue
        _send(
            con, client, report, cfg.discord.review_channel_id, fmt.format_bundle_result(conv),
            repo.POST_BUNDLE_RESULT, now, label=f"束ね結果 {conv.conversation_id}",
            conversation_id=conv.conversation_id,
        )


def post_speaker_samples(con, cfg: Config, client: DiscordClient, report: PostReport, now: datetime) -> None:
    """話者の対応付けの依頼を、まだ投稿していない・self が決まっていない会議について投稿する。"""
    done = {p.conversation_id for p in repo.list_discord_posts(con, repo.POST_SPEAKER_SAMPLE)}
    for conv in repo.list_conversations(con):
        if conv.conversation_id in done:
            continue
        if repo.PERSON_SELF in repo.get_speaker_map(con, conv.conversation_id).values():
            continue
        utterances = repo.get_utterances(con, conv.conversation_id)
        samples = fmt.pick_speaker_samples(
            [(u.speaker_label, u.text) for u in utterances], cfg.discord.speaker_sample_count
        )
        if not samples:
            continue  # 文字起こし前、または話者分離が失敗した会議
        _send(
            con, client, report, cfg.discord.review_channel_id,
            fmt.format_speaker_sample(conv.conversation_id, samples),
            repo.POST_SPEAKER_SAMPLE, now, label=f"話者の対応付け {conv.conversation_id}",
            conversation_id=conv.conversation_id,
        )


def post_new_candidates(
    con, cfg: Config, client: DiscordClient, judge: ImportanceJudge | None, report: PostReport, now: datetime
) -> None:
    """新規候補を、重要度の降順・新しい順に、上限の件数まで投稿する。

    重要度の降順に並べるには、投稿しない分も含めて判定が要る。判定は項目に保存するので、
    次回以降は再判定しない。判定に失敗した項目は重要度が NULL のまま(「重要度案：なし」)投稿する。
    """
    pending = pending_candidates(con)
    for item in pending:
        if item.importance is None:
            propose_importance(con, judge, item, now)
    pending = pending_candidates(con)  # 判定結果を読み直す
    ordered = order_candidates(pending, _first_linked_at(con))
    limit = cfg.discord.max_candidates_per_run
    for item in ordered[:limit]:
        evidence = repo.get_item_evidence_requests(con, item.item_id, 1)
        request = evidence[0] if evidence else None
        conv = repo.get_conversation(con, request.conversation_id) if request else None
        speaker_map = repo.get_speaker_map(con, request.conversation_id) if request else {}
        _send(
            con, client, report, cfg.discord.review_channel_id,
            fmt.format_new_candidate(item, request, conv, speaker_map),
            repo.POST_NEW_CANDIDATE, now, label=f"新規候補 {item.item_id}", item_id=item.item_id,
            conversation_id=request.conversation_id if request else None,
        )
    # 上限で回した分と、投稿に失敗した分を合わせた、まだ投稿していない件数
    report.not_posted_candidates = len(pending_candidates(con))


def pending_transcription_unclear(con: duckdb.DuckDBPyConnection, cfg: Config) -> list[repo.RequestRow]:
    """文字起こしが読めなかった抽出結果のうち、まだ投稿していないもの。

    投稿済みかどうかは discord_posts.request_id で見る。この種別は knowledge_items に
    ならないので、item_id では記録できない（バージョン3で request_id を足した）。
    """
    posted = {
        p.request_id for p in repo.list_discord_posts(con, repo.POST_TRANSCRIPTION_UNCLEAR)
    }
    rows = repo.get_transcription_unclear_requests(con, cfg.extract.model_name())
    return [r for r in rows if r.request_id not in posted]


def post_transcription_unclear(
    con, cfg: Config, client: DiscordClient, report: PostReport, now: datetime
) -> None:
    """文字起こしが読めなかった抽出結果を、上限の件数まで #review に投稿する（CLAUDE.md 7章の案A）。

    これらは reusable = FALSE なので照合の対象から外れ、knowledge_items にならない。投稿するのは、
    発話したのが利用者本人であり、壊れた引用を見れば自分が何と言ったか分かるためである。
    投稿しなければ、同じ発話が繰り返し失われ続ける（「箇条書き」「感嘆符」は頻出語である）。
    """
    pending = pending_transcription_unclear(con, cfg)
    for request in pending[: cfg.discord.max_transcription_unclear_per_run]:
        conv = repo.get_conversation(con, request.conversation_id)
        speaker_map = repo.get_speaker_map(con, request.conversation_id)
        _send(
            con, client, report, cfg.discord.review_channel_id,
            fmt.format_transcription_unclear(request, conv, speaker_map),
            repo.POST_TRANSCRIPTION_UNCLEAR, now,
            label=f"文字起こしが読めなかった抽出結果 {request.request_id}",
            conversation_id=request.conversation_id, request_id=request.request_id,
        )


def post_contradictions(con, cfg: Config, client: DiscordClient, report: PostReport, now: datetime) -> None:
    for pair in find_contradictions(con, cfg)[: cfg.discord.max_contradictions_per_run]:
        _send(
            con, client, report, cfg.discord.contradictions_channel_id,
            fmt.format_contradiction(pair.a, pair.b, pair.similarity),
            repo.POST_CONTRADICTION, now, label=f"矛盾 {pair.a.item_id}/{pair.b.item_id}",
            item_id=pair.a.item_id, pair_item_id=pair.b.item_id,
        )


def run_post(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    client: DiscordClient,
    judge: ImportanceJudge | None,
    now: datetime | None = None,
) -> PostReport:
    """5種類を投稿する。チャンネルIDが空の種類は、設定が必要であることを報告して飛ばす。"""
    now = now or datetime.now()
    report = PostReport()
    review, contra = cfg.discord.review_channel_id, cfg.discord.contradictions_channel_id
    if not review and not contra:
        report.skipped_for_config = True
        report.messages.append(
            "設定が必要です: config.toml の [discord] review_channel_id と contradictions_channel_id を書いてください。"
        )
        return report
    if review:
        post_bundle_results(con, cfg, client, report, now)
        post_speaker_samples(con, cfg, client, report, now)
        post_new_candidates(con, cfg, client, judge, report, now)
        post_transcription_unclear(con, cfg, client, report, now)
    else:
        report.messages.append(
            "review_channel_id が空のため、新規候補・話者の対応付け・束ね結果・"
            "文字起こしが読めなかった抽出結果は投稿しません。"
        )
    if contra:
        post_contradictions(con, cfg, client, report, now)
    else:
        report.messages.append("contradictions_channel_id が空のため、矛盾の可能性がある組は投稿しません。")
    return report
