"""議事録の Discord への投稿。

本文は 2000 文字以内(Discord の上限)に収め、Markdown 全体を添付ファイルにする。
本文の構成: 見出し行(会議ID・日時) + 概要 + 決定事項 + ToDo。Discord は Markdown の表を表示しないので、
ToDo は箇条書きに直す。超えたときは末尾を切って「(続きは添付ファイル)」と書く。

投稿済み(minutes.posted_at が非 NULL)は再投稿しない。1件の投稿に失敗しても、残りの投稿は続ける。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.discord.base import DiscordClient, DiscordError
from kijun.minutes.render import (
    EMPTY,
    SECTION_DECISIONS,
    SECTION_OVERVIEW,
    SECTION_TODOS,
    parse_sections,
    todo_rows,
)

# Discord の本文の上限(文字数)
DISCORD_LIMIT = 2000
CONTINUED_NOTICE = "\n（続きは添付ファイル）"


def _todo_lines(body: str) -> list[str]:
    rows = todo_rows(body)
    if not rows:
        return [EMPTY]
    lines = []
    for task, owner, due in rows:
        lines.append(f"- {task}（担当: {owner}、期限: {due}）")
    return lines


def _title_line(markdown: str, conversation_id: str) -> str:
    """見出し行。日時は Markdown の「- 日時: 」の行から読む(無ければ会議IDだけ)。"""
    head, _ = parse_sections(markdown)
    when = ""
    for line in head.splitlines():
        if line.startswith("- 日時:"):
            when = line.split(":", 1)[1].strip()
            break
    return f"**議事録 {conversation_id}**" + (f"（{when}）" if when else "")


def build_post_text(markdown: str, conversation_id: str, limit: int = DISCORD_LIMIT) -> str:
    """Discord に投稿する本文を作る。limit 文字を超えるときは切り詰めて、続きが添付にあると書く。"""
    _, sections = parse_sections(markdown)
    overview = sections.get(SECTION_OVERVIEW, EMPTY) or EMPTY
    decisions = sections.get(SECTION_DECISIONS, EMPTY) or EMPTY
    todos = "\n".join(_todo_lines(sections.get(SECTION_TODOS, "")))
    text = "\n\n".join(
        [
            _title_line(markdown, conversation_id),
            f"**{SECTION_OVERVIEW}**\n{overview}",
            f"**{SECTION_DECISIONS}**\n{decisions}",
            f"**{SECTION_TODOS}**\n{todos}",
        ]
    )
    if len(text) <= limit:
        return text
    keep = limit - len(CONTINUED_NOTICE)
    cut = text[:keep]
    # 行の途中で切れるのを避ける。ただし、残りが半分未満になるほど戻らない。
    newline = cut.rfind("\n")
    if newline >= keep // 2:
        cut = cut[:newline]
    return cut.rstrip() + CONTINUED_NOTICE


@dataclass
class PostReport:
    posted: list[str] = field(default_factory=list)  # 投稿した会議ID
    failures: list[str] = field(default_factory=list)  # 「会議ID: 理由」
    skipped_for_config: bool = False
    messages: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.failures)


def run_post(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    client: DiscordClient,
    now: datetime | None = None,
) -> PostReport:
    """未投稿の議事録をすべて投稿する。設定 discord.minutes_channel_id が空なら何もしない。"""
    report = PostReport()
    channel = cfg.discord.minutes_channel_id
    if not channel:
        report.skipped_for_config = True
        report.messages.append(
            "設定 discord.minutes_channel_id が空のため、何も投稿しませんでした。"
            "Discord のチャンネルIDを config.toml の [discord] に書いてください。"
        )
        return report
    for row in repo.list_unposted_minutes(con):
        path = Path(row.path)
        try:
            markdown = path.read_text(encoding="utf-8")
        except OSError as e:
            report.failures.append(f"{row.conversation_id}: 議事録のファイルを読めません({path}): {e}")
            continue
        text = build_post_text(markdown, row.conversation_id)
        try:
            message_id = client.post_file(channel, text, f"{row.conversation_id}.md", markdown.encode("utf-8"))
        except DiscordError as e:
            report.failures.append(f"{row.conversation_id}: {e}")
            continue
        repo.mark_minutes_posted(con, row.conversation_id, message_id, now or datetime.now())
        report.posted.append(row.conversation_id)
    return report


def print_post_report(report: PostReport, echo: Callable[[str], None]) -> None:
    for line in report.messages:
        echo(line)
    for line in report.failures:
        echo(f"失敗: {line}")
    if report.skipped_for_config:
        return
    posted = "、".join(report.posted) or "なし"
    echo(f"投稿: {len(report.posted)} 件（{posted}）")
