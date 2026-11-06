"""議事録の Markdown の組み立てと、Markdown からの読み出し（Discord の本文の材料）。

書式（節の見出しは `## `。順序は固定）:

    # 議事録 <会議ID>
    - 日時 / 形式・種類 / 長さ / 話者 / 作成
    ## 概要
    ## 議題と要点        （### 議題名 → 箇条書き）
    ## 決定事項
    ## ToDo              （表：内容 | 担当 | 期限。null は「記載なし」）
    ## 未決事項
    ## 付録: 文字起こし   （[mm:ss] SPEAKER_xx: 本文）

空の節は「なし」と書く。Discord の本文は、この Markdown の節を読んで作る（discord/post.py）ので、
節の見出しの文字列はここで定数にして共有する。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from kijun.config import Config
from kijun.models import Conversation, Utterance

SECTION_OVERVIEW = "概要"
SECTION_TOPICS = "議題と要点"
SECTION_DECISIONS = "決定事項"
SECTION_TODOS = "ToDo"
SECTION_OPEN_ISSUES = "未決事項"
SECTION_APPENDIX = "付録: 文字起こし"

EMPTY = "なし"
NOT_STATED = "記載なし"

_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def minutes_path(cfg: Config, conversation_id: str) -> Path:
    return cfg.paths.minutes / f"{conversation_id}.md"


def _timestamp(sec: float) -> str:
    total = int(sec)
    return f"{total // 60:02d}:{total % 60:02d}"


def _cell(text: str | None) -> str:
    """表のセルに入れる文字列。改行は空白にし、| は \\| にする。None は「記載なし」。"""
    if text is None:
        return NOT_STATED
    return " ".join(text.split()).replace("|", "\\|")


def _bullets(items: list[str]) -> list[str]:
    return [f"- {t}" for t in items] if items else [EMPTY]


def render_markdown(
    conv: Conversation,
    utterances: list[Utterance],
    doc,  # minutes.generate.MinutesDoc（循環 import を避けるため型を書かない）
    model_name: str,
    created_at: datetime,
) -> str:
    """議事録の Markdown 全文を作る。"""
    speakers = sorted({u.speaker_label or "UNKNOWN" for u in utterances})
    if conv.duration_sec is not None:
        length = f"{round(conv.duration_sec / 60)}分"
    elif utterances:
        length = f"{round(utterances[-1].end_sec / 60)}分"
    else:
        length = "不明"
    m = doc.merged

    lines: list[str] = [
        f"# 議事録 {conv.conversation_id}",
        "",
        f"- 日時: {conv.recorded_at:%Y-%m-%d %H:%M}",
        f"- 形式・種類: {conv.setting_format} / {conv.setting_kind}",
        f"- 長さ: {length}",
        f"- 話者: {', '.join(speakers) if speakers else EMPTY}",
        f"- 作成: {model_name}、{created_at:%Y-%m-%d %H:%M}（機械の文字起こしと要約。誤りを含む）",
        *[f"- 注意: {w}" for w in getattr(doc, "warnings", [])],
        "",
        f"## {SECTION_OVERVIEW}",
        "",
        doc.overview,
        "",
        f"## {SECTION_TOPICS}",
        "",
    ]
    if m.topics:
        for topic in m.topics:
            lines += [f"### {topic.title}", "", *[f"- {p}" for p in topic.points], ""]
        lines.pop()  # 最後の空行（次の節の前で入れる）
    else:
        lines.append(EMPTY)
    lines += ["", f"## {SECTION_DECISIONS}", "", *_bullets(m.decisions), ""]
    lines += [f"## {SECTION_TODOS}", ""]
    if m.todos:
        lines += ["| 内容 | 担当 | 期限 |", "|---|---|---|"]
        lines += [f"| {_cell(t.task)} | {_cell(t.owner)} | {_cell(t.due)} |" for t in m.todos]
    else:
        lines.append(EMPTY)
    lines += ["", f"## {SECTION_OPEN_ISSUES}", "", *_bullets(m.open_issues), ""]
    lines += [f"## {SECTION_APPENDIX}", ""]
    if utterances:
        # 行末の空白2つは Markdown の改行。1行ずつ表示させるため。
        lines += [f"[{_timestamp(u.start_sec)}] {u.speaker_label or 'UNKNOWN'}: {u.text}  " for u in utterances]
    else:
        lines.append(EMPTY)
    return "\n".join(lines) + "\n"


# --- 読み出し（Discord の本文用） ------------------------------------------------------


def parse_sections(markdown: str) -> tuple[str, dict[str, str]]:
    """Markdown を (先頭の見出し行からなる冒頭部, {節名: 本文}) に分ける。

    節は `## ` で始まる行から次の `## ` の直前まで。「付録: 文字起こし」以降は読まない
    （文字起こしの行が節の見出しと紛れないように、付録の見出しで打ち切る）。
    """
    head: list[str] = []
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in markdown.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
            if name == SECTION_APPENDIX:
                break
            current = sections.setdefault(name, [])
            continue
        (head if current is None else current).append(line)
    return "\n".join(head).strip(), {k: "\n".join(v).strip() for k, v in sections.items()}


def todo_rows(section_body: str) -> list[tuple[str, str, str]]:
    """ToDo の節の表から (内容, 担当, 期限) を読む。表が無い（「なし」）なら空。"""
    rows: list[tuple[str, str, str]] = []
    for line in section_body.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| 内容"):
            continue
        cells = [c.strip().replace("\\|", "|") for c in _UNESCAPED_PIPE.split(line.strip("|"))]
        if len(cells) == 3:
            rows.append((cells[0], cells[1], cells[2]))
    return rows
