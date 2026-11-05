"""標準出力に出す表の整形。日本語（全角）の表示幅を2として桁を揃える。"""

from __future__ import annotations

import unicodedata


def display_width(text: str) -> int:
    """端末での表示幅。全角と幅広の文字を2、それ以外を1として数える。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * (width - display_width(text))


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """見出しと行から、列を揃えたテキストの表を作る。行が無ければ見出しだけを出す。"""
    widths = [display_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], display_width(cell))
    lines = ["  ".join(_pad(h, widths[i]) for i, h in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(_pad(c, widths[i]) for i, c in enumerate(row)).rstrip())
    return "\n".join(lines)
