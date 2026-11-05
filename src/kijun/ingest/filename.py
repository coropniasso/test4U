"""音声ファイル名の解析。

書式は `yyyyMMdd_HHmm_<形式>_<種類>.m4a`（CLAUDE.md 4-2）。
例: 20260920_1930_f2f_teirei.m4a

- <形式> が設定の formats に無ければ解析を失敗させる。形式別の集計が壊れるため。
- <種類> が設定の kinds に無ければ、警告を標準エラーに出したうえで受け入れる。
  CLAUDE.md 4-2 に「運用しながら追加」と書いてあり、拒否すると運用が止まるため。
- 解析に失敗したファイルは、削除も移動もしない。録音は録り直せないため。
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

from kijun.config import SettingsConfig

FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{4})_(?P<fmt>[a-z0-9]+)_(?P<kind>[a-z0-9]+)"
    r"(?:_(?P<part>\d+))?\.(?P<ext>m4a|wav|mp3)$"
)


class FilenameError(ValueError):
    """ファイル名を解析できないときの例外。メッセージに理由を書く。"""


@dataclass(frozen=True)
class ParsedName:
    """解析したファイル名。

    kind_known は、setting_kind が設定の kinds に載っているかどうか。
    """

    date: date
    time: time
    recorded_at: datetime
    setting_format: str
    setting_kind: str
    part: int | None
    ext: str
    path: Path
    kind_known: bool = True


def warn_stderr(message: str) -> None:
    print(f"警告: {message}", file=sys.stderr)


def parse_filename(
    path: Path | str,
    settings: SettingsConfig,
    warn: Callable[[str], None] = warn_stderr,
) -> ParsedName:
    """ファイル名を解析して ParsedName を返す。

    解析できなければ FilenameError を出す。未知の <種類> は warn を呼んで受け入れる。
    """
    p = Path(path)
    m = FILENAME_PATTERN.match(p.name)
    if m is None:
        raise FilenameError(
            f"ファイル名が書式 yyyyMMdd_HHmm_<形式>_<種類>[_連番].(m4a|wav|mp3) に合いません: {p.name}"
        )
    try:
        recorded_at = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M")
    except ValueError as e:
        raise FilenameError(f"日時として不正です: {p.name}（{e}）") from e

    fmt = m["fmt"]
    if fmt not in settings.formats:
        raise FilenameError(
            f"<形式> '{fmt}' は設定の formats {settings.formats} にありません: {p.name}"
        )

    kind = m["kind"]
    kind_known = kind in settings.kinds
    if not kind_known:
        warn(
            f"<種類> '{kind}' は設定の kinds {settings.kinds} にありません。"
            f"そのまま記録します: {p.name}"
        )

    return ParsedName(
        date=recorded_at.date(),
        time=recorded_at.time(),
        recorded_at=recorded_at,
        setting_format=fmt,
        setting_kind=kind,
        part=int(m["part"]) if m["part"] is not None else None,
        ext=m["ext"],
        path=p,
        kind_known=kind_known,
    )


def scan_inbox(
    inbox: Path,
    settings: SettingsConfig,
    warn: Callable[[str], None] = warn_stderr,
) -> tuple[list[ParsedName], list[tuple[Path, str]]]:
    """inbox 直下のファイルを解析する。

    戻り値は (解析できたファイル, [(解析できなかったファイル, 理由)])。
    解析できなかったファイルには何もしない（削除も移動もしない）。
    サブディレクトリと、名前が "." で始まるファイルは対象にしない。
    """
    parsed: list[ParsedName] = []
    failed: list[tuple[Path, str]] = []
    if not inbox.exists():
        return parsed, failed
    for p in sorted(inbox.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            parsed.append(parse_filename(p, settings, warn))
        except FilenameError as e:
            failed.append((p, str(e)))
    return parsed, failed
