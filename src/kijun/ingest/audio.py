"""音声の長さの取得。

mutagen は pure Python で、ffmpeg のような外部コマンドを必要としない。
Windows への導入を1つ減らすため、ffprobe は使わない。
"""

from __future__ import annotations

from pathlib import Path

import mutagen


def get_duration_sec(path: Path | str) -> float | None:
    """音声の長さを秒で返す。読めなかった場合は None を返す。

    None のとき、呼び出し側が警告を出す。
    """
    try:
        audio = mutagen.File(str(path))
    except Exception:  # mutagen は壊れたファイルでさまざまな例外を出す
        return None
    if audio is None or audio.info is None:
        return None
    length = getattr(audio.info, "length", None)
    if length is None or length < 0:
        return None
    return float(length)
