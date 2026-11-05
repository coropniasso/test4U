"""入力の長さをトークン数で概算する。

係数は設定（minutes.tokens_per_character と minutes.tokens_per_meeting_minute）から受け取る。
このモジュールに定数を持たない。

既定値（0.55 と 180）は実測1点からの推測である。以前の抽出（Strata の Qwen3.8-Flash-Next）で、
概算 4,711 トークン（係数 1.5）に対し実際は 1,646 トークンで、約2.9倍の過大評価だった。
この値は Strata での測定で、Ollama で使うモデルのトークナイザには当てはまらない。
`uv run kijun doctor` が tokens_per_character を実測して推奨値に出すので、それで置き換える。

係数をここ1か所の関数に通すのは、minutes/generate.py の入力長の警告と doctor.py の入力の引き伸ばしで
同じ値を使うためである。
"""

from __future__ import annotations


def estimate_tokens(text: str, tokens_per_character: float) -> int:
    """文字列の入力トークン数を概算する。"""
    return int(len(text) * tokens_per_character)


def estimate_messages_tokens(system: str, user: str, tokens_per_character: float) -> int:
    """system と user のプロンプトを合わせた入力トークン数を概算する。"""
    return estimate_tokens(system, tokens_per_character) + estimate_tokens(user, tokens_per_character)


def tokens_for_minutes(minutes: float, tokens_per_meeting_minute: int) -> int:
    """会議の分数を、文字起こしの入力トークン数の概算に変換する。"""
    return int(minutes * tokens_per_meeting_minute)
