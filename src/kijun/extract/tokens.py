"""入力の長さをトークン数で概算する。

この係数は概算である。Qwen3.8-Flash-Next の実際のトークナイザーは、51B の n-gram 埋め込み
テーブルを持つため、日本語の1文字あたりのトークン数がこの係数とずれる可能性がある。
正確な値はこのクラウド環境では測れないので、`uv run kijun doctor` の手順4・手順5で
実測し、その結果を config.toml の max_input_tokens に書いて置き換える。

係数をここ1か所に置くのは、run.py の入力長の警告と doctor.py の入力の引き伸ばしで
同じ値を使うためである。
"""

from __future__ import annotations

# 日本語1文字あたりのトークン数の概算。
# 根拠: 日本語4万字が概ね4万〜6万トークンという概算（計画書 7-6）の上側を取った値。
# 実測値ではないので、doctor の結果で置き換えるべきものである。
TOKENS_PER_JAPANESE_CHARACTER = 1.5


def estimate_tokens(text: str) -> int:
    """文字列の入力トークン数を概算する。"""
    return int(len(text) * TOKENS_PER_JAPANESE_CHARACTER)


def estimate_messages_tokens(system: str, user: str) -> int:
    """system と user のプロンプトを合わせた入力トークン数を概算する。"""
    return estimate_tokens(system) + estimate_tokens(user)


# 会議の音声1分あたりの入力トークン数の概算。
# 根拠: 2時間の会議が日本語3〜4万字で約6万トークンという概算（計画書 7-6）から、
# 60000 / 120 = 500 とした。doctor が入力を引き伸ばすときと、実測値から
# chunk_minutes を逆算するときに使う。これも実測ではないので、doctor の結果で置き換える。
TOKENS_PER_MEETING_MINUTE = 500


def tokens_for_minutes(minutes: float) -> int:
    """会議の分数を、文字起こしの入力トークン数の概算に変換する。"""
    return int(minutes * TOKENS_PER_MEETING_MINUTE)


def minutes_for_tokens(tokens: int) -> float:
    """入力トークン数を、会議の分数の概算に変換する（tokens_for_minutes の逆）。"""
    return tokens / TOKENS_PER_MEETING_MINUTE
