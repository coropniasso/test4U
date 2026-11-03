"""Discord のインターフェース(仕様 2-2)。

Discord に触る処理は、この Protocol の背後に置く(原則2-1)。実際の Discord に接続せずに
テストできるよう、テストでは tests/fakes.py の FakeDiscordClient を差し込む。

bot は常駐しない(CLAUDE.md 4-6、仕様 1-1)。夜間バッチの間だけ接続し、終わったら閉じる。
そのため Protocol は「投稿する」「リアクションを読む」「返信を読む」の3つだけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

# 絵文字の異体字セレクタ(U+FE0F)。Discord は「⏸」を「⏸️」の形で返すことがあるので、比較の前に外す。
_VARIATION_SELECTOR = "\ufe0f"


def normalize_emoji(emoji: str) -> str:
    """異体字セレクタを除いた絵文字を返す。リアクションの絵文字は、この関数を通してから比較する。"""
    return emoji.replace(_VARIATION_SELECTOR, "")


# --- レビューで使う絵文字(CLAUDE.md 4-6)。比較用に、すべて normalize_emoji を通した形で持つ ---
EMOJI_APPROVE = normalize_emoji("✅")
EMOJI_REJECT = normalize_emoji("❌")
EMOJI_HOLD = normalize_emoji("⏸")
EMOJI_IMPORTANCE = {normalize_emoji(e): n for e, n in (("1️⃣", 1), ("2️⃣", 2), ("3️⃣", 3))}


@dataclass(frozen=True)
class Reaction:
    """メッセージに付いているリアクション1種類。count は付けた人数。"""

    emoji: str
    count: int


@dataclass(frozen=True)
class Reply:
    """メッセージへの返信1件(スレッド内の発言、またはそのメッセージへの返信)。"""

    author_is_owner: bool  # サーバーの所有者(ユーザー本人)の返信か。他のメンバーや bot 自身は False
    content: str
    created_at: datetime  # ローカル時刻の naive な datetime(DuckDB の TIMESTAMP と揃える)


class DiscordError(Exception):
    """Discord との通信に失敗した、または権限・設定が足りないときの例外。"""


class DiscordClient(Protocol):
    def post(self, channel_id: str, content: str) -> str:
        """チャンネルに投稿し、message_id を返す。呼び出し側が discord_posts に記録する。"""
        ...

    def fetch_reactions(
        self, channel_id: str, message_ids: list[str]
    ) -> dict[str, list[Reaction]]:
        """メッセージごとのリアクション。取得できなかった(削除された等)メッセージはキーに含めない。"""
        ...

    def fetch_thread_replies(
        self, channel_id: str, message_ids: list[str]
    ) -> dict[str, list[Reply]]:
        """メッセージごとの返信(古い順)。取得できなかったメッセージはキーに含めない。"""
        ...
