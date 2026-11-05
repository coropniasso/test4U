"""Discord のインターフェース。

Discord に触る処理は、この Protocol の背後に置く(原則2-1)。実際の Discord に接続せずに
テストできるよう、テストでは tests/fakes.py の FakeDiscordClient を差し込む。

bot は常駐しない。夜間バッチの間だけ接続し、終わったら閉じる。議事録の投稿だけを行うので、
Protocol は「本文とファイル1つを投稿する」だけを持つ(リアクションと返信の読み取りは、
チェックリスト時代の機能で、2026-11-06 に削除した)。
"""

from __future__ import annotations

from typing import Protocol


class DiscordError(Exception):
    """Discord との通信に失敗した、または権限・設定が足りないときの例外。"""


class DiscordClient(Protocol):
    def post_file(self, channel_id: str, content: str, filename: str, data: bytes) -> str:
        """チャンネルに本文を投稿し、ファイルを1つ添付して、message_id を返す。失敗したら DiscordError。"""
        ...
