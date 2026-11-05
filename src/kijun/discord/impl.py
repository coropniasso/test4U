"""discord.py を使う DiscordClient の実装。

この環境(GPU も Discord のトークンも無いクラウドコンテナ)では実接続を確認できない。
discord.py は `uv sync --extra discord` で入る。deps.py の仕組みで遅延 import し、
入っていなければ入れ方を示す。

接続は呼び出しごとに開いて閉じる(bot を常駐させない)。ゲートウェイには接続せず、REST だけで投稿する。
投稿だけなので Message Content Intent は要らない。
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Awaitable, Callable

from kijun.deps import import_optional
from kijun.discord.base import DiscordError


class DiscordPyClient:
    def __init__(self, token: str) -> None:
        self._discord = import_optional("discord", "discord", "Discord への接続")
        self._token = token

    def _run(self, make: Callable[[object], Awaitable]):
        """ログインして make(client) を実行し、閉じる。通信の失敗は DiscordError にする。"""
        discord = self._discord

        async def main():
            client = discord.Client(intents=discord.Intents.default())
            await client.login(self._token)
            try:
                return await make(client)
            finally:
                await client.close()

        try:
            return asyncio.run(main())
        except discord.LoginFailure as e:
            raise DiscordError(
                "Discord にログインできませんでした。環境変数 DISCORD_BOT_TOKEN を確認してください。"
            ) from e
        except discord.Forbidden as e:
            raise DiscordError(
                "Discord が権限不足を返しました。bot をサーバーに招待し、"
                f"チャンネルの閲覧・投稿・ファイルの添付の権限を付けてください: {e}"
            ) from e
        except discord.HTTPException as e:
            raise DiscordError(f"Discord との通信に失敗しました: {e}") from e

    def post_file(self, channel_id: str, content: str, filename: str, data: bytes) -> str:
        discord = self._discord

        async def make(client):
            channel = await client.fetch_channel(int(channel_id))
            message = await channel.send(content, file=discord.File(io.BytesIO(data), filename=filename))
            return str(message.id)

        return self._run(make)
