"""discord.py を使う DiscordClient の実装(仕様 2-2)。

この環境(GPU も Discord のトークンも無いクラウドコンテナ)では実接続を確認できない。
discord.py は `uv sync --extra discord` で入る。deps.py の仕組みで遅延 import し、
入っていなければ入れ方を示す。

接続は呼び出しごとに開いて閉じる(bot を常駐させない。CLAUDE.md 4-6)。ゲートウェイには
接続せず、REST だけで読み書きする。

Message Content Intent を有効にしていないと、他人のメッセージの本文が空で返ってくる。
本文が空の返信を見つけたら、Developer Portal での設定を示すエラーにする(CLAUDE.md 4-6 の bot の準備の2番)。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from kijun.deps import import_optional
from kijun.discord.base import DiscordError, Reaction, Reply

MESSAGE_CONTENT_HINT = (
    "返信の本文が空で返ってきました。Discord Developer Portal の Bot の設定で "
    "Message Content Intent を有効にしてください(無効のままだと返信本文を読めません)。"
)


class DiscordPyClient:
    def __init__(self, token: str) -> None:
        self._discord = import_optional("discord", "discord", "Discord への接続")
        self._token = token

    def _run(self, make: Callable[[object], Awaitable]):
        """ログインして make(client) を実行し、閉じる。通信の失敗は DiscordError にする。"""
        discord = self._discord

        async def main():
            intents = discord.Intents.default()
            intents.message_content = True
            client = discord.Client(intents=intents)
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
                f"チャンネルの閲覧・投稿・履歴の読み取りの権限を付けてください: {e}"
            ) from e
        except discord.HTTPException as e:
            raise DiscordError(f"Discord との通信に失敗しました: {e}") from e

    def post(self, channel_id: str, content: str) -> str:
        async def make(client):
            channel = await client.fetch_channel(int(channel_id))
            message = await channel.send(content)
            return str(message.id)

        return self._run(make)

    def fetch_reactions(
        self, channel_id: str, message_ids: list[str]
    ) -> dict[str, list[Reaction]]:
        discord = self._discord

        async def make(client):
            channel = await client.fetch_channel(int(channel_id))
            result: dict[str, list[Reaction]] = {}
            for message_id in message_ids:
                try:
                    message = await channel.fetch_message(int(message_id))
                except discord.NotFound:
                    continue  # 削除されたメッセージ。呼び出し側が未処理のまま残す
                result[message_id] = [Reaction(str(r.emoji), r.count) for r in message.reactions]
            return result

        return self._run(make)

    def fetch_thread_replies(
        self, channel_id: str, message_ids: list[str]
    ) -> dict[str, list[Reply]]:
        """返信は2通りの形で受ける: メッセージから作ったスレッドの中の発言と、
        チャンネルの中でそのメッセージを参照して返信したメッセージ。"""
        discord = self._discord

        async def make(client):
            channel = await client.fetch_channel(int(channel_id))
            guild = await client.fetch_guild(channel.guild.id)
            owner_id = guild.owner_id

            def to_reply(m) -> Reply:
                if not m.author.bot and not (m.content or m.attachments or m.embeds or m.stickers):
                    raise DiscordError(MESSAGE_CONTENT_HINT)
                return Reply(
                    author_is_owner=(m.author.id == owner_id and not m.author.bot),
                    content=m.content,
                    created_at=m.created_at.astimezone().replace(tzinfo=None),
                )

            result: dict[str, list[Reply]] = {}
            for message_id in message_ids:
                try:
                    await channel.fetch_message(int(message_id))
                except discord.NotFound:
                    continue
                result[message_id] = []

            # チャンネルの中の参照返信は、履歴を1回だけ読んで振り分ける(投稿ごとに読むと遅い)
            if result:
                first = min(int(i) for i in result)
                async for m in channel.history(
                    after=discord.Object(id=first), limit=None, oldest_first=True
                ):
                    ref = m.reference.message_id if m.reference else None
                    if ref is not None and str(ref) in result:
                        result[str(ref)].append(to_reply(m))

            # メッセージから作ったスレッドの ID は、元のメッセージの ID と同じ
            for message_id in list(result):
                try:
                    thread = await client.fetch_channel(int(message_id))
                except (discord.NotFound, discord.Forbidden, discord.InvalidData):
                    continue
                async for m in thread.history(limit=None, oldest_first=True):
                    result[message_id].append(to_reply(m))
                result[message_id].sort(key=lambda r: r.created_at)
            return result

        return self._run(make)
