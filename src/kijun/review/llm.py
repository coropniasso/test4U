"""抽出以外の LLM の用途(重要度の判定、返信の振り分け、話者の対応付け)が共有する呼び出し(仕様 1-3)。

抽出と同じ Ollama に、同じ `transport.py` の境界を通して投げる。`Extractor` は再利用しない
(入出力のスキーマが違うため)。再利用するのは次の3つ。
- HttpTransport(偽の応答でテストできる)
- config.extract.ollama の設定(接続先、モデル、num_ctx、JSON スキーマの強制の有無、時間切れ)
- repair.py(出力は必ず修復を通してから検証する。原則2-3)

呼び出し側は `JsonChat`(Protocol)だけを知る。Ollama への依存は `OllamaJsonChat` に閉じる。
"""

from __future__ import annotations

import sys
from typing import Protocol, TypeVar

from pydantic import BaseModel

from kijun import ollama_api
from kijun.config import Config
from kijun.extract.repair import UnrepairableJsonError, parse_model_with_repair
from kijun.transport import HttpTransport, HttpxTransport, TransportError

ModelT = TypeVar("ModelT", bound=BaseModel)

# 出力が読めなかったときに、同じ問い合わせを投げ直す回数(最初の1回に加えて)。抽出と同じ。
RETRY_COUNT = 1


class ReviewLlmError(Exception):
    """LLM への問い合わせが失敗した(接続できない、HTTP エラー、2回とも読めない出力)ときの例外。

    呼び出し側は、この例外を捕まえて処理を止めずに続ける。失敗したものは推測で埋めず、
    未処理(重要度は NULL、返信は再試行待ち)のまま残す。
    """


class JsonChat(Protocol):
    def ask(self, system: str, user: str, model_cls: type[ModelT], schema_name: str) -> ModelT:
        """system / user のメッセージを投げ、JSON を model_cls として受け取る。失敗したら ReviewLlmError。"""
        ...


class OllamaJsonChat:
    """Ollama に system / user のメッセージを投げ、JSON を model_cls として受け取る。"""

    def __init__(self, cfg: Config, transport: HttpTransport | None = None) -> None:
        self._ollama = cfg.extract.ollama
        self._transport = transport or HttpxTransport()

    @property
    def chat_url(self) -> str:
        return ollama_api.url_of(self._ollama, ollama_api.CHAT_PATH)

    def build_payload(self, system: str, user: str, model_cls: type[BaseModel], schema_name: str) -> dict:
        # schema_name は Ollama の format には使わない(スキーマそのものを渡す)。エラー表示と警告の識別に使う。
        return ollama_api.build_chat_payload(self._ollama, system, user, model_cls.model_json_schema())

    def _reply(self, payload: dict) -> ollama_api.ChatReply:
        try:
            response = self._transport.post(self.chat_url, payload, self._ollama.timeout_sec)
        except TransportError as e:
            raise ReviewLlmError(
                f"Ollama に要求を送れませんでした({self.chat_url}): {e}\n"
                + ollama_api.connection_hint(self._ollama)
            ) from e
        if not response.ok:
            head = response.text[:300].replace("\n", " ")
            raise ReviewLlmError(f"Ollama が HTTP {response.status_code} を返しました: {head}")
        try:
            return ollama_api.parse_chat_reply(response)
        except ollama_api.OllamaError as e:
            raise ReviewLlmError(str(e)) from e

    def ask(self, system: str, user: str, model_cls: type[ModelT], schema_name: str) -> ModelT:
        """問い合わせて model_cls として受け取る。読めなければ1回だけ投げ直し、それでも読めなければ ReviewLlmError。

        入力が文脈長を超えて切り捨てられた疑いがあれば、投げ直さずに ReviewLlmError にする
        (投げ直しても同じ入力なので、同じように切り捨てられる)。
        """
        payload = self.build_payload(system, user, model_cls, schema_name)
        last_error: Exception | None = None
        for attempt in range(1, 2 + RETRY_COUNT):
            reply = self._reply(payload)
            if ollama_api.input_was_truncated(self._ollama, reply):
                raise ReviewLlmError(f"{schema_name}: {ollama_api.truncation_message(self._ollama, reply)}")
            try:
                result, outcome = parse_model_with_repair(reply.content, model_cls)
            except UnrepairableJsonError as e:
                last_error = e
                print(
                    f"警告: Ollama の出力を JSON として読めませんでした({attempt} 回目、{schema_name}): {e}",
                    file=sys.stderr,
                )
                continue
            if outcome.actions or outcome.truncated_items:
                print(
                    f"注意: 出力を修復しました({schema_name}): {', '.join(outcome.descriptions)}",
                    file=sys.stderr,
                )
            return result
        raise ReviewLlmError(f"Ollama の出力が {1 + RETRY_COUNT} 回とも読めませんでした: {last_error}")
