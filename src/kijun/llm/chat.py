"""Ollama に system / user のメッセージを投げ、JSON を pydantic のモデルとして受け取る呼び出し。

議事録の生成（minutes/）が使う。再利用するのは次の3つ。
- HttpTransport（偽の応答でテストできる。transport.py）
- 設定 [llm]（接続先、モデル、num_ctx、JSON スキーマの強制の有無、時間切れ）
- repair.py（出力は必ず修復を通してから検証する）

呼び出し側は `JsonChat`（Protocol）だけを知る。Ollama への依存は `OllamaJsonChat` に閉じる。
"""

from __future__ import annotations

import sys
from typing import Protocol, TypeVar

from pydantic import BaseModel

from kijun import ollama_api
from kijun.config import OllamaConfig
from kijun.llm.repair import UnrepairableJsonError, parse_model_with_repair
from kijun.transport import HttpTransport, HttpxTransport, TransportError

ModelT = TypeVar("ModelT", bound=BaseModel)

# 出力が読めなかったときに、同じ問い合わせを投げ直す回数（最初の1回に加えて）。
RETRY_COUNT = 1


class LlmError(Exception):
    """LLM への問い合わせが失敗した（接続できない、HTTP エラー、入力の切り捨て、2回とも読めない出力）ときの例外。"""


class JsonChat(Protocol):
    # 最後に成功した問い合わせの応答。表示（トークン数、速度）に使う。無ければ None。
    last_reply: ollama_api.ChatReply | None

    def ask(self, system: str, user: str, model_cls: type[ModelT], name: str) -> ModelT:
        """system / user のメッセージを投げ、JSON を model_cls として受け取る。失敗したら LlmError。"""
        ...


class OllamaJsonChat:
    """Ollama に system / user のメッセージを投げ、JSON を model_cls として受け取る。"""

    def __init__(self, cfg: OllamaConfig, transport: HttpTransport | None = None) -> None:
        self._ollama = cfg
        self._transport = transport or HttpxTransport()
        self.last_reply: ollama_api.ChatReply | None = None

    @property
    def chat_url(self) -> str:
        return ollama_api.url_of(self._ollama, ollama_api.CHAT_PATH)

    def build_payload(self, system: str, user: str, model_cls: type[BaseModel]) -> dict:
        return ollama_api.build_chat_payload(self._ollama, system, user, model_cls.model_json_schema())

    def _reply(self, payload: dict) -> ollama_api.ChatReply:
        try:
            response = self._transport.post(self.chat_url, payload, self._ollama.timeout_sec)
        except TransportError as e:
            raise LlmError(
                f"Ollama に要求を送れませんでした（{self.chat_url}）: {e}\n"
                + ollama_api.connection_hint(self._ollama)
            ) from e
        if not response.ok:
            head = response.text[:300].replace("\n", " ")
            hint = ""
            if response.status_code == 404:
                hint = f" モデルが無い可能性があります。`ollama pull {self._ollama.model}` を実行してください。"
            elif (
                response.status_code == 400
                and self._ollama.think is not None
                and "think" in response.text.lower()
            ):
                hint = " このモデルは think に対応していない可能性があります。llm.think を書かない（未指定）にしてください。"
            raise LlmError(f"Ollama が HTTP {response.status_code} を返しました: {head}{hint}")
        try:
            return ollama_api.parse_chat_reply(response)
        except ollama_api.OllamaError as e:
            raise LlmError(str(e)) from e

    def ask(self, system: str, user: str, model_cls: type[ModelT], name: str) -> ModelT:
        """問い合わせて model_cls として受け取る。読めなければ1回だけ投げ直し、それでも読めなければ LlmError。

        次の場合は投げ直さずに LlmError にする（投げ直しても同じ結果になるため）。
        - 入力が文脈長を超えて切り捨てられた疑いがある（prompt_eval_count が num_ctx - num_predict に達した）
        - 出力が num_predict に達して切れた（done_reason = length）。投げ直しは最大で num_predict 分の
          生成時間を使い、同じ設定では同じように切れる可能性が高い
        """
        payload = self.build_payload(system, user, model_cls)
        last_error: Exception | None = None
        for attempt in range(1, 2 + RETRY_COUNT):
            reply = self._reply(payload)
            self.last_reply = reply
            if ollama_api.input_was_truncated(self._ollama, reply):
                raise LlmError(f"{name}: {ollama_api.truncation_message(self._ollama, reply)}")
            if reply.prompt_eval_count is None:
                print(
                    "注意: Ollama が prompt_eval_count を返さなかったため、入力が切り捨てられて"
                    "いないかを確認できません",
                    file=sys.stderr,
                )
            if reply.done_reason == "length":
                raise LlmError(
                    f"{name}: 出力が num_predict（{self._ollama.num_predict}）に達して切れました。"
                    "think を有効にしたモデルでは、推論も num_predict に数えられます。"
                    "`llm.num_predict` を上げるか、`minutes.chunk_minutes` を下げるか、"
                    "`llm.think = false` にしてください。途中で切れた JSON は、後ろの節が欠けるので修復しません"
                )
            try:
                result, outcome = parse_model_with_repair(reply.content, model_cls)
            except UnrepairableJsonError as e:
                last_error = e
                print(
                    f"警告: Ollama の出力を JSON として読めませんでした（{attempt} 回目、{name}）: {e}",
                    file=sys.stderr,
                )
                continue
            if outcome.actions:
                print(f"注意: 出力を修復しました（{name}）: {', '.join(outcome.descriptions)}", file=sys.stderr)
            return result
        raise LlmError(f"{name}: Ollama の出力が {1 + RETRY_COUNT} 回とも読めませんでした: {last_error}")
