"""抽出以外の LLM の用途(重要度の判定、返信の振り分け、話者の対応付け)が共有する呼び出し(仕様 1-3)。

抽出と同じ Strata のサーバーに、同じ `transport.py` の境界を通して投げる。
`Extractor` は再利用しない(入出力のスキーマが違うため)。再利用するのは次の3つ。
- HttpTransport(偽の応答でテストできる)
- config.extract.strata の設定(接続先、JSON スキーマの強制の有無、時間切れ)
- repair.py(出力は必ず修復を通してから検証する。原則2-3)
"""

from __future__ import annotations

import sys
from typing import TypeVar

from pydantic import BaseModel

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


class StrataJsonChat:
    """Strata に system / user のメッセージを投げ、JSON を model_cls として受け取る。"""

    def __init__(self, cfg: Config, transport: HttpTransport | None = None) -> None:
        self._strata = cfg.extract.strata
        self._transport = transport or HttpxTransport()

    @property
    def chat_url(self) -> str:
        return f"{self._strata.base_url}{self._strata.chat_path}"

    def build_payload(self, system: str, user: str, model_cls: type[BaseModel], schema_name: str) -> dict:
        payload: dict = {
            "model": self._strata.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "stream": False,
        }
        if self._strata.enforces_json_schema:
            # 対応していないサーバーが 400 を返す可能性があるので、保証すると設定されたときだけ付ける。
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": model_cls.model_json_schema(),
                    "strict": True,
                },
            }
        return payload

    def _content(self, payload: dict) -> str:
        try:
            response = self._transport.post(self.chat_url, payload, self._strata.timeout_sec)
        except TransportError as e:
            raise ReviewLlmError(
                f"Strata に要求を送れませんでした({self.chat_url}): {e}"
            ) from e
        if not response.ok:
            head = response.text[:300].replace("\n", " ")
            raise ReviewLlmError(f"Strata が HTTP {response.status_code} を返しました: {head}")
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (TransportError, KeyError, IndexError, TypeError) as e:
            raise ReviewLlmError(
                "Strata の応答が OpenAI 互換の形(choices[0].message.content)ではありません。"
                "`uv run kijun doctor` でパスを確認してください。"
            ) from e

    def ask(self, system: str, user: str, model_cls: type[ModelT], schema_name: str) -> ModelT:
        """問い合わせて model_cls として受け取る。読めなければ1回だけ投げ直し、それでも読めなければ ReviewLlmError。"""
        payload = self.build_payload(system, user, model_cls, schema_name)
        last_error: Exception | None = None
        for attempt in range(1, 2 + RETRY_COUNT):
            text = self._content(payload)
            try:
                result, outcome = parse_model_with_repair(text, model_cls)
            except UnrepairableJsonError as e:
                last_error = e
                print(
                    f"警告: Strata の出力を JSON として読めませんでした({attempt} 回目、{schema_name}): {e}",
                    file=sys.stderr,
                )
                continue
            if outcome.actions or outcome.truncated_items:
                print(
                    f"注意: 出力を修復しました({schema_name}): {', '.join(outcome.descriptions)}",
                    file=sys.stderr,
                )
            return result
        raise ReviewLlmError(f"Strata の出力が {1 + RETRY_COUNT} 回とも読めませんでした: {last_error}")
