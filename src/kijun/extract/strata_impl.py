"""Strata（ローカルの MoE 推論エンジン）による抽出。

Strata は OpenAI 互換と Anthropic 互換の両方のエンドポイントを公開するが、ここでは
OpenAI 互換に httpx で直接投げる。Anthropic 互換側を anthropic SDK で叩かない理由は、
SDK の構造化出力（messages.parse）や betas、fallbacks はローカルサーバーが実装している
保証がなく、失敗したときに原因が SDK 側かサーバー側か切り分けられなくなるため。

確認できていないこと（`uv run kijun doctor` で実測する）。
- OpenAI 互換のパス（設定の chat_path で変えられるようにしてある）
- JSON スキーマへの適合を保証するか（設定の enforces_json_schema。既定は保証しない側）
- 1リクエストに入る入力トークンの上限（設定の max_input_tokens。0 は未確認）
"""

from __future__ import annotations

import sys

from kijun.config import Config
from kijun.extract.base import ExtractionError, RepairEvent
from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractionResult
from kijun.extract.prompt import build_messages
from kijun.extract.repair import UnrepairableJsonError, parse_with_repair
from kijun.transport import HttpTransport, HttpxTransport, TransportError

# 出力が読めなかったときに、同じチャンクを投げ直す回数（最初の1回に加えて）
RETRY_COUNT = 1


def build_payload(cfg: Config, chunk: Chunk, enforces_json_schema: bool) -> dict:
    """1チャンク分のリクエストの本文を作る。プロンプトの文面は prompt.py のものを使う。

    モジュールの関数にしてあるのは、doctor.py が「抽出が実際に送るのと同じ形」を
    組み立てるのに使うためである。doctor が測るものと extract が送るものを
    食い違わせないようにする。
    """
    system, user = build_messages(chunk, cfg)
    payload: dict = {
        "model": cfg.extract.strata.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "stream": False,
    }
    # 0 は「上限なし」。そのときは max_tokens を送らない。上限が無いと、推論トークンを含む生成が
    # n_ctx（実機で 204,800）まで走り得る（実機で 2,383 秒経っても終わらなかった事例がある）。
    # 既定値 20000 は実測（出力 13,050 トークンで stop）に余裕を持たせた値（設定 extract.strata.max_tokens）。
    if cfg.extract.strata.max_tokens > 0:
        payload["max_tokens"] = cfg.extract.strata.max_tokens
    if enforces_json_schema:
        # 対応していないサーバーが 400 を返す可能性があるので、保証すると宣言されたときだけ付ける。
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "schema": ExtractionResult.model_json_schema(),
                "strict": True,
            },
        }
    return payload


class StrataExtractor:
    def __init__(self, cfg: Config, transport: HttpTransport | None = None) -> None:
        self._cfg = cfg
        self._strata = cfg.extract.strata
        self.model_name = self._strata.model
        self.enforces_json_schema = self._strata.enforces_json_schema
        # 設定の 0 は「未確認」を表す（TOML に null が無いため）。属性では None にする。
        self.max_input_tokens = self._strata.max_input_tokens or None
        self.repair_events: list[RepairEvent] = []
        self._transport = transport or HttpxTransport()

    @property
    def chat_url(self) -> str:
        return f"{self._strata.base_url}{self._strata.chat_path}"

    def build_payload(self, chunk: Chunk) -> dict:
        return build_payload(self._cfg, chunk, self.enforces_json_schema)

    def _post(self, payload: dict):
        try:
            response = self._transport.post(self.chat_url, payload, self._strata.timeout_sec)
        except TransportError as e:
            raise ExtractionError(
                f"Strata に要求を送れませんでした（{self.chat_url}）: {e}\n"
                "Strata が起動しているか、設定の extract.strata.base_url と chat_path が正しいかを "
                "`uv run kijun doctor` で確認してください。"
            ) from e
        if not response.ok:
            head = response.text[:300].replace("\n", " ")
            raise ExtractionError(
                f"Strata が HTTP {response.status_code} を返しました（{self.chat_url}）: {head}"
            )
        return response

    def _message_content(self, response) -> str:
        """OpenAI 互換の応答から本文を取り出す。形が違えば ExtractionError を出す。"""
        try:
            body = response.json()
        except TransportError as e:
            raise ExtractionError(str(e)) from e
        try:
            # 推論の途中で max_tokens に達すると content が null や空になり得る。その場合は空文字にして
            # 後段の修復に任せる（読めなければ失敗として記録される）。
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            keys = list(body) if isinstance(body, dict) else type(body).__name__
            raise ExtractionError(
                "Strata の応答が OpenAI 互換の形（choices[0].message.content）ではありません"
                f"（最上位のキー: {keys}）。設定の extract.strata.chat_path が正しいかを "
                "`uv run kijun doctor` で確認してください。"
            ) from e

    def _report_truncation(self, response, chunk: Chunk) -> None:
        """finish_reason が length なら、出力が max_tokens に達して切れたことを標準エラーに出す。

        例外にはしない。途中まで読める出力は repair.py が最後の完全な要素まで切り詰める。
        """
        try:
            reason = response.json()["choices"][0].get("finish_reason")
        except (TransportError, KeyError, IndexError, TypeError, AttributeError):
            return
        if reason == "length":
            print(
                f"警告: 出力が max_tokens に達して切れた（発話 {chunk.seq_from}〜{chunk.seq_to}）。"
                "`extract.strata.max_tokens` を上げるか `chunk_minutes` を下げてください",
                file=sys.stderr,
            )

    def _report_usage(self, response) -> None:
        """サーバーが返したトークン数を標準エラーに出す。返さない場合はそのことを出す。"""
        try:
            usage = response.json().get("usage")
        except TransportError:
            return
        if not usage:
            print(
                "注意: サーバーが usage を返さないため、入力・出力のトークン数は記録できません"
                "（入力長の概算は kijun doctor で実測してください）",
                file=sys.stderr,
            )
            return
        print(
            f"Strata: 入力 {usage.get('prompt_tokens')} / 出力 {usage.get('completion_tokens')} トークン"
            f"（{response.elapsed_sec:.1f} 秒）",
            file=sys.stderr,
        )

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """チャンクを抽出する。出力が読めなければ、同じチャンクをもう1回だけ投げ直す。

        スキーマへの適合を保証すると宣言された背面でも、必ず repair.py を通してから検証する
        （保証が実際に効いていれば、修復は何もしないので結果は変わらない）。
        """
        payload = self.build_payload(chunk)
        last_error: Exception | None = None
        for attempt in range(1, 2 + RETRY_COUNT):
            response = self._post(payload)
            self._report_usage(response)
            self._report_truncation(response, chunk)
            text = self._message_content(response)
            try:
                parsed = parse_with_repair(text)
            except UnrepairableJsonError as e:
                last_error = e
                print(
                    f"警告: Strata の出力を JSON として読めませんでした（{attempt} 回目）"
                    f"（発話 {chunk.seq_from}〜{chunk.seq_to}）: {e}",
                    file=sys.stderr,
                )
                continue
            if parsed.repair.actions or parsed.repair.truncated_items:
                self.repair_events.append(
                    RepairEvent(
                        chunk_index=chunk.index,
                        seq_from=chunk.seq_from,
                        seq_to=chunk.seq_to,
                        actions=list(parsed.repair.actions),
                        truncated_items=parsed.repair.truncated_items,
                        attempt=attempt,
                    )
                )
                print(
                    f"注意: 出力を修復しました（発話 {chunk.seq_from}〜{chunk.seq_to}）: "
                    f"{', '.join(parsed.repair.descriptions)}"
                    + (
                        f"。末尾が切れていたため {parsed.repair.truncated_items} 件を捨てました"
                        if parsed.repair.truncated_items
                        else ""
                    ),
                    file=sys.stderr,
                )
            return parsed.result
        raise ExtractionError(
            f"Strata の出力が {1 + RETRY_COUNT} 回とも読めませんでした: {last_error}"
        )
