"""Ollama による抽出。

Ollama のネイティブ API（POST /api/chat）に httpx で直接投げる。OpenAI 互換の /v1/chat/completions を
使わない理由は、num_ctx と keep_alive をリクエストごとに渡せないためである。num_ctx を送らないと
Ollama の既定の文脈長になり、入力が長いと**エラーを出さずに先頭が切り捨てられる**。

2026-11-05 に、背面を Strata（Qwen3.8-Flash-Next）から Ollama に移した。git 履歴の 7350993 にも
Ollama の実装があるが、num_ctx を送らない・transport.py と repair.py を通さない・usage を記録しない
という欠陥があったので、そのまま戻さず書き直した。

確認できていないこと（実機の Ollama で `uv run kijun doctor` が測る）。
- 設定のモデルが存在するか（`ollama list`）
- format にスキーマを渡したとき、出力が適合するか（設定の enforces_json_schema）
- think と format を同時に送ったときの挙動（版とモデルによって、推論にも制約が掛かる報告がある）
"""

from __future__ import annotations

import sys

from kijun import ollama_api
from kijun.config import Config
from kijun.extract.base import ExtractionError, RepairEvent
from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractionResult
from kijun.extract.prompt import build_messages
from kijun.extract.repair import UnrepairableJsonError, parse_with_repair
from kijun.transport import HttpTransport, HttpxTransport, TransportError

# 出力が読めなかったときに、同じチャンクを投げ直す回数（最初の1回に加えて）
RETRY_COUNT = 1


def build_payload(cfg: Config, chunk: Chunk) -> dict:
    """1チャンク分のリクエストの本文を作る。プロンプトの文面は prompt.py のものを使う。

    モジュールの関数にしてあるのは、doctor.py と scripts/measure_extraction.py が
    「抽出が実際に送るのと同じ形」を組み立てるのに使うためである。測るものと送るものを
    食い違わせないようにする。
    """
    system, user = build_messages(chunk, cfg)
    return ollama_api.build_chat_payload(
        cfg.extract.ollama, system, user, ExtractionResult.model_json_schema()
    )


class OllamaExtractor:
    def __init__(self, cfg: Config, transport: HttpTransport | None = None) -> None:
        self._cfg = cfg
        self._ollama = cfg.extract.ollama
        # run_label が入っていれば model_name に付く。設定を変えた2回の抽出を DB に並べて残すため。
        self.model_name = cfg.extract.model_name()
        self.enforces_json_schema = self._ollama.enforces_json_schema
        # 入力に使ってよいトークン数。run.py が、送る前に概算でチャンクの大きさを警告するのに使う。
        self.max_input_tokens: int | None = ollama_api.input_limit(self._ollama)
        self.repair_events: list[RepairEvent] = []
        self.last_usage: dict | None = None
        self._transport = transport or HttpxTransport()

    @property
    def chat_url(self) -> str:
        return ollama_api.url_of(self._ollama, ollama_api.CHAT_PATH)

    def build_payload(self, chunk: Chunk) -> dict:
        return build_payload(self._cfg, chunk)

    def _post(self, payload: dict):
        try:
            response = self._transport.post(self.chat_url, payload, self._ollama.timeout_sec)
        except TransportError as e:
            raise ExtractionError(
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
                hint = " このモデルは think に対応していない可能性があります。extract.ollama.think を書かない（未指定）にしてください。"
            raise ExtractionError(
                f"Ollama が HTTP {response.status_code} を返しました（{self.chat_url}）: {head}{hint}"
            )
        return response

    def _read(self, response) -> ollama_api.ChatReply:
        try:
            return ollama_api.parse_chat_reply(response)
        except ollama_api.OllamaError as e:
            raise ExtractionError(str(e)) from e

    def _report_usage(self, reply: ollama_api.ChatReply) -> None:
        """Ollama が返したトークン数と速度を標準エラーに出す。"""
        self.last_usage = reply.usage
        if reply.prompt_eval_count is None and reply.eval_count is None:
            print(
                "注意: Ollama が prompt_eval_count と eval_count を返さなかったため、"
                "トークン数は記録できません",
                file=sys.stderr,
            )
            return
        speed = reply.tokens_per_sec
        thinking = f"、推論 {reply.thinking_chars} 文字" if reply.thinking_chars else ""
        print(
            f"Ollama: 入力 {reply.prompt_eval_count} / 出力 {reply.eval_count} トークン"
            f"（{reply.elapsed_sec:.1f} 秒"
            + (f"、{speed:.1f} tokens/s" if speed else "")
            + f"{thinking}）",
            file=sys.stderr,
        )

    def _report_truncation(self, reply: ollama_api.ChatReply, chunk: Chunk) -> bool:
        """done_reason が length なら標準エラーに出し、True を返す。

        例外にはしない。途中まで読める出力は repair.py が最後の完全な要素まで切り詰める。
        戻り値は extract が再試行の可否を決めるのに使う。
        """
        if reply.done_reason != "length":
            return False
        print(
            f"警告: 出力が num_predict に達して切れた（発話 {chunk.seq_from}〜{chunk.seq_to}）。"
            "`extract.ollama.num_predict` を上げるか `chunk_minutes` を下げてください",
            file=sys.stderr,
        )
        return True

    def _check_input(self, reply: ollama_api.ChatReply, chunk: Chunk) -> None:
        """入力が切り捨てられた疑いがあれば ExtractionError。投げ直しても同じなので再試行しない。"""
        if reply.prompt_eval_count is None:
            print(
                "注意: Ollama が prompt_eval_count を返さなかったため、入力が切り捨てられて"
                "いないかを確認できません",
                file=sys.stderr,
            )
            return
        if ollama_api.input_was_truncated(self._ollama, reply):
            raise ExtractionError(
                f"{ollama_api.truncation_message(self._ollama, reply)}"
                f"（発話 {chunk.seq_from}〜{chunk.seq_to}）"
            )

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """チャンクを抽出する。出力が読めなければ、同じチャンクをもう1回だけ投げ直す。

        スキーマへの適合を保証すると宣言された設定でも、必ず repair.py を通してから検証する
        （保証が実際に効いていれば、修復は何もしないので結果は変わらない）。
        """
        payload = self.build_payload(chunk)
        self.last_usage = None
        last_error: Exception | None = None
        for attempt in range(1, 2 + RETRY_COUNT):
            reply = self._read(self._post(payload))
            self._report_usage(reply)
            self._check_input(reply, chunk)
            truncated = self._report_truncation(reply, chunk)
            try:
                parsed = parse_with_repair(reply.content)
            except UnrepairableJsonError as e:
                last_error = e
                print(
                    f"警告: Ollama の出力を JSON として読めませんでした（{attempt} 回目）"
                    f"（発話 {chunk.seq_from}〜{chunk.seq_to}）: {e}",
                    file=sys.stderr,
                )
                if truncated:
                    # 出力が num_predict で切れて完全な要素が1件も読めないのは、設定の問題である。
                    # 投げ直さないのは、temperature 0 でも出力が決定的とは限らず、
                    # 投げ直しに最大で num_predict 分の生成時間がかかるのに、同じ設定では
                    # 同じように切れる可能性が高いからである。設定を直す方が速く、確実である。
                    raise ExtractionError(
                        f"Ollama の出力が num_predict（{self._ollama.num_predict}）に達し、"
                        f"完全な要素が1件も読めませんでした（発話 {chunk.seq_from}〜{chunk.seq_to}）。"
                        "think を有効にしたモデルでは、推論も num_predict に数えられます。"
                        "`extract.ollama.num_predict` を上げるか、`extract.chunk_minutes` を下げるか、"
                        "`extract.ollama.think = false` にしてください。"
                        "投げ直しは最大で num_predict 分の生成時間を使うため、していません"
                    ) from e
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
            f"Ollama の出力が {1 + RETRY_COUNT} 回とも読めませんでした: {last_error}"
        )
