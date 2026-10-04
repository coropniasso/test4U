"""Ollama（Qwen など）による抽出（計画書 7-4）。

Ollama の常駐プロセスとモデルファイルが必要。動作確認は Windows の実機で行う。
`ollama` パッケージは使わず、httpx で /api/chat を呼ぶ。httpx は
`uv sync --extra extract-ollama` で入る。
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from kijun.config import Config
from kijun.deps import import_optional
from kijun.extract.base import ExtractionError
from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractionResult
from kijun.extract.prompt import build_messages

# JSON が壊れていたときに、同じチャンクを投げ直す回数（最初の1回に加えて）
RETRY_COUNT = 1


class OllamaExtractor:
    def __init__(self, cfg: Config) -> None:
        self._httpx = import_optional("httpx", "extract-ollama", "Ollama への通信")
        self._cfg = cfg
        self._ollama = cfg.extract.ollama
        self.model_name = self._ollama.model

    def _build_payload(self, chunk: Chunk) -> dict:
        system, user = build_messages(chunk, self._cfg)
        return {
            "model": self._ollama.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # CLAUDE.md 4-4: format パラメータに JSON スキーマを渡し、出力をスキーマに従った JSON に制限する
            "format": ExtractionResult.model_json_schema(),
            "stream": False,
            # 0 で、処理後にモデルを VRAM からアンロードさせる（計画書 第1節）
            "keep_alive": self._ollama.keep_alive,
            "options": {"temperature": 0},
        }

    def _post(self, payload: dict) -> str:
        try:
            response = self._httpx.post(
                f"{self._ollama.base_url}/api/chat",
                json=payload,
                timeout=self._ollama.timeout_sec,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:  # 通信エラー、HTTP エラー、応答の形が違う場合をまとめて扱う
            raise ExtractionError(f"Ollama への要求に失敗しました: {e}") from e

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """チャンクを抽出する。出力が壊れていたら、同じチャンクをもう1回だけ投げ直す。"""
        payload = self._build_payload(chunk)
        last_error: Exception | None = None
        for attempt in range(1 + RETRY_COUNT):
            content = self._post(payload)
            try:
                return ExtractionResult.model_validate_json(content)
            except ValidationError as e:
                last_error = e
                print(
                    f"警告: Ollama の出力が JSON スキーマに合いません（{attempt + 1} 回目）"
                    f"（発話 {chunk.seq_from}〜{chunk.seq_to}）",
                    file=sys.stderr,
                )
        raise ExtractionError(f"Ollama の出力が {1 + RETRY_COUNT} 回とも不正でした: {last_error}")
