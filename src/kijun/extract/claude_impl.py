"""Claude API による抽出（計画書 7-5）。

API キー（環境変数 ANTHROPIC_API_KEY）と課金が必要。動作確認は実機で行う。
anthropic は `uv sync --extra extract-claude` で入る。
"""

from __future__ import annotations

import sys

from kijun.config import Config
from kijun.deps import import_optional
from kijun.extract.base import ExtractionError
from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractionResult
from kijun.extract.prompt import build_messages

# 安全性の分類器が拒否したときに、サーバー側で別のモデルに切り替えさせる（HTTP 200 で stop_reason が refusal）。
# 会議の録音には感情的なやり取りが含まれ得るため有効にする。
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeExtractor:
    def __init__(self, cfg: Config) -> None:
        anthropic = import_optional("anthropic", "extract-claude", "Claude API による抽出")
        self._cfg = cfg
        self._claude = cfg.extract.claude
        self.model_name = self._claude.model
        # API キーは環境変数 ANTHROPIC_API_KEY から SDK が読む。設定ファイルには書かない。
        self._client = anthropic.Anthropic(timeout=float(self._claude.timeout_sec))

    def extract(self, chunk: Chunk) -> ExtractionResult:
        system, user = build_messages(chunk, self._cfg)
        try:
            response = self._client.beta.messages.parse(
                model=self._claude.model,
                max_tokens=self._claude.max_tokens,
                # システムプロンプトはチャンク間で変わらないのでキャッシュさせる。
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=ExtractionResult,
                # Sonnet 5.5 の effort の既定は high。抽出は分類に近いので既定は low にしてある。
                # thinking={"type": "disabled"} は Sonnet 5.5 では 400 になるので渡さない。
                output_config={"effort": self._claude.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except Exception as e:
            raise ExtractionError(f"Claude API への要求に失敗しました: {e}") from e

        # content に触る前に stop_reason を確認する。
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise ExtractionError(f"Claude API が応答を拒否しました（category={category}）")
        if response.stop_reason == "max_tokens":
            raise ExtractionError(
                f"出力が max_tokens（{self._claude.max_tokens}）に達して途中で切れました。"
                "設定の extract.claude.max_tokens を上げるか、chunk_minutes を下げてください。"
            )

        usage = response.usage
        print(
            f"Claude: 入力 {usage.input_tokens} / 出力 {usage.output_tokens} トークン、"
            f"cache_read_input_tokens={getattr(usage, 'cache_read_input_tokens', None)}"
            "（0 のままならプロンプトキャッシュが効いていない）",
            file=sys.stderr,
        )
        parsed = response.parsed_output
        if parsed is None:
            raise ExtractionError("Claude API の応答を ExtractionResult に変換できませんでした")
        return parsed
