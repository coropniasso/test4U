"""Ollama のネイティブ API（POST /api/chat）のリクエストの組み立てと、応答の読み取り。

抽出（extract/ollama_impl.py）、抽出以外の LLM の用途（review/llm.py）、doctor.py、
scripts/measure_extraction.py が共有する。リクエストの形をここ1か所に置くのは、
doctor と測定スクリプトが測るものと、抽出が実際に送るものを食い違わせないためである。

OpenAI 互換の /v1/chat/completions を使わない理由は、num_ctx と keep_alive をリクエストごとに
渡せないためである。num_ctx を送らないと Ollama の既定の文脈長になり、入力が長いと
**エラーを出さずに先頭から切り捨てられる**。切り捨てられた発話は抽出結果から静かに消える。
その検出も、このモジュールが持つ（input_was_truncated）。
"""

from __future__ import annotations

from dataclasses import dataclass

from kijun.config import OllamaConfig
from kijun.transport import HttpResponse, HttpTransport, TransportError

CHAT_PATH = "/api/chat"
TAGS_PATH = "/api/tags"  # 取得済みのモデルの一覧
PS_PATH = "/api/ps"  # いま読み込まれているモデルと、VRAM に載っている量
GENERATE_PATH = "/api/generate"  # モデルを下ろす要求（keep_alive: 0）に使う

NS_PER_SEC = 1_000_000_000


class OllamaError(Exception):
    """応答が Ollama の /api/chat の形でなかった、などのとき。呼び出し側が自分の例外に包み直す。"""


def url_of(cfg: OllamaConfig, path: str) -> str:
    return f"{cfg.base_url.rstrip('/')}{path}"


def build_chat_payload(
    cfg: OllamaConfig,
    system: str,
    user: str,
    schema: dict | None,
    keep_alive: str | None = None,
) -> dict:
    """1回分の /api/chat のリクエストの本文を作る。

    - schema を渡し、かつ cfg.enforces_json_schema が真のときだけ format にスキーマを付ける。
    - cfg.think が None なら think を送らない（モデルの既定に任せる）。
    - keep_alive を渡すと、設定の値の代わりにそれを送る（doctor が検査の間だけ "2m" にするため）。
    """
    payload: dict = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "keep_alive": cfg.keep_alive if keep_alive is None else keep_alive,
        "options": {"temperature": 0, "num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict},
    }
    if schema is not None and cfg.enforces_json_schema:
        payload["format"] = schema
    if cfg.think is not None:
        payload["think"] = cfg.think
    return payload


@dataclass(frozen=True)
class ChatReply:
    """/api/chat の応答のうち、この仕組みが使う項目。時間は秒に直してある。"""

    content: str
    # message.thinking の文字数。推論の本文そのものは保存しない。
    thinking_chars: int
    # "stop"（自然に終わった）または "length"（num_predict で切れた）など。
    done_reason: str | None
    # 入力のトークン数。返らないことがある（プロンプトのキャッシュが効いた場合など）。
    prompt_eval_count: int | None
    # 出力のトークン数（推論を含む）。
    eval_count: int | None
    total_sec: float | None
    eval_sec: float | None
    # モデルを VRAM に読み込むのにかかった時間と、入力の処理にかかった時間。
    load_sec: float | None
    prompt_eval_sec: float | None
    elapsed_sec: float

    @property
    def tokens_per_sec(self) -> float | None:
        """出力の生成速度。出力が無い、または時間が取れないときは None。"""
        if self.eval_count and self.eval_sec:
            return self.eval_count / self.eval_sec
        return None

    @property
    def usage(self) -> dict:
        """extract/run.py が表示に使う形（Extractor.last_usage）。"""
        return {"prompt_tokens": self.prompt_eval_count, "completion_tokens": self.eval_count}


def _count(body: dict, key: str) -> int | None:
    value = body.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _seconds(body: dict, key: str) -> float | None:
    value = body.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value / NS_PER_SEC
    return None


def parse_chat_reply(response: HttpResponse) -> ChatReply:
    """200 の応答を読む。Ollama の形（message.content）でなければ OllamaError。"""
    try:
        body = response.json()
    except TransportError as e:
        raise OllamaError(str(e)) from e
    message = body.get("message") if isinstance(body, dict) else None
    if not isinstance(message, dict):
        keys = list(body) if isinstance(body, dict) else type(body).__name__
        raise OllamaError(
            "応答が Ollama の /api/chat の形（message.content）ではありません"
            f"（最上位のキー: {keys}）。extract.ollama.base_url が Ollama を指しているかを "
            "`uv run kijun doctor` で確認してください。"
        )
    # 推論の途中で num_predict に達すると content が空になり得る。空文字にして後段の修復に任せる
    # （読めなければ失敗として扱われる）。
    content = message.get("content") or ""
    thinking = message.get("thinking") or ""
    done_reason = body.get("done_reason")
    return ChatReply(
        content=content if isinstance(content, str) else "",
        thinking_chars=len(thinking) if isinstance(thinking, str) else 0,
        done_reason=done_reason if isinstance(done_reason, str) else None,
        prompt_eval_count=_count(body, "prompt_eval_count"),
        eval_count=_count(body, "eval_count"),
        total_sec=_seconds(body, "total_duration"),
        eval_sec=_seconds(body, "eval_duration"),
        load_sec=_seconds(body, "load_duration"),
        prompt_eval_sec=_seconds(body, "prompt_eval_duration"),
        elapsed_sec=response.elapsed_sec,
    )


def input_limit(cfg: OllamaConfig) -> int:
    """入力に使ってよいトークン数の上限（これに達したら切り捨ての疑い）。num_ctx - num_predict。"""
    return cfg.num_ctx - cfg.num_predict


def input_was_truncated(cfg: OllamaConfig, reply: ChatReply) -> bool:
    """入力が文脈長を超えて切り捨てられた疑いがあるか。

    Ollama は入力が num_ctx を超えると、エラーを出さずに先頭を捨てる。捨てられた入力から抽出すると、
    捨てられた発話が静かに消える。出力の上限 num_predict の分を引いた値に達していたら、疑う。
    prompt_eval_count が返らなかったときは判定できないので False を返す（呼び出し側が注意を出す）。
    """
    return reply.prompt_eval_count is not None and reply.prompt_eval_count >= input_limit(cfg)


def truncation_message(cfg: OllamaConfig, reply: ChatReply) -> str:
    return (
        f"入力が {reply.prompt_eval_count} トークンで、上限（num_ctx {cfg.num_ctx} - num_predict "
        f"{cfg.num_predict} = {input_limit(cfg)}）に達しました。Ollama は入力が文脈長を超えると、"
        "エラーを出さずに先頭を捨てるため、捨てられた発話が結果から消えます。"
        "`extract.chunk_minutes` を下げるか、`extract.ollama.num_ctx` を上げてください"
    )


def connection_hint(cfg: OllamaConfig) -> str:
    return (
        f"Ollama が起動しているか（タスクトレイ。{cfg.base_url}）、`ollama list` に {cfg.model} があるか、"
        "`uv run kijun doctor` で確認してください。"
    )


def unload_model(transport: HttpTransport, cfg: OllamaConfig) -> tuple[bool, str]:
    """モデルを VRAM から下ろす(POST /api/generate、keep_alive 0)。(成功したか, 説明) を返す。

    Ollama に接続できないときも False を返す。呼び出し側が、読み込まれていない(Ollama が動いていない)
    場合と区別して扱えるよう、説明に理由を入れる。
    """
    url = url_of(cfg, GENERATE_PATH)
    try:
        response = transport.post(url, {"model": cfg.model, "keep_alive": 0}, 60)
    except TransportError as e:
        return False, f"モデルを下ろす要求を送れませんでした: {e}"
    if not response.ok:
        return False, f"HTTP {response.status_code}: {response.text[:200]}"
    return True, "モデルを下ろす要求を送りました"


def keeps_model_loaded(cfg: OllamaConfig) -> bool:
    """keep_alive が "0" 以外か。真なら、応答の後もモデルが VRAM に残り得る。"""
    return cfg.keep_alive.strip() != "0"
