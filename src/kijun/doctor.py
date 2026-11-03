"""推論サーバー（Strata）の実測（`uv run kijun doctor`）。

このコマンドの目的は、推測を測定に置き換えることである。

Strata の API 仕様は、計画を書いた時点で確認できていない（stratallm.org に到達できない）。
また Qwen3.8-Flash-Next は 125B の MoE で、Unsloth は1bit量子化でも最低 75GB の RAM が
必要としているが、ユーザーの PC は 64GB で 11GB 足りない。動くかどうか、動くとして
どれだけの速さと入力長で動くかは、実機で測るしかない。

そこで、設定に推測値を書いて決め打ちせず、次の5つを実測して「config.toml に書くべき値」を出す。

1. 接続確認          : /v1/models に応答するか。設定の model がその一覧にあるか。
2. 最小リクエスト    : response_format なしで投げ、出力がそのまま読めるか・修復で読めるかを見る。
3. スキーマ適合の実測: response_format ありで投げ、サーバーが対応するか・保証するかを見る。
4. 速度の実測        : 2つの長さで投げ、tokens/s と所要秒数を出す。
5. 最大入力長の実測  : chunk_minutes 相当・2倍・4倍の3段階で、通る手前の長さを出す。

途中で失敗しても残りの手順を続け、最後にまとめを出す。音声が1本も無くても実行できるよう、
入力は固定の日本語の文を繰り返して作る。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kijun.config import Config
from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractionResult
from kijun.extract.prompt import build_messages
from kijun.extract.repair import parse_with_repair, repair_json_text, UnrepairableJsonError
from kijun.extract.strata_impl import build_payload
from kijun.extract.tokens import (
    estimate_messages_tokens,
    minutes_for_tokens,
    tokens_for_minutes,
)
from kijun.models import Utterance
from kijun.table import format_table
from kijun.transport import HttpResponse, HttpTransport, TransportError

# 検査の状態。英数字で持ち、表示のときに日本語にする（CLAUDE.md 4-2 の方針）。
STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"

STATUS_LABELS = {
    STATUS_OK: "良",
    STATUS_WARN: "注意",
    STATUS_ERROR: "失敗",
    STATUS_SKIPPED: "未実行",
}

# 手順2・手順3の分類
OUTPUT_DIRECT_JSON = "direct_json"  # (a) そのまま JSON として読めた
OUTPUT_REPAIRED = "repaired"  # (b) repair.py を通せば読めた
OUTPUT_UNREADABLE = "unreadable"  # (c) どちらでもない

SCHEMA_UNSUPPORTED = "unsupported"  # (a) エラー。サーバーは response_format に対応していない
SCHEMA_ENFORCED = "enforced"  # (b) 200 でスキーマに適合。enforces_json_schema = true を推奨
SCHEMA_ACCEPTED_NOT_ENFORCED = "accepted_not_enforced"  # (c) 200 だが不適合。受け付けるが保証しない

# 手順4の小さいリクエストの大きさ（入力トークンの概算）
SMALL_REQUEST_TOKENS = 1000
# 手順5で試す倍率。段階を増やすと実機の時間を使うので3段階にとどめる。
MAX_INPUT_MULTIPLIERS = (1, 2, 4)
# timeout_sec の推奨値を出すときの余裕の倍率と下限
TIMEOUT_MARGIN = 3
TIMEOUT_FLOOR_SEC = 600
# max_tokens の推奨値は、出力見込みのこの倍数。下限は 4096
MAX_TOKENS_MARGIN = 2
# timeout_sec = プロンプト処理の秒数 + max_tokens ÷ 観測された最も遅い生成速度 × この係数
TIMEOUT_GENERATION_SAFETY = 1.5
MAX_TOKENS_FLOOR = 4096
# 会議1回を1リクエストで渡せると判断する分数（2時間の会議を想定）
WHOLE_MEETING_MINUTES = 120
# 会議全体を1リクエストで渡すときに設定する chunk_minutes（2時間に余裕を持たせた値）
WHOLE_MEETING_CHUNK_MINUTES = 180

# 入力を引き伸ばすための固定の日本語の文（音声が無くても実行できるようにするため）
STRETCH_SENTENCE = "告知文の冒頭では、読み手にしてほしい行動を先に書いてください。"
# 引き伸ばすときの1発話あたりの文字数
STRETCH_CHARS_PER_UTTERANCE = 120

# 手順2・3で使う固定の短い発話3件
SAMPLE_UTTERANCES = [
    Utterance(0, "SPEAKER_00", 0.0, 4.0, "この告知文、募集要項の最初に何をしてほしいかを書いてください。"),
    Utterance(1, "SPEAKER_01", 4.0, 6.0, "はい、直します。"),
    Utterance(2, "SPEAKER_00", 6.0, 10.0, "あと、グラフには単位と出典を必ず付けてください。"),
]


@dataclass
class CheckResult:
    """検査1つの結果。detail は JSON に書く詳細。"""

    key: str
    title: str
    status: str
    summary: str
    detail: dict = field(default_factory=dict)


@dataclass
class DoctorReport:
    base_url: str
    chat_path: str
    model: str
    started_at: datetime
    checks: list[CheckResult] = field(default_factory=list)
    recommendations: dict = field(default_factory=dict)

    def get(self, key: str) -> CheckResult | None:
        return next((c for c in self.checks if c.key == key), None)

    def to_json(self) -> str:
        return json.dumps(
            {
                "base_url": self.base_url,
                "chat_path": self.chat_path,
                "model": self.model,
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "checks": [
                    {
                        "key": c.key,
                        "title": c.title,
                        "status": c.status,
                        "summary": c.summary,
                        "detail": c.detail,
                    }
                    for c in self.checks
                ],
                "recommendations": self.recommendations,
            },
            ensure_ascii=False,
            indent=2,
        )


# --- 入力を作る ------------------------------------------------------------------------------


def sample_chunk() -> Chunk:
    """手順2・3で使う固定の短いチャンク。"""
    return Chunk(index=0, seq_from=0, seq_to=len(SAMPLE_UTTERANCES) - 1, utterances=list(SAMPLE_UTTERANCES))


def stretched_chunk(cfg: Config, target_user_tokens: int) -> Chunk:
    """固定の日本語の文を繰り返して、文字起こし部分が約 target_user_tokens になるチャンクを作る。

    システムプロンプトの分はこの長さに含まれない（文字起こしの長さだけを指定する）。
    入力全体の概算は estimate_messages_tokens で別に求める。
    """
    target_chars = max(1, int(target_user_tokens / cfg.extract.tokens_per_character))
    count = max(1, target_chars // STRETCH_CHARS_PER_UTTERANCE)
    repeated = STRETCH_SENTENCE * (STRETCH_CHARS_PER_UTTERANCE // len(STRETCH_SENTENCE) + 2)
    text = repeated[:STRETCH_CHARS_PER_UTTERANCE]
    utterances = [
        Utterance(
            seq=i,
            speaker_label=f"SPEAKER_{i % 2:02d}",
            start_sec=i * 30.0,
            end_sec=(i + 1) * 30.0,
            text=text,
        )
        for i in range(count)
    ]
    return Chunk(index=0, seq_from=0, seq_to=count - 1, utterances=utterances)


def _payload(cfg: Config, chunk: Chunk, with_schema: bool) -> dict:
    """抽出が実際に送るのと同じ形のリクエストを作る。

    strata_impl.build_payload をそのまま使うのは、doctor が測るものと extract が
    送るものを食い違わせないためである。
    """
    return build_payload(cfg, chunk, with_schema)


def _usage_tokens(response: HttpResponse) -> tuple[int | None, int | None]:
    """サーバーが返した入力・出力のトークン数。返していなければ (None, None)。"""
    try:
        usage = response.json().get("usage") or {}
    except (TransportError, AttributeError):
        return None, None
    if not isinstance(usage, dict):
        return None, None
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _finish_reason(response: HttpResponse) -> str | None:
    """応答の finish_reason。length なら出力が max_tokens で切れている。"""
    try:
        return response.json()["choices"][0].get("finish_reason")
    except (TransportError, KeyError, IndexError, TypeError, AttributeError):
        return None


def _content(response: HttpResponse) -> str | None:
    """OpenAI 互換の応答から本文を取り出す。形が違えば None。"""
    try:
        body = response.json()
        return body["choices"][0]["message"]["content"]
    except (TransportError, KeyError, IndexError, TypeError):
        return None


def classify_output(text: str) -> tuple[str, list[str], str]:
    """出力を (分類, 行った修復, 説明) に分ける。

    そのまま読めれば direct_json、修復して読めれば repaired、読めなければ unreadable。
    """
    bare = repair_json_text(text)
    try:
        parse_with_repair(text)
    except UnrepairableJsonError as e:
        return OUTPUT_UNREADABLE, list(bare.actions), str(e)[:300]
    if not bare.actions and not bare.truncated_items:
        return OUTPUT_DIRECT_JSON, [], "出力はそのまま JSON として読めた"
    return (
        OUTPUT_REPAIRED,
        list(bare.actions),
        "修復して読めた: " + "、".join(bare.descriptions),
    )


# --- 手順1: 接続確認 -------------------------------------------------------------------------


def _model_ids(body) -> list[str]:
    """/v1/models の応答からモデルIDの一覧を取り出す。形が分からないので複数の形に備える。"""
    if isinstance(body, list):
        entries = body
    elif isinstance(body, dict):
        entries = body.get("data") or body.get("models") or []
    else:
        return []
    ids = []
    for e in entries:
        if isinstance(e, dict):
            value = e.get("id") or e.get("name")
            if value:
                ids.append(str(value))
        elif isinstance(e, str):
            ids.append(e)
    return ids


def check_models(transport: HttpTransport, cfg: Config) -> CheckResult:
    """手順1: /v1/models に応答するか。設定の model がその一覧にあるか。"""
    url = f"{cfg.extract.strata.base_url}/v1/models"
    try:
        response = transport.get(url, timeout_sec=30)
    except TransportError as e:
        return CheckResult(
            "models", "1. 接続確認", STATUS_ERROR,
            f"{url} に接続できません（Strata が起動しているか、base_url を確認してください）: {e}",
            {"url": url},
        )
    if not response.ok:
        return CheckResult(
            "models", "1. 接続確認", STATUS_ERROR,
            f"{url} が HTTP {response.status_code} を返しました",
            {"url": url, "status_code": response.status_code, "body_head": response.text[:300]},
        )
    try:
        ids = _model_ids(response.json())
    except TransportError as e:
        return CheckResult(
            "models", "1. 接続確認", STATUS_WARN,
            f"応答を JSON として読めませんでした: {e}",
            {"url": url, "body_head": response.text[:300]},
        )
    configured = cfg.extract.strata.model
    detail = {"url": url, "model_ids": ids, "configured_model": configured}
    if not ids:
        return CheckResult(
            "models", "1. 接続確認", STATUS_WARN,
            "応答はあったが、モデルIDの一覧を読み取れませんでした", detail,
        )
    if configured in ids:
        return CheckResult(
            "models", "1. 接続確認", STATUS_OK,
            f"接続できました。設定の model「{configured}」は一覧にあります（全 {len(ids)} 件）",
            detail,
        )
    return CheckResult(
        "models", "1. 接続確認", STATUS_WARN,
        f"接続できましたが、設定の model「{configured}」は一覧にありません。"
        f"一覧: {', '.join(ids[:10])}",
        detail,
    )


# --- 手順2・3: 出力の形とスキーマ適合 ----------------------------------------------------------


def _probe(
    transport: HttpTransport, cfg: Config, chunk: Chunk, with_schema: bool
) -> tuple[HttpResponse | None, str | None]:
    """1回投げて (応答, 失敗の理由) を返す。"""
    url = f"{cfg.extract.strata.base_url}{cfg.extract.strata.chat_path}"
    try:
        return transport.post(url, _payload(cfg, chunk, with_schema), cfg.extract.strata.timeout_sec), None
    except TransportError as e:
        return None, f"{url} に要求を送れませんでした: {e}"


def check_minimal_request(transport: HttpTransport, cfg: Config) -> CheckResult:
    """手順2: response_format なしで短い抽出を投げ、出力が読めるかを3つに分類する。"""
    title = "2. 最小リクエスト（response_format なし）"
    response, error = _probe(transport, cfg, sample_chunk(), with_schema=False)
    if error is not None:
        return CheckResult("minimal", title, STATUS_ERROR, error, {})
    assert response is not None
    if not response.ok:
        return CheckResult(
            "minimal", title, STATUS_ERROR,
            f"HTTP {response.status_code} が返りました: {response.text[:200]}",
            {"status_code": response.status_code, "body_head": response.text[:300]},
        )
    text = _content(response)
    if text is None:
        return CheckResult(
            "minimal", title, STATUS_ERROR,
            "応答が OpenAI 互換の形（choices[0].message.content）ではありません。"
            "設定の chat_path を確認してください。",
            {"body_head": response.text[:300]},
        )
    classification, actions, note = classify_output(text)
    detail = {
        "classification": classification,
        "repair_actions": actions,
        "note": note,
        "elapsed_sec": round(response.elapsed_sec, 2),
        "content_head": text[:300],
    }
    if classification == OUTPUT_DIRECT_JSON:
        return CheckResult("minimal", title, STATUS_OK, "(a) 出力はそのまま JSON として読めました", detail)
    if classification == OUTPUT_REPAIRED:
        return CheckResult(
            "minimal", title, STATUS_WARN,
            f"(b) repair.py を通せば読めました（{note}）", detail,
        )
    return CheckResult(
        "minimal", title, STATUS_ERROR,
        f"(c) 修復しても読めませんでした。プロンプトか設定の見直しが必要です（{note}）", detail,
    )


def check_schema_enforcement(transport: HttpTransport, cfg: Config) -> CheckResult:
    """手順3: response_format ありで投げ、サーバーが対応するか・保証するかを3つに分類する。"""
    title = "3. スキーマ適合の実測（response_format あり）"
    response, error = _probe(transport, cfg, sample_chunk(), with_schema=True)
    if error is not None:
        return CheckResult("schema", title, STATUS_ERROR, error, {})
    assert response is not None
    if not response.ok:
        return CheckResult(
            "schema", title, STATUS_WARN,
            f"(a) HTTP {response.status_code}。サーバーは response_format に対応していません。"
            "enforces_json_schema は false のままにしてください。",
            {
                "classification": SCHEMA_UNSUPPORTED,
                "status_code": response.status_code,
                "body_head": response.text[:300],
            },
        )
    text = _content(response)
    if text is None:
        return CheckResult(
            "schema", title, STATUS_WARN,
            "(c) 200 が返りましたが、応答が OpenAI 互換の形ではありません。",
            {"classification": SCHEMA_ACCEPTED_NOT_ENFORCED, "body_head": response.text[:300]},
        )
    classification, actions, note = classify_output(text)
    detail = {
        "repair_actions": actions,
        "note": note,
        "elapsed_sec": round(response.elapsed_sec, 2),
        "content_head": text[:300],
    }
    if classification == OUTPUT_DIRECT_JSON:
        detail["classification"] = SCHEMA_ENFORCED
        return CheckResult(
            "schema", title, STATUS_OK,
            "(b) 200 でスキーマに適合しました。config.toml の enforces_json_schema = true を推奨します。",
            detail,
        )
    detail["classification"] = SCHEMA_ACCEPTED_NOT_ENFORCED
    return CheckResult(
        "schema", title, STATUS_WARN,
        f"(c) 200 ですがスキーマに適合しません（受け付けるが保証しない）。"
        f"enforces_json_schema は false のままにしてください（{note}）",
        detail,
    )


# --- 手順4: 速度の実測 -----------------------------------------------------------------------


def _measure(transport: HttpTransport, cfg: Config, chunk: Chunk, label: str) -> dict:
    """1回投げて、入力・出力のトークン数と所要秒数を測る。"""
    system, user = build_messages(chunk, cfg)
    estimated = estimate_messages_tokens(system, user, cfg.extract.tokens_per_character)
    record: dict = {
        "label": label,
        "utterances": len(chunk.utterances),
        "estimated_input_tokens": estimated,
        # 送った文字数。tokens_per_character の実測（usage.prompt_tokens ÷ この値）に使う。
        "input_chars": len(system) + len(user),
    }
    response, error = _probe(transport, cfg, chunk, with_schema=cfg.extract.strata.enforces_json_schema)
    if error is not None:
        record["error"] = error
        return record
    assert response is not None
    record["status_code"] = response.status_code
    record["elapsed_sec"] = round(response.elapsed_sec, 2)
    if not response.ok:
        record["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
        return record
    prompt_tokens, completion_tokens = _usage_tokens(response)
    record["input_tokens"] = prompt_tokens
    record["output_tokens"] = completion_tokens
    record["usage_reported"] = prompt_tokens is not None or completion_tokens is not None
    if completion_tokens and response.elapsed_sec > 0:
        record["output_tokens_per_sec"] = round(completion_tokens / response.elapsed_sec, 1)
    if prompt_tokens and response.elapsed_sec > 0:
        record["input_tokens_per_sec"] = round(prompt_tokens / response.elapsed_sec, 1)
    record["finish_reason"] = _finish_reason(response)
    text = _content(response)
    if text is not None:
        record["classification"] = classify_output(text)[0]
    return record


def check_speed(transport: HttpTransport, cfg: Config) -> CheckResult:
    """手順4: 約1,000トークンと、chunk_minutes 相当の長さで投げ、tokens/s を出す。"""
    title = "4. 速度の実測"
    chunk_tokens = tokens_for_minutes(cfg.extract.chunk_minutes, cfg.extract.tokens_per_meeting_minute)
    measurements = [
        _measure(transport, cfg, stretched_chunk(cfg, SMALL_REQUEST_TOKENS), f"約{SMALL_REQUEST_TOKENS}トークン"),
        _measure(
            transport, cfg, stretched_chunk(cfg, chunk_tokens),
            f"chunk_minutes={cfg.extract.chunk_minutes} 相当（約{chunk_tokens}トークン）",
        ),
    ]
    detail = {"measurements": measurements}
    ok = [m for m in measurements if "error" not in m]
    if not ok:
        return CheckResult("speed", title, STATUS_ERROR, "どちらの長さも失敗しました", detail)
    parts = []
    for m in ok:
        speed = m.get("output_tokens_per_sec")
        parts.append(
            f"{m['label']}: {m.get('elapsed_sec')} 秒"
            + (f"、出力 {speed} tokens/s" if speed else "、出力 tokens/s は usage が無いため不明")
        )
    status = STATUS_OK if len(ok) == len(measurements) else STATUS_WARN
    if not any(m.get("usage_reported") for m in ok):
        status = STATUS_WARN
        parts.append("サーバーが usage を返さないため、トークン数は概算しか分かりません")
    return CheckResult("speed", title, status, " / ".join(parts), detail)


# --- 手順5: 最大入力長の実測 ------------------------------------------------------------------


def check_max_input(transport: HttpTransport, cfg: Config) -> CheckResult:
    """手順5: chunk_minutes 相当・2倍・4倍の3段階で投げ、通る手前の長さを出す。

    段階を増やすと実機の時間を使うので3段階にとどめる。
    HTTP エラーか接続不可を「その長さは通らなかった」とみなす。200 が返れば、
    出力が読めるかどうかに関わらず「その長さは通った」とみなす（長さの問題と出力の
    壊れ方は別の問題なので、分けて報告する）。
    """
    title = "5. 最大入力長の実測"
    base_tokens = tokens_for_minutes(cfg.extract.chunk_minutes, cfg.extract.tokens_per_meeting_minute)
    attempts: list[dict] = []
    largest_passed: int | None = None
    for multiplier in MAX_INPUT_MULTIPLIERS:
        target = base_tokens * multiplier
        chunk = stretched_chunk(cfg, target)
        system, user = build_messages(chunk, cfg)
        record: dict = {
            "multiplier": multiplier,
            "target_user_tokens": target,
            "target_minutes": round(minutes_for_tokens(target, cfg.extract.tokens_per_meeting_minute), 1),
            "estimated_input_tokens": estimate_messages_tokens(system, user, cfg.extract.tokens_per_character),
        }
        response, error = _probe(transport, cfg, chunk, with_schema=cfg.extract.strata.enforces_json_schema)
        if error is not None:
            record["passed"] = False
            record["error"] = error
            attempts.append(record)
            break
        assert response is not None
        record["status_code"] = response.status_code
        record["elapsed_sec"] = round(response.elapsed_sec, 2)
        prompt_tokens, _completion = _usage_tokens(response)
        record["input_tokens"] = prompt_tokens
        if not response.ok:
            record["passed"] = False
            record["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
            attempts.append(record)
            break
        text = _content(response)
        record["classification"] = classify_output(text)[0] if text is not None else None
        record["passed"] = True
        largest_passed = target
        attempts.append(record)

    detail = {
        "attempts": attempts,
        "largest_passed_user_tokens": largest_passed,
        "all_multipliers_passed": largest_passed == base_tokens * MAX_INPUT_MULTIPLIERS[-1],
    }
    if largest_passed is None:
        return CheckResult(
            "max_input", title, STATUS_ERROR,
            f"最小の段階（chunk_minutes={cfg.extract.chunk_minutes} 相当、約{base_tokens}トークン）も通りませんでした",
            detail,
        )
    minutes = round(minutes_for_tokens(largest_passed, cfg.extract.tokens_per_meeting_minute), 1)
    if detail["all_multipliers_passed"]:
        summary = (
            f"{MAX_INPUT_MULTIPLIERS[-1]}倍でも通りました（会議 約{minutes} 分相当、"
            f"約{largest_passed}トークン）。上限はこれ以上で、未確定です。"
        )
        if minutes < WHOLE_MEETING_MINUTES:
            summary += (
                f" 2時間（{WHOLE_MEETING_MINUTES}分）相当はまだ確かめていません。"
                f"確かめるには chunk_minutes を {WHOLE_MEETING_MINUTES // MAX_INPUT_MULTIPLIERS[-1]} 以上にして"
                " doctor をもう一度実行してください。"
            )
        return CheckResult("max_input", title, STATUS_OK, summary, detail)
    failed = attempts[-1]
    return CheckResult(
        "max_input", title, STATUS_WARN,
        f"会議 約{minutes} 分相当（約{largest_passed}トークン）までは通り、"
        f"{failed['multiplier']}倍（約{failed['target_user_tokens']}トークン）で失敗しました",
        detail,
    )


# --- 推奨値 ---------------------------------------------------------------------------------


def _fit_latency(cfg: Config, measurements: list[dict]) -> tuple[float, float, int] | None:
    """検査4の2点（短い・chunk_minutes 相当）から、所要秒数 = 切片 + 傾き × 入力トークン数 を当てはめる。

    戻り値は (傾き, 切片, 上乗せトークン数)。上乗せは、実測した入力（システムプロンプトを含む）が
    tokens_for_minutes(chunk_minutes) より多い分で、推奨する長さにも同じだけ加える。
    2点が揃わない、または傾きが0以下なら None。
    """
    if len(measurements) < 2:
        return None
    short, long_ = measurements[0], measurements[1]
    if any("error" in m or not m.get("elapsed_sec") for m in (short, long_)):
        return None
    x1 = short.get("estimated_input_tokens") or SMALL_REQUEST_TOKENS
    x2 = long_.get("estimated_input_tokens") or tokens_for_minutes(cfg.extract.chunk_minutes, cfg.extract.tokens_per_meeting_minute)
    if x2 <= x1:
        return None
    slope = (long_["elapsed_sec"] - short["elapsed_sec"]) / (x2 - x1)
    if slope <= 0:
        return None
    intercept = max(0.0, short["elapsed_sec"] - x1 * slope)
    overhead = max(0, x2 - tokens_for_minutes(cfg.extract.chunk_minutes, cfg.extract.tokens_per_meeting_minute))
    return slope, intercept, overhead


def _usable(m: dict) -> bool:
    """usage の入力・出力トークン数と所要秒数が揃った測定か。"""
    return (
        "error" not in m
        and bool(m.get("input_tokens"))
        and bool(m.get("output_tokens"))
        and bool(m.get("elapsed_sec"))
    )


def _recommend_tokens_per_character(rec: dict, measurements: list[dict]) -> None:
    """検査4の usage.prompt_tokens と送った文字数から、tokens_per_character を実測して推奨値に入れる。"""
    used = [m for m in measurements if "error" not in m and m.get("input_tokens") and m.get("input_chars")]
    if not used:
        return
    tokens = sum(m["input_tokens"] for m in used)
    chars = sum(m["input_chars"] for m in used)
    value = math.ceil(tokens / chars * 100) / 100
    rec["tokens_per_character"] = {
        "value": value,
        "reason": (
            f"実測値。検査4で送った {chars} 文字（system と user の合計）に対し、サーバーが数えた "
            f"入力は {tokens} トークン（usage.prompt_tokens）で、{tokens / chars:.3f} トークン/文字。"
            "小数第2位に切り上げた。固定の文を繰り返した入力からの値なので、"
            "実際の会議の文字起こしでは多少ずれる可能性がある"
        ),
    }


def _fit_generation(measurements: list[dict]) -> dict | None:
    """検査4の2点から、入力1トークンあたりの秒数と出力1トークンあたりの秒数を当てはめる。

    2点・2係数（切片なし）の連立方程式を解く。2点の「出力 ÷ 入力」の比がほぼ同じだと、入力と出力の
    寄与を分けられない（連立方程式が特異に近づく）。そのとき、または係数が負になるときは、
    入力の係数は 0 とする（プロンプト処理の時間は出力の速度に含まれる）。

    戻り値: sec_per_input（分けられないときは 0）, min_output_speed（測定ごとの 出力 ÷ 秒数 の最小）,
    output_ratio（長い側の 出力 ÷ 入力）, mode（separated / output_only / single）など。
    usage が取れた測定が1点も無ければ None。1点だけのときは、その値を使う（mode = single）。
    """
    usable = [m for m in measurements[:2] if _usable(m)]
    if not usable:
        return None
    # 生成速度は測定ごとに 出力 ÷ 所要秒数（プロンプト処理の時間を含む）。最も遅いものを採る。
    # 同じ profile・モデルでも 61〜69 tokens/s と 14 tokens/s が観測されており、1回の測定では決められない。
    min_speed = min(m["output_tokens"] / m["elapsed_sec"] for m in usable)
    long_ = usable[-1]
    base = {
        "output_ratio": long_["output_tokens"] / long_["input_tokens"],
        "input_tokens": long_["input_tokens"],
        "min_output_speed": min_speed,
        "measured_points": len(usable),
        "truncated": any(m.get("finish_reason") == "length" for m in usable),
    }
    if len(usable) < 2:
        return {**base, "sec_per_input": 0.0, "mode": "single"}
    short = usable[0]
    i1, o1, t1 = short["input_tokens"], short["output_tokens"], short["elapsed_sec"]
    i2, o2, t2 = long_["input_tokens"], long_["output_tokens"], long_["elapsed_sec"]
    r1, r2 = o1 / i1, o2 / i2
    mode = "separated"
    a = b = 0.0
    if abs(r1 - r2) / max(r1, r2) >= 0.2:
        det = i1 * o2 - i2 * o1
        a = (t1 * o2 - t2 * o1) / det
        b = (i1 * t2 - i2 * t1) / det
    if not (a > 0 and b > 0):
        mode = "output_only"
        a = 0.0
    return {**base, "sec_per_input": a, "mode": mode}


def _recommend_with_output_length(
    cfg: Config, rec: dict, gen: dict, recommended_chunk_minutes: int
) -> None:
    """出力長を入れた所要時間の推定から、timeout_sec と max_tokens の推奨値を作る。

    推奨する chunk_minutes の入力トークン数は、検査4の長い側の実測入力（usage.prompt_tokens）を
    chunk_minutes の比で伸縮して求める。出力トークン数は 入力 × 実測の比 とする。
    """
    scale = recommended_chunk_minutes / cfg.extract.chunk_minutes
    planned_in = int(gen["input_tokens"] * scale)
    planned_out = int(planned_in * gen["output_ratio"])
    max_tokens = max(MAX_TOKENS_FLOOR, int(math.ceil(planned_out * MAX_TOKENS_MARGIN / 1000) * 1000))
    prompt_sec = gen["sec_per_input"] * planned_in
    generation_sec = max_tokens / gen["min_output_speed"]
    estimated_sec = prompt_sec + generation_sec * TIMEOUT_GENERATION_SAFETY
    suggested = max(TIMEOUT_FLOOR_SEC, int(math.ceil(estimated_sec / 60) * 60))
    expected_sec = prompt_sec + planned_out / gen["min_output_speed"]
    if gen["mode"] == "separated":
        fit_note = (
            f"プロンプト処理は入力 1トークンあたり {gen['sec_per_input']:.4f} 秒と当てはめた"
            f"（約{prompt_sec:.0f} 秒）。"
        )
    else:
        fit_note = (
            "プロンプト処理の時間は、出力の速度に含まれるものとして別には足していない"
            + ("（測定が1点しか取れなかった）" if gen["mode"] == "single" else "（2点の 出力÷入力 の比が近い、または分けると係数が負になるため、入力と出力を分けられなかった）")
            + "。"
        )
    truncated_note = (
        " 注意: 検査4の出力は max_tokens で切れていた（finish_reason = length）ので、出力の比は下限である。"
        if gen["truncated"]
        else ""
    )
    rec["timeout_sec"] = {
        "value": suggested,
        "reason": (
            f"timeout_sec = プロンプト処理の秒数 + max_tokens ÷ 観測された最も遅い生成速度 × {TIMEOUT_GENERATION_SAFETY}"
            f"（下限 {TIMEOUT_FLOOR_SEC} 秒、分単位に切り上げ）。検査4で観測された最も遅い生成速度は "
            f"{gen['min_output_speed']:.1f} tokens/s（プロンプト処理を含む）、推奨の max_tokens={max_tokens}。{fit_note}"
            "生成速度は同じ profile・モデルでも 61〜69 tokens/s と 14 tokens/s の4.4倍の振れが観測されているため、"
            "1回の測定の定数倍では決めていない。"
            f"参考: 推奨する chunk_minutes={recommended_chunk_minutes} では、入力約{planned_in}トークン、"
            f"出力約{planned_out}トークン（実測の比 {gen['output_ratio']:.2f} 倍）と見込み、"
            f"最も遅い速度での見込みは約{expected_sec:.0f} 秒。"
            "出力の比は固定の文を繰り返した入力からの値で、実際の会議では異なる可能性がある。"
            + truncated_note
        ),
    }
    rec["max_tokens"] = {
        "value": max_tokens,
        "reason": (
            f"推奨する chunk_minutes={recommended_chunk_minutes} での出力見込み約{planned_out}トークン"
            f"（入力約{planned_in}トークン × 実測の比 {gen['output_ratio']:.2f}）の {MAX_TOKENS_MARGIN} 倍"
            f"（下限 {MAX_TOKENS_FLOOR}、1000 単位に切り上げ）。推論トークンを含む。"
            + truncated_note
        ),
    }


def build_recommendations(cfg: Config, report: DoctorReport) -> dict:
    """実測の結果から、config.toml に書くべき値を出す。

    測れなかった項目は「現在の値のまま」とし、推測値を書かない。
    """
    rec: dict = {}

    schema = report.get("schema")
    if schema is None or schema.status == STATUS_SKIPPED:
        rec["enforces_json_schema"] = {
            "value": cfg.extract.strata.enforces_json_schema,
            "reason": "測れなかったので現在の値のまま",
        }
    else:
        enforced = schema.detail.get("classification") == SCHEMA_ENFORCED
        rec["enforces_json_schema"] = {
            "value": enforced,
            "reason": (
                "response_format ありで 200 が返り、出力がスキーマに適合した"
                if enforced
                else f"スキーマへの適合を保証しないため false のまま（分類: {schema.detail.get('classification')}）"
            ),
        }

    max_input = report.get("max_input")
    largest = max_input.detail.get("largest_passed_user_tokens") if max_input else None
    if largest:
        estimated_total = next(
            (
                a.get("estimated_input_tokens")
                for a in reversed(max_input.detail.get("attempts", []))
                if a.get("passed")
            ),
            largest,
        )
        rec["max_input_tokens"] = {
            "value": estimated_total,
            "reason": (
                f"この長さ（会議 約{round(minutes_for_tokens(largest, cfg.extract.tokens_per_meeting_minute), 1)} 分相当）までは通った。"
                + (
                    "最大の段階まで通ったので、実際の上限はこれ以上で未確定"
                    if max_input.detail.get("all_multipliers_passed")
                    else "次の段階では失敗した"
                )
            ),
        }
    else:
        rec["max_input_tokens"] = {
            "value": 0,
            "reason": "通る長さが1つも見つからなかったので、未確認（0）のまま",
        }

    if largest:
        minutes = minutes_for_tokens(largest, cfg.extract.tokens_per_meeting_minute)
        if minutes >= WHOLE_MEETING_MINUTES:
            rec["chunk_minutes"] = {
                "value": WHOLE_MEETING_CHUNK_MINUTES,
                "reason": f"会議 約{round(minutes, 1)} 分相当が1リクエストで通ったので、会議全体を分割せずに渡せる",
            }
            rec["overlap_minutes"] = {
                "value": 0,
                "reason": "分割しないので重複は不要",
            }
        else:
            # 通った長さの8割を目安にし、5分単位に切り下げる（概算なので余裕を取る）
            safe = max(5, int(minutes * 0.8 // 5) * 5)
            rec["chunk_minutes"] = {
                "value": safe,
                "reason": (
                    f"通ったのは会議 約{round(minutes, 1)} 分相当まで。概算に余裕を持たせて8割にし、"
                    f"5分単位に切り下げた。2時間を1リクエストで渡すには {WHOLE_MEETING_MINUTES} 分相当が必要"
                ),
            }
            rec["overlap_minutes"] = {
                "value": cfg.extract.overlap_minutes,
                "reason": "分割するので重複は現在の値のまま",
            }
    else:
        rec["chunk_minutes"] = {
            "value": cfg.extract.chunk_minutes,
            "reason": "測れなかったので現在の値のまま",
        }
        rec["overlap_minutes"] = {
            "value": cfg.extract.overlap_minutes,
            "reason": "測れなかったので現在の値のまま",
        }
    speed = report.get("speed")
    measurements = speed.detail.get("measurements", []) if speed is not None else []
    measured = next(
        (
            m.get("elapsed_sec")
            for m in reversed(measurements)
            if "error" not in m and m.get("elapsed_sec")
        ),
        None,
    )
    recommended_chunk_minutes = rec["chunk_minutes"]["value"]
    _recommend_tokens_per_character(rec, measurements)
    gen = _fit_generation(measurements)
    fit = _fit_latency(cfg, measurements)
    if measured and gen is not None:
        _recommend_with_output_length(cfg, rec, gen, recommended_chunk_minutes)
    elif measured and fit is not None:
        slope, intercept, overhead = fit
        planned_tokens = (
            tokens_for_minutes(recommended_chunk_minutes, cfg.extract.tokens_per_meeting_minute) + overhead
        )
        estimated_sec = intercept + slope * planned_tokens
        suggested = max(TIMEOUT_FLOOR_SEC, int(math.ceil(estimated_sec * TIMEOUT_MARGIN / 60) * 60))
        rec["timeout_sec"] = {
            "value": suggested,
            "reason": (
                f"検査4の2点（約{SMALL_REQUEST_TOKENS}トークンと chunk_minutes={cfg.extract.chunk_minutes} 相当）から"
                f"直線で当てはめた（1トークンあたり {slope:.4f} 秒、固定の立ち上がり {intercept:.0f} 秒。"
                "トークン数は概算）。推奨する chunk_minutes="
                f"{recommended_chunk_minutes}（約{planned_tokens}トークン）での推定所要時間は約{estimated_sec:.0f} 秒。"
                f"その {TIMEOUT_MARGIN} 倍（下限 {TIMEOUT_FLOOR_SEC} 秒、分単位に切り上げ）。"
                "注意: サーバーが usage を返さないため、出力長を推定に入れていない"
                "（実測では出力トークン数のほうが所要時間を支配する）"
            ),
        }
    elif measured:
        suggested = max(TIMEOUT_FLOOR_SEC, int(math.ceil(measured * TIMEOUT_MARGIN / 60) * 60))
        rec["timeout_sec"] = {
            "value": suggested,
            "reason": (
                f"実測した長さの {TIMEOUT_MARGIN} 倍（実測 {measured} 秒。下限 {TIMEOUT_FLOOR_SEC} 秒、分単位に切り上げ）。"
                "推奨する chunk_minutes での所要時間は推定できていない"
                "（検査4の2点が揃わない、または長い入力のほうが速いなど直線に当てはまらない）。"
                "出力長を推定に入れていない"
            ),
        }
    else:
        rec["timeout_sec"] = {
            "value": cfg.extract.strata.timeout_sec,
            "reason": "測れなかったので現在の値のまま",
        }
    if "max_tokens" not in rec:
        rec["max_tokens"] = {
            "value": cfg.extract.strata.max_tokens,
            "reason": (
                "測れなかったので現在の値のまま"
                "（検査4の2点で usage の入力・出力トークン数が取れないと、出力見込みを計算できない）"
            ),
        }
    order = [
        "enforces_json_schema",
        "max_input_tokens",
        "max_tokens",
        "tokens_per_character",
        "timeout_sec",
        "chunk_minutes",
        "overlap_minutes",
    ]
    rec = {k: rec[k] for k in order if k in rec}
    return rec


# --- 実行 -----------------------------------------------------------------------------------


def run_doctor(transport: HttpTransport, cfg: Config, now: datetime | None = None) -> DoctorReport:
    """手順1から5を順に実行する。途中で失敗しても残りを続ける。"""
    report = DoctorReport(
        base_url=cfg.extract.strata.base_url,
        chat_path=cfg.extract.strata.chat_path,
        model=cfg.extract.strata.model,
        started_at=now or datetime.now(),
    )
    for check in (check_models, check_minimal_request, check_schema_enforcement, check_speed, check_max_input):
        report.checks.append(check(transport, cfg))
    report.recommendations = build_recommendations(cfg, report)
    return report


def format_report(report: DoctorReport) -> str:
    """結果を表にする。

    説明文は長くなるので、表には「検査」と「結果」だけを入れ、内容はその下に字下げして並べる
    （1列が極端に広い表は端末で読みにくいため）。
    """
    lines = [
        f"Strata の実測: {report.base_url}{report.chat_path}（model = {report.model}）",
        f"実行時刻: {report.started_at:%Y-%m-%d %H:%M:%S}",
        "",
        format_table(
            ["検査", "結果"],
            [[c.title, STATUS_LABELS.get(c.status, c.status)] for c in report.checks],
        ),
        "",
        "各検査の内容:",
        "",
    ]
    for c in report.checks:
        lines.append(f"{c.title}（{STATUS_LABELS.get(c.status, c.status)}）")
        lines.append(f"  {c.summary}")
        lines.append("")
    lines += ["config.toml に書くべき推奨値:", ""]
    lines.append(
        format_table(
            ["項目", "推奨値"],
            [
                [key, json.dumps(value["value"], ensure_ascii=False)]
                for key, value in report.recommendations.items()
            ],
        )
    )
    lines += ["", "推奨値の根拠:", ""]
    for key, value in report.recommendations.items():
        lines.append(f"{key} = {json.dumps(value['value'], ensure_ascii=False)}")
        lines.append(f"  {value['reason']}")
        lines.append("")
    lines.append(
        "[extract] の chunk_minutes と overlap_minutes、"
        "[extract.strata] の enforces_json_schema・max_input_tokens・timeout_sec に書いてください。"
    )
    return "\n".join(lines)


def report_path(cfg: Config, at: datetime) -> Path:
    """結果の JSON の保存先。"""
    return cfg.paths.db.parent / "doctor" / f"strata_{at:%Y%m%d_%H%M%S}.json"
