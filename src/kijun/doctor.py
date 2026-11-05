"""推論サーバー（Ollama）の実測（`uv run kijun doctor`）。

このコマンドの目的は、推測を測定に置き換えることである。特に、**モデルが GPU に収まっているか**を
確かめる。前の背面（Strata）は 125B の MoE で、プロセスが RAM を約56GB 保持し、PC の他の作業を
圧迫した。Ollama に移したのは、VRAM 12GB に収まる 9B〜14B のモデルを動かすためであり、
一部がメインメモリに溢れていれば速度が落ち、同じ問題が再発する。検査3がそれを防ぐ要である。

次の5つを実測して「config.toml に書くべき値」を出す。

1. 接続とモデルの有無: GET /api/tags に設定のモデルがあるか。
2. スキーマへの適合   : 抽出と同じ形のリクエストを format にスキーマを付けて送り、
                        repair を通さずに ExtractionResult として読めるか。
3. GPU に収まっているか: 検査2の直後に GET /api/ps を読み、size_vram と size を比べる。
4. 速度               : chunk_minutes 相当の長さの入力を送り、tokens/s と所要時間を出す。
                        2時間の会議の所要時間の見込みを、最も遅い測定値で計算して出す。
5. 入力の切り捨て     : 検査4の prompt_eval_count が num_ctx - num_predict に達していないか。
                        tokens_per_character の実測値も出す。

途中で失敗しても残りの手順を続ける。ただし検査1が失敗したとき（接続できない、モデルが無い）は、
残りは同じ理由で失敗するので実行しない。音声が1本も無くても実行できるよう、入力は固定の日本語の文を
繰り返して作る。

検査3のために、モデルが VRAM に残っている必要がある。doctor 自身の要求だけは keep_alive を "2m" にし、
最後に keep_alive 0 の要求を送ってモデルを下ろす。
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kijun import ollama_api
from kijun.config import Config
from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractionResult
from kijun.extract.ollama_impl import build_payload
from kijun.extract.prompt import build_messages
from kijun.extract.repair import UnrepairableJsonError, parse_with_repair, repair_json_text
from kijun.extract.tokens import estimate_messages_tokens, tokens_for_minutes
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

# 出力の分類（検査2）
OUTPUT_DIRECT_JSON = "direct_json"  # (a) そのまま JSON として読めた
OUTPUT_REPAIRED = "repaired"  # (b) repair.py を通せば読めた
OUTPUT_UNREADABLE = "unreadable"  # (c) どちらでもない

# doctor 自身の要求でモデルを VRAM に残す時間。検査2の直後に検査3が /api/ps を読むために要る。
DOCTOR_KEEP_ALIVE = "2m"
# timeout_sec の推奨値の余裕の倍率と下限
TIMEOUT_GENERATION_SAFETY = 1.5
TIMEOUT_FLOOR_SEC = 600
# 生成速度を出すのに必要な、出力の最小トークン数。これより短い出力の速度は誤差が大きい
# （固定の文を繰り返した入力では、モデルが短い出力しか返さないことがある）。
MIN_EVAL_TOKENS_FOR_SPEED = 50
# 2時間の会議の所要時間を見込むときの会議の長さ（分）
WHOLE_MEETING_MINUTES = 120

# 入力を引き伸ばすための固定の日本語の文（音声が無くても実行できるようにするため）
STRETCH_SENTENCE = "告知文の冒頭では、読み手にしてほしい行動を先に書いてください。"
# 引き伸ばすときの1発話あたりの文字数
STRETCH_CHARS_PER_UTTERANCE = 120

# 検査2で使う固定の短い発話3件
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
    model: str
    started_at: datetime
    checks: list[CheckResult] = field(default_factory=list)
    recommendations: dict = field(default_factory=dict)
    # 最後にモデルを下ろした結果（{"ok": bool, "detail": str}）。下ろす要求を送らなかったときは None。
    unload: dict | None = None
    # 2時間の会議の所要時間の見込み（estimate_whole_meeting の結果）。見積もれなかったときは None。
    whole_meeting: dict | None = None

    def get(self, key: str) -> CheckResult | None:
        return next((c for c in self.checks if c.key == key), None)

    def to_json(self) -> str:
        return json.dumps(
            {
                "base_url": self.base_url,
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
                "unload": self.unload,
                "whole_meeting": self.whole_meeting,
            },
            ensure_ascii=False,
            indent=2,
        )


# --- 入力を作る ------------------------------------------------------------------------------


def sample_chunk() -> Chunk:
    """検査2で使う固定の短いチャンク。"""
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


def doctor_payload(cfg: Config, chunk: Chunk, with_schema: bool) -> dict:
    """抽出が実際に送るのと同じ形のリクエストを作る。ただし keep_alive は "2m" にする。

    extract.ollama_impl.build_payload をそのまま使うのは、doctor が測るものと extract が
    送るものを食い違わせないためである。with_schema は format の有無だけを揃える
    （検査2は設定によらず付けて、スキーマに適合するかを測る）。
    """
    payload = copy.deepcopy(build_payload(cfg, chunk))
    if with_schema:
        payload["format"] = ollama_api.require_all_properties(ExtractionResult.model_json_schema())
    else:
        payload.pop("format", None)
    payload["keep_alive"] = DOCTOR_KEEP_ALIVE
    return payload


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


def _post(transport: HttpTransport, cfg: Config, payload: dict) -> tuple[HttpResponse | None, str | None]:
    """/api/chat に1回投げて (応答, 失敗の理由) を返す。"""
    url = ollama_api.url_of(cfg.extract.ollama, ollama_api.CHAT_PATH)
    try:
        return transport.post(url, payload, cfg.extract.ollama.timeout_sec), None
    except TransportError as e:
        return None, f"{url} に要求を送れませんでした: {e}"


def _reply_record(response: HttpResponse) -> tuple[ollama_api.ChatReply | None, str | None]:
    try:
        return ollama_api.parse_chat_reply(response), None
    except ollama_api.OllamaError as e:
        return None, str(e)


# --- 検査1: 接続とモデルの有無 ----------------------------------------------------------------


def _model_names(body) -> list[str]:
    """/api/tags の応答からモデル名の一覧を取り出す。"""
    entries = body.get("models") if isinstance(body, dict) else None
    names: list[str] = []
    for e in entries or []:
        if isinstance(e, dict):
            for key in ("name", "model"):
                value = e.get(key)
                if isinstance(value, str) and value not in names:
                    names.append(value)
    return names


def _has_model(names: list[str], model: str) -> bool:
    """タグを省いた名前（qwen3）は、Ollama が :latest に解決する。"""
    return model in names or (":" not in model and f"{model}:latest" in names)


def check_models(transport: HttpTransport, cfg: Config) -> CheckResult:
    """検査1: GET /api/tags に設定のモデルがあるか。"""
    title = "1. 接続とモデルの有無"
    url = ollama_api.url_of(cfg.extract.ollama, ollama_api.TAGS_PATH)
    model = cfg.extract.ollama.model
    try:
        response = transport.get(url, timeout_sec=30)
    except TransportError as e:
        return CheckResult(
            "models", title, STATUS_ERROR,
            f"{url} に接続できません（Ollama が起動しているか、タスクトレイを確認してください。"
            f"base_url は数字で書く）: {e}",
            {"url": url},
        )
    if not response.ok:
        return CheckResult(
            "models", title, STATUS_ERROR,
            f"{url} が HTTP {response.status_code} を返しました",
            {"url": url, "status_code": response.status_code, "body_head": response.text[:300]},
        )
    try:
        names = _model_names(response.json())
    except TransportError as e:
        return CheckResult(
            "models", title, STATUS_ERROR,
            f"応答を JSON として読めませんでした。base_url が Ollama を指しているか確認してください: {e}",
            {"url": url, "body_head": response.text[:300]},
        )
    detail = {"url": url, "model_names": names, "configured_model": model}
    if _has_model(names, model):
        return CheckResult(
            "models", title, STATUS_OK,
            f"接続できました。設定の model「{model}」は取得済みです（全 {len(names)} 件）", detail,
        )
    return CheckResult(
        "models", title, STATUS_ERROR,
        f"接続できましたが、設定の model「{model}」は取得されていません。"
        f"`ollama pull {model}` を実行してください。取得済み: {', '.join(names[:10]) or 'なし'}",
        detail,
    )


# --- 検査2: スキーマへの適合 ------------------------------------------------------------------


def check_schema_enforcement(transport: HttpTransport, cfg: Config) -> CheckResult:
    """検査2: format にスキーマを付けて送り、repair を通さずに ExtractionResult として読めるか。"""
    title = "2. スキーマへの適合（format あり）"
    response, error = _post(transport, cfg, doctor_payload(cfg, sample_chunk(), with_schema=True))
    if error is not None:
        return CheckResult("schema", title, STATUS_ERROR, error, {})
    assert response is not None
    if not response.ok:
        return CheckResult(
            "schema", title, STATUS_ERROR,
            f"HTTP {response.status_code} が返りました: {response.text[:200]}",
            {"status_code": response.status_code, "body_head": response.text[:300]},
        )
    reply, problem = _reply_record(response)
    if reply is None:
        return CheckResult("schema", title, STATUS_ERROR, str(problem), {"body_head": response.text[:300]})
    classification, actions, note = classify_output(reply.content)
    detail = {
        "classification": classification,
        "repair_actions": actions,
        "note": note,
        "elapsed_sec": round(response.elapsed_sec, 2),
        "load_sec": None if reply.load_sec is None else round(reply.load_sec, 2),
        "done_reason": reply.done_reason,
        "prompt_eval_count": reply.prompt_eval_count,
        "eval_count": reply.eval_count,
        "eval_sec": None if reply.eval_sec is None else round(reply.eval_sec, 2),
        "thinking_chars": reply.thinking_chars,
        "content_head": reply.content[:300],
    }
    if classification == OUTPUT_DIRECT_JSON:
        return CheckResult("schema", title, STATUS_OK, "出力はスキーマに適合し、そのまま読めました", detail)
    if classification == OUTPUT_REPAIRED:
        return CheckResult(
            "schema", title, STATUS_WARN,
            f"format を付けても出力が適合しません（修復すれば読めます: {note}）。"
            "enforces_json_schema = false を推奨します",
            detail,
        )
    return CheckResult(
        "schema", title, STATUS_ERROR,
        f"修復しても読めませんでした。enforces_json_schema = false を試すか、プロンプトか設定の見直しが"
        f"必要です（{note}）。done_reason = {reply.done_reason}",
        detail,
    )


# --- 検査3: GPU に収まっているか ---------------------------------------------------------------


def _find_loaded(body, model: str) -> dict | None:
    entries = body.get("models") if isinstance(body, dict) else None
    wanted = {model, f"{model}:latest"} if ":" not in model else {model}
    for e in entries or []:
        if isinstance(e, dict) and (e.get("name") in wanted or e.get("model") in wanted):
            return e
    return None


def check_gpu(transport: HttpTransport, cfg: Config) -> CheckResult:
    """検査3: GET /api/ps で、対象モデルの size_vram と size を比べる。

    size は読み込んだモデルの全体（重みと KV キャッシュ）、size_vram はそのうち VRAM にある量。
    等しければ全部が GPU に載っている。小さければ、差の分がメインメモリに溢れていて、遅くなる。
    """
    title = "3. GPU に収まっているか"
    url = ollama_api.url_of(cfg.extract.ollama, ollama_api.PS_PATH)
    model = cfg.extract.ollama.model
    try:
        response = transport.get(url, timeout_sec=30)
    except TransportError as e:
        return CheckResult("gpu", title, STATUS_ERROR, f"{url} に接続できません: {e}", {"url": url})
    if not response.ok:
        return CheckResult(
            "gpu", title, STATUS_ERROR, f"{url} が HTTP {response.status_code} を返しました",
            {"url": url, "status_code": response.status_code, "body_head": response.text[:300]},
        )
    try:
        entry = _find_loaded(response.json(), model)
    except TransportError as e:
        return CheckResult("gpu", title, STATUS_ERROR, f"応答を JSON として読めませんでした: {e}", {"url": url})
    if entry is None:
        return CheckResult(
            "gpu", title, STATUS_WARN,
            f"{model} が読み込まれていません。検査2の直後に読むので、通常はあります。"
            "keep_alive が効いていない可能性があります",
            {"url": url, "body_head": response.text[:300]},
        )
    size, size_vram = entry.get("size"), entry.get("size_vram")
    if not isinstance(size, int) or not isinstance(size_vram, int) or size <= 0:
        return CheckResult(
            "gpu", title, STATUS_WARN, "応答に size と size_vram がありません（Ollama の版が古い可能性があります）",
            {"entry": entry},
        )
    gib = 1024**3
    detail = {
        "size_bytes": size,
        "size_vram_bytes": size_vram,
        "vram_ratio": round(size_vram / size, 4),
        "context_length": entry.get("context_length"),
        "configured_num_ctx": cfg.extract.ollama.num_ctx,
    }
    if size_vram >= size:
        return CheckResult(
            "gpu", title, STATUS_OK,
            f"全体が GPU に載っています（{size / gib:.1f} GiB。size_vram = size）", detail,
        )
    spilled = size - size_vram
    return CheckResult(
        "gpu", title, STATUS_ERROR,
        f"一部がメインメモリに溢れています（全体 {size / gib:.1f} GiB のうち VRAM は {size_vram / gib:.1f} GiB、"
        f"溢れ {spilled / gib:.1f} GiB。VRAM の割合 {size_vram / size:.0%}）。"
        "`extract.ollama.num_ctx` を下げるか、小さいモデルにしてください。"
        "このまま使うと遅くなり、Strata で起きたメモリの圧迫が再発します",
        detail,
    )


# --- 検査4・5: 速度と入力の切り捨て -------------------------------------------------------------


def _measure(transport: HttpTransport, cfg: Config, chunk: Chunk, label: str) -> dict:
    """抽出と同じ形で1回投げて、入力・出力のトークン数と所要時間を測る。"""
    system, user = build_messages(chunk, cfg)
    record: dict = {
        "label": label,
        "utterances": len(chunk.utterances),
        "estimated_input_tokens": estimate_messages_tokens(system, user, cfg.extract.tokens_per_character),
        # 送った文字数。tokens_per_character の実測（prompt_eval_count ÷ この値）に使う。
        "input_chars": len(system) + len(user),
    }
    response, error = _post(
        transport, cfg, doctor_payload(cfg, chunk, with_schema=cfg.extract.ollama.enforces_json_schema)
    )
    if error is not None:
        record["error"] = error
        return record
    assert response is not None
    record["status_code"] = response.status_code
    record["elapsed_sec"] = round(response.elapsed_sec, 2)
    if not response.ok:
        record["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
        return record
    reply, problem = _reply_record(response)
    if reply is None:
        record["error"] = problem
        return record
    record.update(
        {
            "input_tokens": reply.prompt_eval_count,
            "output_tokens": reply.eval_count,
            "done_reason": reply.done_reason,
            "thinking_chars": reply.thinking_chars,
            "load_sec": reply.load_sec,
            "prompt_eval_sec": reply.prompt_eval_sec,
            "total_sec": reply.total_sec,
            "eval_sec": reply.eval_sec,
            "output_tokens_per_sec": None if reply.tokens_per_sec is None else round(reply.tokens_per_sec, 1),
            "classification": classify_output(reply.content)[0],
        }
    )
    return record


def check_speed(transport: HttpTransport, cfg: Config) -> CheckResult:
    """検査4: chunk_minutes 相当の長さで投げ、tokens/s と所要時間を出す。結果を出すだけで、合否は付けない。"""
    title = "4. 速度"
    chunk_tokens = tokens_for_minutes(cfg.extract.chunk_minutes, cfg.extract.tokens_per_meeting_minute)
    measurement = _measure(
        transport, cfg, stretched_chunk(cfg, chunk_tokens),
        f"chunk_minutes={cfg.extract.chunk_minutes} 相当（約{chunk_tokens}トークン）",
    )
    detail = {"measurements": [measurement]}
    if "error" in measurement:
        return CheckResult("speed", title, STATUS_ERROR, measurement["error"], detail)
    speed = measurement.get("output_tokens_per_sec")
    parts = [f"{measurement['label']}: {measurement['elapsed_sec']} 秒"]
    parts.append(
        f"出力 {measurement.get('output_tokens')} トークン、{speed} tokens/s"
        if speed else "出力の速度は Ollama が値を返さないため不明"
    )
    status = STATUS_OK
    if (measurement.get("output_tokens") or 0) < MIN_EVAL_TOKENS_FOR_SPEED:
        status = STATUS_WARN
        parts.append(
            f"出力が {MIN_EVAL_TOKENS_FOR_SPEED} トークン未満と短く、速度は誤差が大きい"
            "（固定の文を繰り返した入力のためと考えられる）"
        )
    return CheckResult("speed", title, status, " / ".join(parts), detail)


def check_truncation(cfg: Config, speed: CheckResult | None) -> CheckResult:
    """検査5: 検査4の prompt_eval_count が num_ctx - num_predict に達していないか。"""
    title = "5. 入力が切り捨てられていないか"
    o = cfg.extract.ollama
    measurement = (speed.detail.get("measurements") or [None])[0] if speed is not None else None
    if measurement is None or "error" in measurement:
        return CheckResult("truncation", title, STATUS_SKIPPED, "検査4が実行できなかったため確認していません", {})
    count = measurement.get("input_tokens")
    limit = ollama_api.input_limit(o)
    if count is None:
        return CheckResult(
            "truncation", title, STATUS_WARN,
            "Ollama が prompt_eval_count を返さなかったため、確認できません", {"limit": limit},
        )
    chars = measurement.get("input_chars")
    detail = {
        "input_tokens": count,
        "limit": limit,
        "num_ctx": o.num_ctx,
        "num_predict": o.num_predict,
        "estimated_input_tokens": measurement.get("estimated_input_tokens"),
        "tokens_per_character_measured": round(count / chars, 4) if chars else None,
    }
    ratio = f"{count / chars:.3f} トークン/文字" if chars else "不明"
    note = (
        f"入力 {count} トークン（上限 num_ctx {o.num_ctx} - num_predict {o.num_predict} = {limit}）。"
        f"概算は {measurement.get('estimated_input_tokens')} トークンで、実測は {ratio}"
        f"（設定の tokens_per_character = {cfg.extract.tokens_per_character}）"
    )
    if count >= limit:
        return CheckResult(
            "truncation", title, STATUS_ERROR,
            f"{note}。上限に達しており、Ollama が入力の先頭を捨てた疑いがあります。"
            "chunk_minutes を下げるか、num_ctx を上げてください",
            detail,
        )
    return CheckResult("truncation", title, STATUS_OK, note, detail)


# --- 推奨値 -----------------------------------------------------------------------------------


def estimate_whole_meeting(cfg: Config, measurement: dict, extra: list[dict]) -> dict | None:
    """2時間の会議の所要時間の見込みを、測定値から2通りで計算する。

    - measured: 検査4の1チャンクの所要時間 × チャンク数。
    - worst: 1チャンクを「入力の処理 + num_predict ÷ 観測された最も遅い生成速度」とした上限。
      最も遅い生成速度は、出力が MIN_EVAL_TOKENS_FOR_SPEED 以上ある測定（検査2・4）の最小。
      Strata のときに、同じ設定でも生成速度が 4.9 倍振れた実績があるため、1回の測定の値では決めない
      （CLAUDE.md 7章）。
    keep_alive が "0" のときは、チャンクごとにモデルの読み込みが入るので、検査2で観測した
    読み込みの時間(load_sec)を足す。
    """
    e = cfg.extract
    stride = e.chunk_minutes - e.overlap_minutes
    chunks = max(1, math.ceil((WHOLE_MEETING_MINUTES - e.overlap_minutes) / stride))
    reload_each = cfg.extract.ollama.keep_alive.strip() == "0"
    loads = [m.get("load_sec") for m in extra if m.get("load_sec")]
    load = max(loads) if (reload_each and loads) else 0.0
    out: dict = {
        "meeting_minutes": WHOLE_MEETING_MINUTES,
        "chunks": chunks,
        "reload_each_chunk": reload_each,
        "load_sec_assumed": round(load, 1),
    }
    elapsed = measurement.get("elapsed_sec")
    if elapsed:
        out["measured_sec"] = round(chunks * (elapsed + load))
    speeds = [
        m["output_tokens"] / m["eval_sec"]
        for m in [measurement, *extra]
        if (m.get("output_tokens") or 0) >= MIN_EVAL_TOKENS_FOR_SPEED and m.get("eval_sec")
    ]
    if speeds and measurement.get("prompt_eval_sec") is not None:
        slowest = min(speeds)
        per_chunk = load + measurement["prompt_eval_sec"] + cfg.extract.ollama.num_predict / slowest
        out["slowest_tokens_per_sec"] = round(slowest, 1)
        out["worst_per_chunk_sec"] = round(per_chunk)
        out["worst_sec"] = round(chunks * per_chunk)
    return out if ("measured_sec" in out or "worst_sec" in out) else None


def _hours(sec: float) -> str:
    return f"{sec / 3600:.1f} 時間" if sec >= 3600 else f"{sec / 60:.0f} 分"


def build_recommendations(cfg: Config, report: DoctorReport) -> dict:
    """実測の結果から、config.toml に書くべき値を出す。測れなかった項目は現在の値のままにする。"""
    o = cfg.extract.ollama
    rec: dict = {}

    schema = report.get("schema")
    if schema is None or "classification" not in schema.detail:
        rec["enforces_json_schema"] = {"value": o.enforces_json_schema, "reason": "測れなかったので現在の値のまま"}
    else:
        enforced = schema.detail.get("classification") == OUTPUT_DIRECT_JSON
        rec["enforces_json_schema"] = {
            "value": enforced,
            "reason": (
                "format ありで、出力がスキーマに適合し、そのまま読めた"
                if enforced
                else f"format を付けても出力が適合しない（分類: {schema.detail.get('classification')}）"
            ),
        }

    speed = report.get("speed")
    measurements = speed.detail.get("measurements", []) if speed is not None else []
    measured = measurements[0] if measurements and "error" not in measurements[0] else None

    if measured and measured.get("input_tokens") and measured.get("input_chars"):
        tokens, chars = measured["input_tokens"], measured["input_chars"]
        value = math.ceil(tokens / chars * 100) / 100
        rec["tokens_per_character"] = {
            "value": value,
            "reason": (
                f"実測値。検査4で送った {chars} 文字（system と user の合計）に対し、Ollama が数えた入力は "
                f"{tokens} トークン（prompt_eval_count）で、{tokens / chars:.3f} トークン/文字。"
                "小数第2位に切り上げた。固定の文を繰り返した入力からの値なので、"
                "実際の会議の文字起こしでは多少ずれる可能性がある"
            ),
        }

    schema_detail = schema.detail if schema is not None else {}
    first = {"load_sec": schema_detail.get("load_sec")}
    if measured and measured.get("elapsed_sec"):
        whole = estimate_whole_meeting(cfg, measured, [first])
        report.whole_meeting = whole
        if whole and "worst_per_chunk_sec" in whole:
            suggested = max(
                TIMEOUT_FLOOR_SEC,
                int(math.ceil(whole["worst_per_chunk_sec"] * TIMEOUT_GENERATION_SAFETY / 60) * 60),
            )
            reason = (
                "timeout_sec = 1チャンクの最悪値（モデルの読み込み + 入力の処理 + num_predict ÷ 観測された"
                f"最も遅い生成速度 {whole['slowest_tokens_per_sec']} tokens/s）× {TIMEOUT_GENERATION_SAFETY}"
                f"（下限 {TIMEOUT_FLOOR_SEC} 秒、分単位に切り上げ）。最悪値は {whole['worst_per_chunk_sec']} 秒"
            )
        else:
            suggested = max(TIMEOUT_FLOOR_SEC, int(math.ceil(measured["elapsed_sec"] * 3 / 60) * 60))
            reason = (
                f"実測した所要時間 {measured['elapsed_sec']} 秒の3倍（下限 {TIMEOUT_FLOOR_SEC} 秒、分単位に切り上げ）。"
                "生成速度を出せるだけの出力が無かったため、最悪値は計算していない"
            )
        rec["timeout_sec"] = {"value": suggested, "reason": reason}
    else:
        rec["timeout_sec"] = {"value": o.timeout_sec, "reason": "測れなかったので現在の値のまま"}

    truncation = report.get("truncation")
    if truncation is not None and truncation.status == STATUS_ERROR:
        rec["chunk_minutes"] = {
            "value": max(1, cfg.extract.chunk_minutes // 2),
            "reason": "検査5で入力が上限に達したため、半分にした。再度 doctor を実行して確かめること",
        }
    return rec


# --- 実行 -------------------------------------------------------------------------------------


def unload_model(transport: HttpTransport, cfg: Config) -> dict:
    """モデルを VRAM から下ろす（POST /api/generate、keep_alive 0）。結果を {"ok", "detail"} で返す。"""
    ok, detail = ollama_api.unload_model(transport, cfg.extract.ollama)
    return {"ok": ok, "detail": detail}


def run_doctor(transport: HttpTransport, cfg: Config, now: datetime | None = None) -> DoctorReport:
    """検査1から5を順に実行する。検査1が失敗したときは残りを実行しない。最後にモデルを下ろす。"""
    report = DoctorReport(
        base_url=cfg.extract.ollama.base_url,
        model=cfg.extract.ollama.model,
        started_at=now or datetime.now(),
    )
    models = check_models(transport, cfg)
    report.checks.append(models)
    if models.status == STATUS_ERROR:
        for key, title in (
            ("schema", "2. スキーマへの適合（format あり）"),
            ("gpu", "3. GPU に収まっているか"),
            ("speed", "4. 速度"),
            ("truncation", "5. 入力が切り捨てられていないか"),
        ):
            report.checks.append(
                CheckResult(key, title, STATUS_SKIPPED, "検査1が失敗したため実行していません")
            )
    else:
        schema = check_schema_enforcement(transport, cfg)
        report.checks.append(schema)
        if "classification" not in schema.detail:
            # 検査2の応答が得られていない(接続失敗、HTTP エラー等)ので、モデルは読み込まれていない
            report.checks.append(
                CheckResult("gpu", "3. GPU に収まっているか", STATUS_SKIPPED, "検査2が応答を得られなかったため実行していません")
            )
        else:
            report.checks.append(check_gpu(transport, cfg))
        speed = check_speed(transport, cfg)
        report.checks.append(speed)
        report.checks.append(check_truncation(cfg, speed))
        report.unload = unload_model(transport, cfg)
    report.recommendations = build_recommendations(cfg, report)
    return report


def format_report(report: DoctorReport) -> str:
    """結果を表にする。

    説明文は長くなるので、表には「検査」と「結果」だけを入れ、内容はその下に字下げして並べる
    （1列が極端に広い表は端末で読みにくいため）。
    """
    lines = [
        f"Ollama の実測: {report.base_url}（model = {report.model}）",
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
    whole = report.whole_meeting
    if whole:
        lines.append(f"{whole['meeting_minutes']} 分の会議の所要時間の見込み（{whole['chunks']} チャンク）:")
        if "measured_sec" in whole:
            lines.append(
                f"  検査4の実測から: 約{_hours(whole['measured_sec'])}"
                + (f"（チャンクごとにモデルの読み込み約{whole['load_sec_assumed']}秒を含む）" if whole["reload_each_chunk"] else "")
            )
        if "worst_sec" in whole:
            lines.append(
                f"  最も遅い測定値（{whole['slowest_tokens_per_sec']} tokens/s）で num_predict まで生成した最悪値: "
                f"約{_hours(whole['worst_sec'])}"
            )
        else:
            lines.append("  最悪値は、生成速度を出せるだけの出力が無かったため計算できません")
        lines.append("")
    if report.unload is not None:
        lines.append(f"モデルの解放: {report.unload['detail']}")
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
        "[extract] の chunk_minutes・tokens_per_character、"
        "[extract.ollama] の enforces_json_schema・timeout_sec に書いてください。"
    )
    return "\n".join(lines)


def report_path(cfg: Config, at: datetime) -> Path:
    """結果の JSON の保存先。"""
    return cfg.paths.db.parent / "doctor" / f"ollama_{at:%Y%m%d_%H%M%S}.json"
