"""kijun doctor(Ollama の実測)のテスト。Ollama には接続せず、偽の HTTP 層に決めた応答を返させる。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from kijun import ollama_api
from kijun.config import Config
from kijun.doctor import (
    DOCTOR_KEEP_ALIVE,
    OUTPUT_DIRECT_JSON,
    OUTPUT_REPAIRED,
    OUTPUT_UNREADABLE,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_WARN,
    build_recommendations,
    check_gpu,
    check_models,
    check_schema_enforcement,
    check_speed,
    check_truncation,
    classify_output,
    doctor_payload,
    estimate_whole_meeting,
    format_report,
    report_path,
    run_doctor,
    sample_chunk,
    stretched_chunk,
    unload_model,
)
from kijun.llm.chat import OllamaJsonChat
from kijun.llm.tokens import tokens_for_minutes
from kijun.minutes.models import ChunkMinutes
from kijun.minutes.prompt import build_chunk_messages
from kijun.transport import HttpResponse, TransportError
from tests.fakes import ollama_body

GOOD_JSON = json.dumps({"topics": [], "decisions": [], "todos": [], "open_issues": []})
GIB = 1024**3


def tags(*names: str) -> str:
    return json.dumps({"models": [{"name": n, "model": n} for n in names]})


def ps(model: str = "qwen3.5:9b", size: int = 9 * GIB, size_vram: int = 9 * GIB, **extra) -> str:
    return json.dumps({"models": [{"name": model, "model": model, "size": size, "size_vram": size_vram, **extra}]})


class FakeServer:
    """URL の末尾で応答を振り分ける偽の HttpTransport。呼ばれた順序を log に残す。

    chat は /api/chat への POST の応答を順に返す(足りなければ最後を繰り返す)。
    """

    def __init__(self, tags_response=None, ps_response=None, chat=None, generate=None, elapsed_sec: float = 8.0) -> None:
        self.tags_response = tags_response if tags_response is not None else (200, tags("qwen3.5:9b"))
        self.ps_response = ps_response if ps_response is not None else (200, ps())
        self.chat = list(chat if chat is not None else [(200, ollama_body(GOOD_JSON))])
        self.generate = generate if generate is not None else (200, "{}")
        self.elapsed_sec = elapsed_sec
        self.log: list[str] = []
        self.posts: list[dict] = []

    def _respond(self, spec) -> HttpResponse:
        if isinstance(spec, Exception):
            raise spec
        status, text = spec
        return HttpResponse(status, text, self.elapsed_sec)

    def get(self, url: str, timeout_sec: float) -> HttpResponse:
        if url.endswith("/api/tags"):
            self.log.append("GET tags")
            return self._respond(self.tags_response)
        if url.endswith("/api/ps"):
            self.log.append("GET ps")
            return self._respond(self.ps_response)
        raise AssertionError(url)

    def post(self, url: str, payload: dict, timeout_sec: float) -> HttpResponse:
        if url.endswith("/api/generate"):
            self.log.append("POST generate")
            self.posts.append({"url": url, "payload": payload})
            return self._respond(self.generate)
        assert url.endswith("/api/chat"), url
        self.log.append("POST chat")
        self.posts.append({"url": url, "payload": payload})
        spec = self.chat[0] if len(self.chat) == 1 else self.chat.pop(0)
        return self._respond(spec)


# --- 入力を作る ---------------------------------------------------------------------------------


def test_sample_chunk_は固定の3発話():
    chunk = sample_chunk()
    assert len(chunk.utterances) == 3 and chunk.seq_from == 0 and chunk.seq_to == 2


def test_stretched_chunk_は指定したトークン数に近い長さになる():
    cfg = Config()
    chunk = stretched_chunk(cfg, 2000)
    chars = sum(len(u.text) for u in chunk.utterances)
    assert abs(chars * cfg.minutes.tokens_per_character - 2000) < 120 * cfg.minutes.tokens_per_character
    assert chunk.seq_to == len(chunk.utterances) - 1


def test_doctor_payload_は議事録の生成が送るものと_keep_alive_だけが違う():
    cfg = Config()
    chunk = sample_chunk()
    system, user = build_chunk_messages(chunk)
    extract = OllamaJsonChat(cfg.llm, object()).build_payload(system, user, ChunkMinutes)
    doctor = doctor_payload(cfg, chunk, with_schema=True)
    assert extract["keep_alive"] == "0" and doctor["keep_alive"] == DOCTOR_KEEP_ALIVE == "2m"
    assert {**doctor, "keep_alive": "0"} == extract
    # format の有無は引数で決まる(設定の enforces_json_schema によらない)
    cfg.llm.enforces_json_schema = False
    assert "format" in doctor_payload(cfg, chunk, with_schema=True)
    cfg.llm.enforces_json_schema = True
    assert "format" not in doctor_payload(cfg, chunk, with_schema=False)


def test_classify_output_の3分類():
    assert classify_output(GOOD_JSON)[0] == OUTPUT_DIRECT_JSON
    assert classify_output("結果です\n" + GOOD_JSON)[0] == OUTPUT_REPAIRED
    assert classify_output("読めない")[0] == OUTPUT_UNREADABLE


# --- 検査1: 接続とモデルの有無 -------------------------------------------------------------------


def test_検査1_モデルがあれば良():
    r = check_models(FakeServer(), Config())
    assert r.status == STATUS_OK and "qwen3.5:9b" in r.summary


def test_検査1_タグを省いた名前は_latest_に解決する():
    cfg = Config()
    cfg.llm.model = "qwen3"
    assert check_models(FakeServer(tags_response=(200, tags("qwen3:latest"))), cfg).status == STATUS_OK


def test_検査1_モデルが無ければ_ollama_pull_を案内する():
    r = check_models(FakeServer(tags_response=(200, tags("llama3:8b"))), Config())
    assert r.status == STATUS_ERROR
    assert "ollama pull qwen3.5:9b" in r.summary and "llama3:8b" in r.summary


def test_検査1_接続できなければ失敗():
    r = check_models(FakeServer(tags_response=TransportError("接続拒否")), Config())
    assert r.status == STATUS_ERROR and "接続できません" in r.summary and "タスクトレイ" in r.summary


def test_検査1_HTTP_エラーと_JSON_でない応答は失敗():
    assert check_models(FakeServer(tags_response=(404, "not found")), Config()).status == STATUS_ERROR
    r = check_models(FakeServer(tags_response=(200, "<html>")), Config())
    assert r.status == STATUS_ERROR and "base_url" in r.summary


# --- 検査2: スキーマへの適合 ---------------------------------------------------------------------


def test_検査2_適合すれば良_format_と_keep_alive_と_num_ctx_を付けて送る():
    server = FakeServer(chat=[(200, ollama_body(GOOD_JSON))])
    cfg = Config()
    cfg.llm.enforces_json_schema = False  # 設定が偽でも、検査2は format を付けて測る
    r = check_schema_enforcement(server, cfg)
    assert r.status == STATUS_OK and r.detail["classification"] == OUTPUT_DIRECT_JSON
    payload = server.posts[0]["payload"]
    assert "format" in payload and payload["keep_alive"] == "2m"
    assert payload["options"]["num_ctx"] == cfg.llm.num_ctx


def test_検査2_前置きがあれば注意_enforces_json_schema_を_false_にする推奨():
    r = check_schema_enforcement(FakeServer(chat=[(200, ollama_body("結果です\n" + GOOD_JSON))]), Config())
    assert r.status == STATUS_WARN and r.detail["classification"] == OUTPUT_REPAIRED
    assert "enforces_json_schema = false" in r.summary


def test_検査2_読めなければ失敗(capsys):
    r = check_schema_enforcement(FakeServer(chat=[(200, ollama_body("無理です", done_reason="length"))]), Config())
    assert r.status == STATUS_ERROR and r.detail["classification"] == OUTPUT_UNREADABLE
    assert "length" in r.summary


def test_検査2_HTTP_エラーと接続不可と形の違う応答():
    assert check_schema_enforcement(FakeServer(chat=[(400, "bad")]), Config()).status == STATUS_ERROR
    assert check_schema_enforcement(FakeServer(chat=[TransportError("切断")]), Config()).status == STATUS_ERROR
    r = check_schema_enforcement(FakeServer(chat=[(200, json.dumps({"x": 1}))]), Config())
    assert r.status == STATUS_ERROR and "/api/chat の形" in r.summary
    # 応答が得られなかった検査は classification を持たない(検査3を飛ばす判断に使う)
    assert "classification" not in r.detail


# --- 検査3: GPU に収まっているか -----------------------------------------------------------------


def test_検査3_size_vram_が_size_と等しければ良():
    r = check_gpu(FakeServer(), Config())
    assert r.status == STATUS_OK and "GPU に載っています" in r.summary and r.detail["vram_ratio"] == 1.0


def test_検査3_溢れていれば失敗_num_ctx_を下げるよう案内する():
    r = check_gpu(FakeServer(ps_response=(200, ps(size=12 * GIB, size_vram=9 * GIB))), Config())
    assert r.status == STATUS_ERROR
    assert "メインメモリに溢れています" in r.summary and "75%" in r.summary
    assert "num_ctx" in r.summary and "小さいモデル" in r.summary
    assert r.detail["size_bytes"] == 12 * GIB and r.detail["size_vram_bytes"] == 9 * GIB


def test_検査3_全部が_CPU_側にある場合も失敗():
    r = check_gpu(FakeServer(ps_response=(200, ps(size_vram=0))), Config())
    assert r.status == STATUS_ERROR and "0%" in r.summary


def test_検査3_別のモデルだけが読み込まれていれば注意():
    r = check_gpu(FakeServer(ps_response=(200, ps(model="other:1b"))), Config())
    assert r.status == STATUS_WARN and "読み込まれていません" in r.summary


def test_検査3_size_が無い古い版は注意():
    body = json.dumps({"models": [{"name": "qwen3.5:9b"}]})
    assert check_gpu(FakeServer(ps_response=(200, body)), Config()).status == STATUS_WARN


def test_検査3_接続不可と_HTTP_エラーは失敗():
    assert check_gpu(FakeServer(ps_response=TransportError("x")), Config()).status == STATUS_ERROR
    assert check_gpu(FakeServer(ps_response=(500, "x")), Config()).status == STATUS_ERROR


def test_検査3_タグを省いたモデル名は_latest_として探す():
    cfg = Config()
    cfg.llm.model = "qwen3"
    assert check_gpu(FakeServer(ps_response=(200, ps(model="qwen3:latest"))), cfg).status == STATUS_OK


# --- 検査4: 速度 -----------------------------------------------------------------------------------


def speed_body(**kw) -> str:
    kw.setdefault("eval_count", 600)
    kw.setdefault("eval_sec", 20.0)
    kw.setdefault("prompt_eval_count", 2400)
    return ollama_body(GOOD_JSON, **kw)


def test_検査4_chunk_minutes_相当の入力で_tokens_per_sec_を出す():
    cfg = Config()
    server = FakeServer(chat=[(200, speed_body())], elapsed_sec=30.0)
    r = check_speed(server, cfg)
    assert r.status == STATUS_OK
    m = r.detail["measurements"][0]
    assert m["output_tokens_per_sec"] == 30.0 and m["input_tokens"] == 2400 and m["output_tokens"] == 600
    assert m["elapsed_sec"] == 30.0 and "30.0 tokens/s" in r.summary
    # 送った入力は chunk_minutes 相当の長さで、keep_alive は 2m
    sent = server.posts[0]["payload"]
    assert sent["keep_alive"] == "2m"
    user_chars = len(sent["messages"][1]["content"])
    expected = tokens_for_minutes(cfg.minutes.chunk_minutes, cfg.minutes.tokens_per_meeting_minute) / cfg.minutes.tokens_per_character
    assert abs(user_chars - expected) < expected * 0.3


def test_検査4_出力が短ければ注意_速度は誤差が大きい():
    r = check_speed(FakeServer(chat=[(200, speed_body(eval_count=10, eval_sec=0.5))]), Config())
    assert r.status == STATUS_WARN and "誤差が大きい" in r.summary


def test_検査4_失敗は失敗():
    assert check_speed(FakeServer(chat=[(500, "x")]), Config()).status == STATUS_ERROR
    assert check_speed(FakeServer(chat=[TransportError("x")]), Config()).status == STATUS_ERROR


# --- 検査5: 入力の切り捨て -----------------------------------------------------------------------


def speed_result(**kw):
    return check_speed(FakeServer(chat=[(200, speed_body(**kw))]), Config())


def test_検査5_上限未満なら良_tokens_per_character_の実測を出す():
    cfg = Config()
    speed = speed_result(prompt_eval_count=3000)
    r = check_truncation(cfg, speed)
    assert r.status == STATUS_OK
    limit = cfg.llm.num_ctx - cfg.llm.num_predict
    assert r.detail["limit"] == limit and r.detail["input_tokens"] == 3000
    chars = speed.detail["measurements"][0]["input_chars"]
    assert r.detail["tokens_per_character_measured"] == round(3000 / chars, 4)
    assert "トークン/文字" in r.summary


def test_検査5_上限に達していれば失敗():
    cfg = Config()
    limit = cfg.llm.num_ctx - cfg.llm.num_predict
    r = check_truncation(cfg, speed_result(prompt_eval_count=limit))
    assert r.status == STATUS_ERROR and "先頭を捨てた疑い" in r.summary
    assert check_truncation(cfg, speed_result(prompt_eval_count=limit - 1)).status == STATUS_OK


def test_検査5_prompt_eval_count_が無ければ注意():
    r = check_truncation(Config(), speed_result(prompt_eval_count=None))
    assert r.status == STATUS_WARN


def test_検査5_検査4が失敗していれば未実行():
    bad = check_speed(FakeServer(chat=[(500, "x")]), Config())
    assert check_truncation(Config(), bad).status == STATUS_SKIPPED
    assert check_truncation(Config(), None).status == STATUS_SKIPPED


# --- 所要時間の見込み -----------------------------------------------------------------------------


def measurement(**kw) -> dict:
    base = {
        "elapsed_sec": 40.0, "output_tokens": 600, "eval_sec": 20.0, "prompt_eval_sec": 3.0, "load_sec": 4.0,
    }
    return {**base, **kw}


def test_2時間の会議の見込み_チャンク数と実測から計算する():
    cfg = Config()  # chunk 15 分・重なり 0 → 120 / 15 = 8 チャンク。概要の1回を足して 9 回呼ぶ
    whole = estimate_whole_meeting(cfg, measurement(), [{"load_sec": 5.0}])
    assert whole["chunks"] == 8 and whole["calls"] == 9 and whole["reload_each_chunk"] is True and whole["load_sec_assumed"] == 5.0
    assert whole["measured_sec"] == round(9 * (40.0 + 5.0))  # keep_alive が "0" なので毎チャンク読み込む
    # 最悪値: 読み込み 5 + 入力 3 + num_predict 8192 ÷ 30 tokens/s
    assert whole["slowest_tokens_per_sec"] == 30.0
    per_chunk = 5.0 + 3.0 + 8192 / 30.0
    assert whole["worst_per_chunk_sec"] == round(per_chunk) and whole["worst_sec"] == round(9 * per_chunk)


def test_最も遅い測定値を使う():
    # 検査2の測定の方が遅ければ、そちらを採る
    slow = {"output_tokens": 100, "eval_sec": 10.0, "load_sec": 1.0}  # 10 tokens/s
    whole = estimate_whole_meeting(Config(), measurement(), [slow])
    assert whole["slowest_tokens_per_sec"] == 10.0


def test_出力が短い測定は速度に使わない():
    short = {"output_tokens": 10, "eval_sec": 5.0, "load_sec": 1.0}  # 2 tokens/s だが誤差が大きい
    whole = estimate_whole_meeting(Config(), measurement(), [short])
    assert whole["slowest_tokens_per_sec"] == 30.0


def test_keep_alive_が_0_でなければ読み込みを足さない():
    cfg = Config()
    cfg.llm.keep_alive = "5m"
    whole = estimate_whole_meeting(cfg, measurement(), [{"load_sec": 5.0}])
    assert whole["reload_each_chunk"] is False and whole["measured_sec"] == round(9 * 40.0)


def test_速度を出せるだけの出力が無ければ最悪値は出さない():
    whole = estimate_whole_meeting(Config(), measurement(output_tokens=10), [])
    assert "measured_sec" in whole and "worst_sec" not in whole


# --- 推奨値 ---------------------------------------------------------------------------------------


def full_report(chat=None, **server_kw):
    server = FakeServer(chat=chat or [(200, ollama_body(GOOD_JSON)), (200, speed_body())], **server_kw)
    return run_doctor(server, Config(), now=datetime(2026, 11, 5, 23, 30)), server


def test_推奨値_enforces_json_schema_は検査2の結果に従う():
    report, _ = full_report()
    assert report.recommendations["enforces_json_schema"]["value"] is True
    report, _ = full_report(chat=[(200, ollama_body("結果です\n" + GOOD_JSON)), (200, speed_body())])
    assert report.recommendations["enforces_json_schema"]["value"] is False


def test_推奨値_測れなかった項目は現在の値のまま():
    cfg = Config()
    cfg.llm.enforces_json_schema = False
    report, _ = full_report(tags_response=(200, tags("x")))  # 検査1が失敗し、以降は未実行
    assert report.recommendations["enforces_json_schema"]["value"] is True  # Config() の値
    assert report.recommendations["timeout_sec"]["value"] == 1800
    assert "tokens_per_character" not in report.recommendations


def test_推奨値_tokens_per_character_は実測を切り上げる():
    report, _ = full_report(chat=[(200, ollama_body(GOOD_JSON)), (200, speed_body(prompt_eval_count=2400))])
    chars = report.get("speed").detail["measurements"][0]["input_chars"]
    import math

    assert report.recommendations["tokens_per_character"]["value"] == math.ceil(2400 / chars * 100) / 100


def test_推奨値_timeout_sec_は最悪値の1_5倍_下限600秒():
    report, _ = full_report()
    whole = report.whole_meeting
    import math

    expected = max(600, math.ceil(whole["worst_per_chunk_sec"] * 1.5 / 60) * 60)
    assert report.recommendations["timeout_sec"]["value"] == expected
    assert "最も遅い生成速度" in report.recommendations["timeout_sec"]["reason"]


def test_推奨値_検査5が失敗したら_chunk_minutes_を半分にする():
    limit = Config().llm.num_ctx - Config().llm.num_predict
    report, _ = full_report(chat=[(200, ollama_body(GOOD_JSON)), (200, speed_body(prompt_eval_count=limit))])
    assert report.get("truncation").status == STATUS_ERROR
    assert report.recommendations["chunk_minutes"]["value"] == Config().minutes.chunk_minutes // 2


def test_推奨値_出力が短ければ測定の3倍を勧め_最悪値は計算しない():
    report, _ = full_report(chat=[(200, ollama_body(GOOD_JSON, eval_count=5)), (200, speed_body(eval_count=5))], elapsed_sec=50.0)
    assert report.recommendations["timeout_sec"]["value"] == 600  # 50 × 3 = 150 → 下限 600
    assert "最悪値は計算していない" in report.recommendations["timeout_sec"]["reason"]


# --- 実行 -----------------------------------------------------------------------------------------


def test_run_doctor_は検査2の直後に_ps_を読み_最後にモデルを下ろす():
    report, server = full_report()
    assert [c.key for c in report.checks] == ["models", "schema", "gpu", "speed", "truncation"]
    # 検査3のために、モデルが VRAM にある間(検査2の直後)に /api/ps を読む
    assert server.log == ["GET tags", "POST chat", "GET ps", "POST chat", "POST generate"]
    unload = server.posts[-1]["payload"]
    assert unload == {"model": "qwen3.5:9b", "keep_alive": 0}
    assert report.unload == {"ok": True, "detail": "モデルを下ろす要求を送りました"}
    assert all(c.status in (STATUS_OK, STATUS_WARN) for c in report.checks)


def test_検査1が失敗したら残りは実行せず_モデルも下ろさない():
    report, server = full_report(tags_response=TransportError("接続拒否"))
    assert [c.status for c in report.checks] == [STATUS_ERROR, STATUS_SKIPPED, STATUS_SKIPPED, STATUS_SKIPPED, STATUS_SKIPPED]
    assert server.log == ["GET tags"] and report.unload is None


def test_検査2が応答を得られなければ検査3は飛ばすが_検査4は続ける():
    report, server = full_report(chat=[(500, "x"), (200, speed_body())])
    assert report.get("schema").status == STATUS_ERROR
    assert report.get("gpu").status == STATUS_SKIPPED
    assert report.get("speed").status == STATUS_OK
    assert "GET ps" not in server.log


def test_モデルを下ろす要求の失敗は結果に残す():
    assert unload_model(FakeServer(generate=(500, "x")), Config())["ok"] is False
    assert unload_model(FakeServer(generate=TransportError("切断")), Config())["ok"] is False


def test_GPU_に溢れていれば報告の検査3が失敗になる():
    report, _ = full_report(ps_response=(200, ps(size=12 * GIB, size_vram=6 * GIB)))
    assert report.get("gpu").status == STATUS_ERROR


def test_結果の_JSON_とファイル名():
    report, _ = full_report()
    data = json.loads(report.to_json())
    assert data["model"] == "qwen3.5:9b" and len(data["checks"]) == 5
    assert data["unload"]["ok"] is True and data["whole_meeting"]["chunks"] == 8
    cfg = Config()
    path = report_path(cfg, datetime(2026, 11, 5, 23, 30, 15))
    assert path.name == "ollama_20261105_233015.json" and path.parent.name == "doctor"


def test_報告の表示には検査と推奨値と所要時間の見込みが出る():
    report, _ = full_report()
    text = format_report(report)
    assert "Ollama の実測" in text and "qwen3.5:9b" in text
    for title in ("1. 接続とモデルの有無", "2. スキーマへの適合", "3. GPU に収まっているか", "4. 速度", "5. 入力が切り捨てられていないか"):
        assert title in text
    assert "config.toml に書くべき推奨値" in text and "enforces_json_schema" in text
    assert "120 分の会議の所要時間の見込み" in text and "最悪値" in text
    assert "モデルの解放" in text
    assert "strata" not in text.lower()


def test_build_recommendations_は検査が1つも無くても落ちない():
    from kijun.doctor import DoctorReport

    report = DoctorReport("http://x", "m", datetime(2026, 11, 5))
    rec = build_recommendations(Config(), report)
    assert rec["enforces_json_schema"]["value"] is True and rec["timeout_sec"]["value"] == 1800


def test_ollama_api_の_url_は末尾のスラッシュを除く():
    cfg = Config().llm
    cfg.base_url = "http://127.0.0.1:11434/"
    assert ollama_api.url_of(cfg, ollama_api.PS_PATH) == "http://127.0.0.1:11434/api/ps"


@pytest.mark.parametrize("count,expected", [(8191, False), (8192, True), (9000, True), (None, False)])
def test_入力の切り捨ての判定は_num_ctx_から_num_predict_を引いた値(count, expected):
    cfg = Config().llm  # num_ctx 16384 - num_predict 8192 = 8192
    reply = ollama_api.ChatReply("", 0, "stop", count, 1, 1.0, 1.0, None, None, 1.0)
    assert ollama_api.input_was_truncated(cfg, reply) is expected
