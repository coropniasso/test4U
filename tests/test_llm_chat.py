"""Ollama への問い合わせ（llm/chat.py）。偽の HttpTransport で、リクエストの形と失敗の扱いを検査する。"""

from __future__ import annotations

import json

import pytest

from kijun.config import Config
from kijun.llm.chat import LlmError, OllamaJsonChat
from kijun.minutes.models import ChunkMinutes, Overview
from kijun.ollama_api import require_all_properties
from kijun.transport import TransportError
from tests.fakes import FakeOllamaTransport, ollama_body

OK_BODY = json.dumps({"topics": [], "decisions": ["水曜日にする"], "todos": [], "open_issues": []}, ensure_ascii=False)
LIMIT = Config().llm.num_ctx - Config().llm.num_predict  # 8192


def ok(content: str, **kw) -> tuple[int, str]:
    return 200, ollama_body(content, **kw)


def chat(contents, cfg=None):
    cfg = cfg or Config()
    transport = FakeOllamaTransport(contents)
    return OllamaJsonChat(cfg.llm, transport), transport


def ask(c):
    return c.ask("S", "U", ChunkMinutes, "試験")


def test_リクエストの形():
    c, transport = chat([ok(OK_BODY)])
    assert ask(c).decisions == ["水曜日にする"]

    post = transport.posts[0]
    # OpenAI 互換の /v1/chat/completions ではなく、num_ctx と keep_alive を送れるネイティブ API を使う
    assert post["url"] == "http://127.0.0.1:11434/api/chat"
    assert post["timeout_sec"] == 1800
    payload = post["payload"]
    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "0"
    assert payload["options"] == {"temperature": 0, "num_ctx": 16384, "num_predict": 8192}
    # 既定値のある項目もモデルに必ず出させるため、全項目を required にしたスキーマを送る
    assert payload["format"] == require_all_properties(ChunkMinutes.model_json_schema())
    assert "think" not in payload  # None は送らない(モデルの既定に任せる)
    assert payload["messages"] == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert c.last_reply is not None and c.last_reply.eval_count == 20


def test_スキーマ適合を保証しないと設定したときは_format_を付けない():
    cfg = Config()
    cfg.llm.enforces_json_schema = False
    c, transport = chat([ok(OK_BODY)], cfg)
    ask(c)
    assert "format" not in transport.posts[0]["payload"]


@pytest.mark.parametrize("value", [True, False])
def test_think_は設定したときだけ送る(value):
    cfg = Config()
    cfg.llm.think = value
    c, transport = chat([ok(OK_BODY)], cfg)
    ask(c)
    assert transport.posts[0]["payload"]["think"] is value


def test_設定の値を送る_base_url_num_ctx_keep_alive():
    cfg = Config()
    cfg.llm.base_url = "http://10.0.0.5:11434/"
    cfg.llm.num_ctx = 32768
    cfg.llm.keep_alive = "5m"
    c, transport = chat([ok(OK_BODY)], cfg)
    ask(c)
    assert transport.posts[0]["url"] == "http://10.0.0.5:11434/api/chat"
    assert transport.posts[0]["payload"]["options"]["num_ctx"] == 32768
    assert transport.posts[0]["payload"]["keep_alive"] == "5m"


def test_前置きつきの出力を修復して読み_標準エラーに出す(capsys):
    c, _ = chat([ok("はい、まとめました。\n" + OK_BODY)])
    assert ask(c).decisions == ["水曜日にする"]
    assert "出力を修復しました" in capsys.readouterr().err


def test_入力が上限に達したら_投げ直さずに失敗する():
    c, transport = chat([ok(OK_BODY, prompt_eval_count=LIMIT)])
    with pytest.raises(LlmError) as e:
        ask(c)
    assert "先頭を捨てる" in str(e.value) and "minutes.chunk_minutes" in str(e.value)
    assert len(transport.posts) == 1  # 投げ直しても同じなので再試行しない


def test_入力が上限に達する直前なら読める():
    c, _ = chat([ok(OK_BODY, prompt_eval_count=LIMIT - 1)])
    assert ask(c).decisions == ["水曜日にする"]


def test_prompt_eval_count_が返らなければ確認できないことを出して続ける(capsys):
    c, _ = chat([ok(OK_BODY, prompt_eval_count=None)])
    assert ask(c).decisions == ["水曜日にする"]
    assert "確認できません" in capsys.readouterr().err


def test_num_predict_で切れたら_投げ直さずに失敗する_途中の_JSON_は読まない():
    c, transport = chat([ok(OK_BODY[:30], done_reason="length")])
    with pytest.raises(LlmError) as e:
        ask(c)
    assert "num_predict" in str(e.value) and "llm.think = false" in str(e.value)
    assert len(transport.posts) == 1


def test_読めない出力は1回だけ投げ直す(capsys):
    c, transport = chat([ok("すみません"), ok(OK_BODY)])
    assert ask(c).decisions == ["水曜日にする"]
    assert len(transport.posts) == 2
    assert transport.posts[0]["payload"] == transport.posts[1]["payload"]  # 同じリクエストを送る
    assert "読めませんでした（1 回目" in capsys.readouterr().err


def test_2回とも読めなければ_LlmError(capsys):
    c, transport = chat([ok("すみません")])
    with pytest.raises(LlmError) as e:
        ask(c)
    assert "2 回とも読めませんでした" in str(e.value) and len(transport.posts) == 2


def test_content_が_null_なら読めない出力として扱う():
    c, _ = chat([ok(None)])
    with pytest.raises(LlmError):
        ask(c)


def test_スキーマに合わない_JSON_は読めない出力として扱う():
    c, _ = chat([ok('{"overview": 1}')])
    with pytest.raises(LlmError):
        c.ask("S", "U", Overview, "概要")


def test_接続できなければ_起動とモデルと_doctor_を案内する():
    c, _ = chat([TransportError("接続拒否")])
    with pytest.raises(LlmError) as e:
        ask(c)
    msg = str(e.value)
    assert "接続拒否" in msg and "ollama list" in msg and "kijun doctor" in msg


def test_HTTP_404_はモデルの取得を案内する():
    c, _ = chat([(404, '{"error":"model not found"}')])
    with pytest.raises(LlmError) as e:
        ask(c)
    assert "ollama pull qwen3.5:9b" in str(e.value)


def test_HTTP_400_で_think_が理由なら_think_を外すよう案内する():
    cfg = Config()
    cfg.llm.think = False
    c, _ = chat([(400, '{"error":"\\"qwen\\" does not support thinking"}')], cfg)
    with pytest.raises(LlmError) as e:
        ask(c)
    assert "llm.think" in str(e.value)


def test_think_を送っていなければ_400_でも_think_を外す案内は出さない():
    c, _ = chat([(400, '{"error":"thinking is bad"}')])
    with pytest.raises(LlmError) as e:
        ask(c)
    assert "llm.think" not in str(e.value) and "HTTP 400" in str(e.value)


def test_応答の形が_Ollama_の_api_chat_でなければ_base_url_を疑うよう案内する():
    c, _ = chat([(200, '{"choices": []}')])
    with pytest.raises(LlmError) as e:
        ask(c)
    assert "llm.base_url" in str(e.value)


def test_応答が_JSON_でなければ_LlmError():
    c, _ = chat([(200, "<html>")])
    with pytest.raises(LlmError):
        ask(c)
