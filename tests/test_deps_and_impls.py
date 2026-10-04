"""重い依存を使う実装クラスが、依存が無くても import でき、入れ方を示すエラーを出すことを検査する。

また、偽のモジュールを差し込んで、Ollama と Claude の実装が同一のプロンプトを使うこと、
リクエストの形が計画書どおりであることを検査する（実際の通信はしない）。
"""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from kijun.config import Config
from kijun.deps import MissingDependencyError, import_optional
from kijun.extract.base import ExtractionError
from kijun.extract.chunk import split_into_chunks
from kijun.extract.models import ExtractionResult
from kijun.extract.prompt import build_messages
from tests.conftest import make_utterances

IMPL_MODULES = [
    "kijun.transcribe.faster_whisper_impl",
    "kijun.transcribe.pyannote_impl",
    "kijun.extract.ollama_impl",
    "kijun.extract.claude_impl",
    "kijun.match.e5_impl",
]
HEAVY = ["torch", "faster_whisper", "pyannote", "sentence_transformers", "anthropic", "httpx"]


def test_実装モジュールは重い依存を_import_せずに_import_できる():
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in IMPL_MODULES)
        + f"heavy = {HEAVY!r}\n"
        + "loaded = [h for h in heavy if h in sys.modules]\n"
        + "assert not loaded, loaded\n"
    )
    import subprocess

    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("module", IMPL_MODULES)
def test_import_だけでは_ImportError_にならない(module):
    importlib.import_module(module)


def block(monkeypatch, *names: str) -> None:
    """名前の import を失敗させる（インストールされていない状態を作る）。"""
    for n in names:
        monkeypatch.setitem(sys.modules, n, None)


def test_import_optional_は入れ方を示す(monkeypatch):
    block(monkeypatch, "faster_whisper")
    with pytest.raises(MissingDependencyError) as e:
        import_optional("faster_whisper", "transcribe", "文字起こし")
    assert "uv sync --extra transcribe" in str(e.value)
    assert isinstance(e.value, ImportError)


@pytest.mark.parametrize(
    "factory,extra,heavy",
    [
        (lambda c: __import__("kijun.transcribe.faster_whisper_impl", fromlist=["x"]).FasterWhisperTranscriber(c.transcribe), "transcribe", "faster_whisper"),
        (lambda c: __import__("kijun.transcribe.pyannote_impl", fromlist=["x"]).PyannoteDiarizer(c.diarize), "transcribe", "pyannote.audio"),
        (lambda c: __import__("kijun.extract.ollama_impl", fromlist=["x"]).OllamaExtractor(c), "extract-ollama", "httpx"),
        (lambda c: __import__("kijun.extract.claude_impl", fromlist=["x"]).ClaudeExtractor(c), "extract-claude", "anthropic"),
        (lambda c: __import__("kijun.match.e5_impl", fromlist=["x"]).E5Embedder(c.match), "match", "sentence_transformers"),
    ],
)
def test_依存が無ければ_extra_名を含むエラーになる(monkeypatch, factory, extra, heavy):
    block(monkeypatch, heavy)
    with pytest.raises(MissingDependencyError) as e:
        factory(Config())
    assert f"uv sync --extra {extra}" in str(e.value)


def test_HF_TOKEN_が無ければ両方のモデルページへの同意を案内する(monkeypatch):
    fake_pyannote = types.ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = object
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", types.ModuleType("faster_whisper.audio"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from kijun.transcribe.pyannote_impl import PyannoteDiarizer

    with pytest.raises(RuntimeError) as e:
        PyannoteDiarizer(Config().diarize)
    msg = str(e.value)
    assert "HF_TOKEN" in msg
    assert "pyannote/speaker-diarization-3.1" in msg and "pyannote/segmentation-3.0" in msg


# --- Ollama --------------------------------------------------------------------------------


GOOD = json.dumps({"items": []})


class FakeHttpx(types.ModuleType):
    def __init__(self, contents: list[str]):
        super().__init__("httpx")
        self.contents = list(contents)
        self.requests: list[dict] = []

    def post(self, url, json=None, timeout=None):  # noqa: A002 - httpx の引数名に合わせる
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        content = self.contents.pop(0)

        class Response:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"message": {"content": content}}

        return Response()


def a_chunk():
    return split_into_chunks(make_utterances(5), 25, 2)[0]


def test_Ollama_のリクエストの形(monkeypatch):
    fake = FakeHttpx([GOOD])
    monkeypatch.setitem(sys.modules, "httpx", fake)
    from kijun.extract.ollama_impl import OllamaExtractor

    cfg = Config()
    ex = OllamaExtractor(cfg)
    assert ex.model_name == "qwen3:14b"
    assert ex.extract(a_chunk()).items == []

    req = fake.requests[0]
    assert req["url"] == "http://localhost:11434/api/chat"
    payload = req["json"]
    assert payload["model"] == "qwen3:14b"
    assert payload["stream"] is False
    assert payload["keep_alive"] == 0  # 処理後に VRAM からアンロードさせる
    assert payload["options"] == {"temperature": 0}
    assert payload["format"] == ExtractionResult.model_json_schema()
    assert req["timeout"] == 900
    system, user = build_messages(a_chunk(), cfg)
    assert payload["messages"] == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def test_Ollama_の出力が壊れていたら1回だけ投げ直す(monkeypatch, capsys):
    fake = FakeHttpx(["これは JSON ではありません", GOOD])
    monkeypatch.setitem(sys.modules, "httpx", fake)
    from kijun.extract.ollama_impl import OllamaExtractor

    assert OllamaExtractor(Config()).extract(a_chunk()).items == []
    assert len(fake.requests) == 2
    assert "警告" in capsys.readouterr().err


def test_Ollama_の出力が2回壊れていたら失敗として例外にする(monkeypatch):
    fake = FakeHttpx(["壊れ1", "壊れ2", GOOD])
    monkeypatch.setitem(sys.modules, "httpx", fake)
    from kijun.extract.ollama_impl import OllamaExtractor

    with pytest.raises(ExtractionError):
        OllamaExtractor(Config()).extract(a_chunk())
    assert len(fake.requests) == 2  # 3回目は投げない


def test_Ollama_への通信エラーは_ExtractionError(monkeypatch):
    class Down(FakeHttpx):
        def post(self, *a, **k):
            raise ConnectionError("接続できません")

    monkeypatch.setitem(sys.modules, "httpx", Down([]))
    from kijun.extract.ollama_impl import OllamaExtractor

    with pytest.raises(ExtractionError, match="Ollama"):
        OllamaExtractor(Config()).extract(a_chunk())


# --- Claude --------------------------------------------------------------------------------


class FakeAnthropic(types.ModuleType):
    def __init__(self, stop_reason="end_turn", parsed=None):
        super().__init__("anthropic")
        outer = self
        self.calls: list[dict] = []
        self.init_kwargs: dict = {}

        class Usage:
            input_tokens = 10
            output_tokens = 5
            cache_read_input_tokens = 0

        class Response:
            usage = Usage()
            parsed_output = parsed if parsed is not None else ExtractionResult(items=[])

        Response.stop_reason = stop_reason

        class Messages:
            def parse(self_inner, **kwargs):
                outer.calls.append(kwargs)
                return Response()

        class Beta:
            messages = Messages()

        class Client:
            def __init__(self_inner, **kwargs):
                outer.init_kwargs = kwargs
                self_inner.beta = Beta()

        self.Anthropic = Client


def test_Claude_のリクエストの形(monkeypatch, capsys):
    fake = FakeAnthropic()
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    from kijun.extract.claude_impl import ClaudeExtractor

    cfg = Config()
    ex = ClaudeExtractor(cfg)
    assert ex.model_name == "claude-sonnet-5-5"
    assert ex.extract(a_chunk()).items == []

    call = fake.calls[0]
    system, user = build_messages(a_chunk(), cfg)
    # Ollama と同一の文面
    assert call["system"] == [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    assert call["messages"] == [{"role": "user", "content": user}]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["max_tokens"] == 16000
    assert call["output_format"] is ExtractionResult
    assert call["output_config"] == {"effort": "low"}
    assert "thinking" not in call  # Sonnet 5.5 では disabled が 400 になるので渡さない
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert fake.init_kwargs == {"timeout": 600.0}
    assert "cache_read_input_tokens=0" in capsys.readouterr().err


def test_Claude_が拒否したら失敗として例外にする(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropic(stop_reason="refusal"))
    from kijun.extract.claude_impl import ClaudeExtractor

    with pytest.raises(ExtractionError, match="拒否"):
        ClaudeExtractor(Config()).extract(a_chunk())


def test_Claude_の出力が_max_tokens_で切れたら失敗として例外にする(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropic(stop_reason="max_tokens"))
    from kijun.extract.claude_impl import ClaudeExtractor

    with pytest.raises(ExtractionError, match="max_tokens"):
        ClaudeExtractor(Config()).extract(a_chunk())


def test_Ollama_と_Claude_は同一のプロンプトを使う(monkeypatch):
    http = FakeHttpx([GOOD])
    claude = FakeAnthropic()
    monkeypatch.setitem(sys.modules, "httpx", http)
    monkeypatch.setitem(sys.modules, "anthropic", claude)
    from kijun.extract.claude_impl import ClaudeExtractor
    from kijun.extract.ollama_impl import OllamaExtractor

    chunk = a_chunk()
    OllamaExtractor(Config()).extract(chunk)
    ClaudeExtractor(Config()).extract(chunk)
    ollama_msgs = http.requests[0]["json"]["messages"]
    claude_call = claude.calls[0]
    assert ollama_msgs[0]["content"] == claude_call["system"][0]["text"]
    assert ollama_msgs[1]["content"] == claude_call["messages"][0]["content"]
