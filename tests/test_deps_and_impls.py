"""重い依存を使う実装クラスが、依存が無くても import でき、入れ方を示すエラーを出すことを検査する。

また、偽の HTTP 層を差し込んで、Strata の実装がリクエストの形と再試行・修復を正しく行うことを
検査する（実際の通信はしない）。
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types

import pytest

from kijun.config import Config
from kijun.deps import MissingDependencyError, import_optional
from kijun.extract.base import ExtractionError
from kijun.extract.chunk import split_into_chunks
from kijun.extract.models import ExtractionResult
from kijun.extract.prompt import build_messages
from kijun.extract.repair import ACTION_REMOVED_PROSE_BEFORE, ACTION_WRAPPED_ARRAY_IN_ITEMS
from kijun.extract.strata_impl import StrataExtractor
from kijun.transport import HttpResponse, TransportError
from tests.conftest import make_utterances

IMPL_MODULES = [
    "kijun.transcribe.faster_whisper_impl",
    "kijun.transcribe.pyannote_impl",
    "kijun.extract.strata_impl",
    "kijun.match.e5_impl",
    "kijun.transport",
]
HEAVY = ["torch", "faster_whisper", "pyannote", "sentence_transformers", "httpx"]


def test_実装モジュールは重い依存を_import_せずに_import_できる():
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in IMPL_MODULES)
        + f"heavy = {HEAVY!r}\n"
        + "loaded = [h for h in heavy if h in sys.modules]\n"
        + "assert not loaded, loaded\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_anthropic_に依存しない():
    """抽出から Claude API が消えたので、anthropic は import されない。"""
    code = "import sys\nimport kijun.cli\nimport kijun.extract.strata_impl\nassert 'anthropic' not in sys.modules\n"
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
        (
            lambda c: __import__(
                "kijun.transcribe.faster_whisper_impl", fromlist=["x"]
            ).FasterWhisperTranscriber(c.transcribe),
            "transcribe",
            "faster_whisper",
        ),
        (
            lambda c: __import__("kijun.transcribe.pyannote_impl", fromlist=["x"]).PyannoteDiarizer(
                c.diarize
            ),
            "transcribe",
            "pyannote.audio",
        ),
        (lambda c: StrataExtractor(c), "extract", "httpx"),
        (
            lambda c: __import__("kijun.match.e5_impl", fromlist=["x"]).E5Embedder(c.match),
            "match",
            "sentence_transformers",
        ),
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


# --- Strata ---------------------------------------------------------------------------------

ITEM = {
    "seq_from": 0,
    "seq_to": 0,
    "speaker_from_label": "SPEAKER_00",
    "speaker_to_label": None,
    "counterpart_role": None,
    "artifact_type": None,
    "quote": "結論を先に書いて",
    "request_summary": "結論を先に書くことを求めた",
    "stated_reason": None,
    "applies_when": None,
    "reusable": True,
    "pass_criterion": None,
}
EMPTY = json.dumps({"items": []})
ONE_ITEM = json.dumps({"items": [ITEM]}, ensure_ascii=False)


class FakeTransport:
    """決めた応答を順に返す偽の HTTP 層。送ったリクエストを記録する。"""

    def __init__(self, responses: list, elapsed_sec: float = 1.0) -> None:
        self._responses = list(responses)
        self._elapsed = elapsed_sec
        self.posts: list[dict] = []
        self.gets: list[str] = []

    def _next(self) -> HttpResponse:
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, HttpResponse):
            return item
        status, text = item
        return HttpResponse(status_code=status, text=text, elapsed_sec=self._elapsed)

    def get(self, url: str, timeout_sec: float) -> HttpResponse:
        self.gets.append(url)
        return self._next()

    def post(self, url: str, payload: dict, timeout_sec: float) -> HttpResponse:
        self.posts.append({"url": url, "payload": payload, "timeout_sec": timeout_sec})
        return self._next()


def chat_body(content: str) -> str:
    return json.dumps({"choices": [{"message": {"content": content}}]}, ensure_ascii=False)


def chat_body_with_usage(content: str, prompt: int = 100, completion: int = 20) -> str:
    return json.dumps(
        {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
        ensure_ascii=False,
    )


def a_chunk():
    return split_into_chunks(make_utterances(5), 25, 2)[0]


def test_Strata_のリクエストの形():
    transport = FakeTransport([(200, chat_body(EMPTY))])
    cfg = Config()
    ex = StrataExtractor(cfg, transport)
    assert ex.model_name == "qwen3.8-flash-next"
    assert ex.enforces_json_schema is False
    assert ex.max_input_tokens is None  # 設定の 0 は未確認なので None
    assert ex.extract(a_chunk()).items == []

    post = transport.posts[0]
    assert post["url"] == "http://localhost:8080/v1/chat/completions"
    assert post["timeout_sec"] == 3600
    payload = post["payload"]
    assert payload["model"] == "qwen3.8-flash-next"
    assert payload["temperature"] == 0
    assert payload["stream"] is False
    # 保証しないと宣言されているので response_format は付けない
    assert "response_format" not in payload
    system, user = build_messages(a_chunk(), cfg)
    assert payload["messages"] == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def test_プロンプトの文面は_prompt_py_のものをそのまま使う():
    transport = FakeTransport([(200, chat_body(EMPTY))])
    cfg = Config()
    StrataExtractor(cfg, transport).extract(a_chunk())
    system = transport.posts[0]["payload"]["messages"][0]["content"]
    assert system == build_messages(a_chunk(), cfg)[0]
    assert "推測" in system and "Yes か No で判定できる" in system


def test_スキーマ適合を保証すると宣言したときだけ_response_format_を付ける():
    cfg = Config()
    cfg.extract.strata.enforces_json_schema = True
    transport = FakeTransport([(200, chat_body(EMPTY))])
    ex = StrataExtractor(cfg, transport)
    assert ex.enforces_json_schema is True
    ex.extract(a_chunk())
    fmt = transport.posts[0]["payload"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "extraction"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == ExtractionResult.model_json_schema()


def test_max_input_tokens_は設定の値をそのまま属性にする():
    cfg = Config()
    cfg.extract.strata.max_input_tokens = 50000
    assert StrataExtractor(cfg, FakeTransport([])).max_input_tokens == 50000


def test_chat_path_と_base_url_は設定で変えられる():
    cfg = Config()
    cfg.extract.strata.base_url = "http://192.168.1.5:9000"
    cfg.extract.strata.chat_path = "/openai/v1/chat/completions"
    transport = FakeTransport([(200, chat_body(EMPTY))])
    StrataExtractor(cfg, transport).extract(a_chunk())
    assert transport.posts[0]["url"] == "http://192.168.1.5:9000/openai/v1/chat/completions"


# --- 修復と再試行 -----------------------------------------------------------------------------


def test_前置きつきの出力を修復して読み_修復の記録を残す(capsys):
    transport = FakeTransport([(200, chat_body("はい、結果です。\n" + ONE_ITEM))])
    ex = StrataExtractor(Config(), transport)
    result = ex.extract(a_chunk())

    assert len(result.items) == 1
    assert len(ex.repair_events) == 1
    event = ex.repair_events[0]
    assert event.actions == [ACTION_REMOVED_PROSE_BEFORE]
    assert event.attempt == 1 and event.truncated_items == 0
    assert event.seq_from == 0
    assert "修復しました" in capsys.readouterr().err


def test_トップレベルが配列の出力も読める():
    transport = FakeTransport([(200, chat_body(json.dumps([ITEM], ensure_ascii=False)))])
    ex = StrataExtractor(Config(), transport)
    assert len(ex.extract(a_chunk()).items) == 1
    assert ex.repair_events[0].actions == [ACTION_WRAPPED_ARRAY_IN_ITEMS]


def test_修復が要らなかったときは記録を残さない():
    transport = FakeTransport([(200, chat_body(EMPTY))])
    ex = StrataExtractor(Config(), transport)
    ex.extract(a_chunk())
    assert ex.repair_events == []


def test_末尾が切れた出力は捨てた件数を記録する():
    full = json.dumps({"items": [ITEM, ITEM]}, ensure_ascii=False)
    cut = full[: full.rindex("{") + 30]
    transport = FakeTransport([(200, chat_body(cut))])
    ex = StrataExtractor(Config(), transport)
    assert len(ex.extract(a_chunk()).items) == 1
    assert ex.repair_events[0].truncated_items == 1


def test_読めない出力は1回だけ投げ直す(capsys):
    transport = FakeTransport([(200, chat_body("抽出できませんでした")), (200, chat_body(EMPTY))])
    ex = StrataExtractor(Config(), transport)
    assert ex.extract(a_chunk()).items == []
    assert len(transport.posts) == 2
    assert "警告" in capsys.readouterr().err


def test_2回とも読めなければ_ExtractionError():
    transport = FakeTransport([(200, chat_body("だめ1")), (200, chat_body("だめ2")), (200, chat_body(EMPTY))])
    ex = StrataExtractor(Config(), transport)
    with pytest.raises(ExtractionError, match="2 回とも"):
        ex.extract(a_chunk())
    assert len(transport.posts) == 2  # 3回目は投げない


def test_投げ直しても同じリクエストを送る():
    transport = FakeTransport([(200, chat_body("だめ")), (200, chat_body(EMPTY))])
    StrataExtractor(Config(), transport).extract(a_chunk())
    assert transport.posts[0]["payload"] == transport.posts[1]["payload"]


# --- 失敗の扱い -------------------------------------------------------------------------------


def test_接続できなければ_doctor_を案内する():
    transport = FakeTransport([TransportError("接続拒否")])
    with pytest.raises(ExtractionError) as e:
        StrataExtractor(Config(), transport).extract(a_chunk())
    assert "kijun doctor" in str(e.value) and "送れませんでした" in str(e.value)


def test_HTTP_エラーは状態コードと本文を出す():
    transport = FakeTransport([(400, '{"error": "response_format is not supported"}')])
    with pytest.raises(ExtractionError) as e:
        StrataExtractor(Config(), transport).extract(a_chunk())
    assert "HTTP 400" in str(e.value) and "response_format" in str(e.value)


def test_応答の形が_OpenAI_互換でなければ_chat_path_を疑うよう案内する():
    transport = FakeTransport([(200, json.dumps({"content": "x"}))])
    with pytest.raises(ExtractionError) as e:
        StrataExtractor(Config(), transport).extract(a_chunk())
    assert "chat_path" in str(e.value) and "choices" in str(e.value)


def test_応答が_JSON_でなければ_ExtractionError():
    transport = FakeTransport([(200, "<html>404</html>")])
    with pytest.raises(ExtractionError):
        StrataExtractor(Config(), transport).extract(a_chunk())


def test_usage_があればトークン数を出す(capsys):
    transport = FakeTransport([(200, chat_body_with_usage(EMPTY, 1234, 56))])
    StrataExtractor(Config(), transport).extract(a_chunk())
    err = capsys.readouterr().err
    assert "1234" in err and "56" in err


def test_usage_が無ければそのことを出す(capsys):
    transport = FakeTransport([(200, chat_body(EMPTY))])
    StrataExtractor(Config(), transport).extract(a_chunk())
    assert "usage を返さない" in capsys.readouterr().err
