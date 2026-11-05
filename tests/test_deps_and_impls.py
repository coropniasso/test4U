"""重い依存を使う実装クラスが、依存が無くても import でき、入れ方を示すエラーを出すことを検査する。

また、偽の HTTP 層を差し込んで、Ollama の実装がリクエストの形と再試行・修復を正しく行うことを
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
from kijun.minutes.models import ChunkMinutes
from kijun.ollama_api import require_all_properties

IMPL_MODULES = [
    "kijun.transcribe.faster_whisper_impl",
    "kijun.transcribe.pyannote_impl",
    "kijun.llm.chat",
    "kijun.minutes.generate",
    "kijun.discord.impl",
    "kijun.batch.defer",
    "kijun.transport",
    "kijun.doctor",
]
HEAVY = ["torch", "faster_whisper", "pyannote", "discord", "httpx"]


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
        (lambda c: __import__("kijun.llm.chat", fromlist=["x"]).OllamaJsonChat(c.llm), "extract", "httpx"),
        (lambda c: __import__("kijun.discord.impl", fromlist=["x"]).DiscordPyClient("t"), "discord", "discord"),
    ],
)
def test_依存が無ければ_extra_名を含むエラーになる(monkeypatch, factory, extra, heavy):
    block(monkeypatch, heavy)
    with pytest.raises(MissingDependencyError) as e:
        factory(Config())
    assert f"uv sync --extra {extra}" in str(e.value)


def test_HF_TOKEN_が無ければ3つのモデルページへの同意を案内する(monkeypatch):
    fake_pyannote = types.ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = object
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from kijun.transcribe.pyannote_impl import PyannoteDiarizer

    with pytest.raises(RuntimeError) as e:
        PyannoteDiarizer(Config().diarize)
    msg = str(e.value)
    assert "HF_TOKEN" in msg
    for url in PYANNOTE_PAGE_URLS:
        assert url in msg
    assert "4.x" in msg


PYANNOTE_PAGE_URLS = [
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
    "https://huggingface.co/pyannote/speaker-diarization-community-1",
]


def test_モデル読み込みが失敗したときも3つのURLを案内する(monkeypatch):
    class FakePipeline:
        @staticmethod
        def from_pretrained(name, token=None):
            raise OSError("Cannot access gated repo (403)")

    fake_pyannote = types.ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    from kijun.transcribe.pyannote_impl import PyannoteDiarizer

    with pytest.raises(RuntimeError) as e:
        PyannoteDiarizer(Config().diarize)
    msg = str(e.value)
    for url in PYANNOTE_PAGE_URLS:
        assert url in msg
    assert "4.x" in msg


# --- faster-whisper の VAD パラメータ ---------------------------------------------------------
#
# 既定のままでは区間が切れないことが実機で分かった（2026-11-04。34.2分の通話で 564.8 秒の区間が
# 1つ出て、中身は3文字だった）。max_utterance_sec は結合を抑えるだけで分割はしないので、
# 区間の長さを抑えるのは VAD 側である。


def test_VAD_パラメータは設定から作られる():
    from kijun.transcribe.faster_whisper_impl import build_vad_parameters

    cfg = Config().transcribe
    got = build_vad_parameters(cfg)
    assert got["threshold"] == cfg.vad_threshold
    assert got["min_silence_duration_ms"] == cfg.vad_min_silence_duration_ms
    assert got["speech_pad_ms"] == cfg.vad_speech_pad_ms
    # 既定で上限が入ること。これが入らないと 564.8 秒の区間が再発する
    assert got["max_speech_duration_s"] == 30.0


def test_上限を_0_にすると_max_speech_duration_s_を渡さない():
    from kijun.transcribe.faster_whisper_impl import build_vad_parameters

    cfg = Config().transcribe
    cfg.vad_max_speech_duration_sec = 0
    assert "max_speech_duration_s" not in build_vad_parameters(cfg)


def test_その版が受け付けない項目は渡さない():
    # faster-whisper の VadOptions は版によって項目が増減する。知らない名前を渡すと TypeError。
    from kijun.transcribe.faster_whisper_impl import build_vad_parameters

    cfg = Config().transcribe
    got = build_vad_parameters(cfg, accepted={"threshold", "speech_pad_ms"})
    assert set(got) == {"threshold", "speech_pad_ms"}


def test_VadOptions_の項目名を取り出す():
    import dataclasses
    import types

    from kijun.transcribe.faster_whisper_impl import vad_option_names

    @dataclasses.dataclass
    class VadOptions:
        threshold: float = 0.5
        max_speech_duration_s: float = float("inf")

    # faster_whisper.vad.VadOptions にある場合
    fw = types.SimpleNamespace(vad=types.SimpleNamespace(VadOptions=VadOptions))
    assert vad_option_names(fw) == {"threshold", "max_speech_duration_s"}
    # faster_whisper.VadOptions にある場合
    assert vad_option_names(types.SimpleNamespace(VadOptions=VadOptions)) == {
        "threshold", "max_speech_duration_s"
    }
    # 見つからない場合は None（絞り込まない）
    assert vad_option_names(types.SimpleNamespace()) is None
    # データクラスでない場合も None
    assert vad_option_names(types.SimpleNamespace(VadOptions=object)) is None


def test_require_all_properties_は既定値のある項目も必須にし_元のスキーマを変えない():
    """抽出の時代の実測: required でない項目を、qwen3.5:9b は一度も出さなかった。"""
    original = ChunkMinutes.model_json_schema()
    assert original.get("required", []) == []  # 全項目に既定値がある
    strict = require_all_properties(original)
    assert set(strict["required"]) == {"topics", "decisions", "todos", "open_issues"}
    todo = strict["$defs"]["Todo"]
    assert set(todo["required"]) == {"task", "owner", "due"}  # ネストしたオブジェクトも必須にする
    # 写しを返す(pydantic が返した辞書を書き換えない)
    assert original["$defs"]["Todo"]["required"] == ["task"]
