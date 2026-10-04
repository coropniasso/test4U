"""実機（GPU、Ollama、Claude API、Hugging Face）でだけ動かす確認。

`uv run pytest` の既定では実行しない（pyproject.toml の addopts で除外している）。
実機で動かすには、例えば次のようにする。

    $env:KIJUN_TEST_AUDIO = "C:/path/to/5min.m4a"
    uv run pytest -m requires_gpu
    uv run pytest -m requires_network
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kijun.config import Config
from kijun.extract.chunk import split_into_chunks
from tests.conftest import make_utterances


def _audio() -> Path:
    path = os.environ.get("KIJUN_TEST_AUDIO")
    if not path:
        pytest.skip("環境変数 KIJUN_TEST_AUDIO に、試験用の音声ファイルのパスを設定してください")
    return Path(path)


@pytest.mark.requires_gpu
def test_CUDA_が使える():
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()


@pytest.mark.requires_gpu
def test_faster_whisper_で日本語の文字起こしができる():
    from kijun.transcribe.faster_whisper_impl import FasterWhisperTranscriber

    segments = FasterWhisperTranscriber(Config().transcribe).transcribe([_audio()])
    assert segments and all(s.end_sec >= s.start_sec for s in segments)


@pytest.mark.requires_gpu
def test_pyannote_で話者ラベルが取れる():
    from kijun.transcribe.pyannote_impl import PyannoteDiarizer

    turns = PyannoteDiarizer(Config().diarize).diarize([_audio()])
    assert turns and all(t.speaker_label.startswith("SPEAKER_") for t in turns)


@pytest.mark.requires_network
def test_Ollama_で抽出が_JSON_として取れる():
    from kijun.extract.ollama_impl import OllamaExtractor

    chunk = split_into_chunks(make_utterances(5), 25, 2)[0]
    result = OllamaExtractor(Config()).extract(chunk)
    assert isinstance(result.items, list)


@pytest.mark.requires_network
def test_Claude_API_で抽出が取れる():
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY が未設定")
    from kijun.extract.claude_impl import ClaudeExtractor

    chunk = split_into_chunks(make_utterances(5), 25, 2)[0]
    result = ClaudeExtractor(Config()).extract(chunk)
    assert isinstance(result.items, list)


@pytest.mark.requires_network
def test_e5_で埋め込みが取れる():
    from kijun.match.e5_impl import E5Embedder

    e = E5Embedder(Config().match)
    assert e.encode_queries(["告知文の冒頭に行動を書く"]).shape == (1, e.dim)
