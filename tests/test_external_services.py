"""実機（GPU、Ollama の推論サーバー、Hugging Face）でだけ動かす確認。

`uv run pytest` の既定では実行しない（pyproject.toml の addopts で除外している）。
実機で動かすには、例えば次のようにする。

    $env:KIJUN_TEST_AUDIO = "C:/path/to/5min.m4a"
    uv run pytest -m requires_gpu
    uv run pytest -m requires_network

Ollama の実測は、このテストよりも `uv run kijun doctor` の方が詳しい結果を出す。
こちらは「つながるか」だけを見る。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kijun.config import Config
from kijun.minutes.chunk import split_into_chunks
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
def test_Ollama_に接続できてモデルの一覧が取れる():
    from kijun.doctor import check_models
    from kijun.transport import HttpxTransport

    result = check_models(HttpxTransport(), Config())
    assert result.status != "error", result.summary


@pytest.mark.requires_network
def test_Ollama_で議事録の材料が_JSON_として取れる():
    from kijun.llm.chat import OllamaJsonChat
    from kijun.minutes.models import ChunkMinutes
    from kijun.minutes.prompt import build_chunk_messages

    cfg = Config()
    chunk = split_into_chunks(make_utterances(5), cfg.minutes.chunk_minutes, 0)[0]
    system, user = build_chunk_messages(chunk)
    result = OllamaJsonChat(cfg.llm).ask(system, user, ChunkMinutes, "試験")
    assert isinstance(result.topics, list)


@pytest.mark.requires_gpu
def test_Ollama_のモデルが_GPU_に収まっている():
    """メモリの問題(Strata で起きた RAM の圧迫)の再発を防ぐ確認。size_vram == size であること。"""
    from kijun.doctor import STATUS_OK, run_doctor
    from kijun.transport import HttpxTransport

    report = run_doctor(HttpxTransport(), Config())
    gpu = report.get("gpu")
    assert gpu is not None and gpu.status == STATUS_OK, gpu.summary if gpu else "検査3が無い"


@pytest.mark.requires_network
def test_discord_py_でチャンネルに本文とファイルを投稿できる():
    """実機でだけ動かす。DISCORD_BOT_TOKEN と、投稿してよい試験用チャンネルの ID(KIJUN_TEST_DISCORD_CHANNEL)が要る。"""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    channel = os.environ.get("KIJUN_TEST_DISCORD_CHANNEL")
    if not token or not channel:
        pytest.skip("環境変数 DISCORD_BOT_TOKEN と KIJUN_TEST_DISCORD_CHANNEL を設定してください")
    from kijun.discord.impl import DiscordPyClient

    message_id = DiscordPyClient(token).post_file(
        channel, "kijun の接続確認です。このメッセージは削除して構いません。", "test.md", "# 試験\n".encode("utf-8")
    )
    assert message_id.isdigit()
