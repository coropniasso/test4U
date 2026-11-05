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
def test_Ollama_に接続できてモデルの一覧が取れる():
    from kijun.doctor import check_models
    from kijun.transport import HttpxTransport

    result = check_models(HttpxTransport(), Config())
    assert result.status != "error", result.summary


@pytest.mark.requires_network
def test_Ollama_で抽出が_JSON_として取れる():
    from kijun.extract.ollama_impl import OllamaExtractor

    chunk = split_into_chunks(make_utterances(5), 12, 1)[0]
    result = OllamaExtractor(Config()).extract(chunk)
    assert isinstance(result.items, list)


@pytest.mark.requires_gpu
def test_Ollama_のモデルが_GPU_に収まっている():
    """メモリの問題(Strata で起きた RAM の圧迫)の再発を防ぐ確認。size_vram == size であること。"""
    from kijun.doctor import STATUS_OK, run_doctor
    from kijun.transport import HttpxTransport

    report = run_doctor(HttpxTransport(), Config())
    gpu = report.get("gpu")
    assert gpu is not None and gpu.status == STATUS_OK, gpu.summary if gpu else "検査3が無い"


@pytest.mark.requires_network
def test_e5_で埋め込みが取れる():
    from kijun.match.e5_impl import E5Embedder

    e = E5Embedder(Config().match)
    assert e.encode_queries(["告知文の冒頭に行動を書く"]).shape == (1, e.dim)


@pytest.mark.requires_network
def test_discord_py_でチャンネルに投稿してリアクションと返信を読める():
    """実機でだけ動かす。DISCORD_BOT_TOKEN と、投稿してよい試験用チャンネルの ID(KIJUN_TEST_DISCORD_CHANNEL)が要る。

    Message Content Intent が無効だと、返信の読み取りで DiscordError になる(それも確認の対象)。
    """
    token = os.environ.get("DISCORD_BOT_TOKEN")
    channel = os.environ.get("KIJUN_TEST_DISCORD_CHANNEL")
    if not token or not channel:
        pytest.skip("環境変数 DISCORD_BOT_TOKEN と KIJUN_TEST_DISCORD_CHANNEL を設定してください")
    from kijun.discord.impl import DiscordPyClient

    client = DiscordPyClient(token)
    message_id = client.post(channel, "kijun の接続確認です。このメッセージは削除して構いません。")
    assert message_id.isdigit()
    assert message_id in client.fetch_reactions(channel, [message_id])
    assert message_id in client.fetch_thread_replies(channel, [message_id])


@pytest.mark.requires_network
def test_Ollama_で重要度の判定が取れる():
    from kijun.review.importance import LlmImportanceJudge
    from kijun.review.llm import OllamaJsonChat

    verdict = LlmImportanceJudge(OllamaJsonChat(Config())).judge(
        "告知文の冒頭2行以内に、読み手にしてほしい行動が書かれているか", ["募集要項の最初に、何をしてほしいかを書いて"]
    )
    assert 1 <= verdict.importance <= 3 and verdict.rationale


@pytest.mark.requires_network
def test_Ollama_で返信の振り分けが取れる():
    from datetime import datetime

    from kijun.db.repo import KnowledgeItemRow
    from kijun.review.llm import OllamaJsonChat
    from kijun.review.reply import LlmReplyRouter

    item = KnowledgeItemRow("item_0001", None, "冒頭に行動を書くか", None, None, None, None, None, "candidate", 1, datetime(2026, 11, 3))
    routed = LlmReplyRouter(OllamaJsonChat(Config())).route("読み飛ばされるから。ただし社内向けでは不要", item)
    assert {e.column for e in routed.items} <= {"reason", "applies_when", "not_applies_when", "check_text"}
