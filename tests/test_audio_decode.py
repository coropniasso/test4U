"""音声デコーダ（transcribe/audio_decode.py）のテスト。av は PyPI から入るので GPU もモデルも要らない。"""

from __future__ import annotations

import math
import re
import struct
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av")

from kijun.config import Config
from kijun.transcribe.audio_decode import SAMPLE_RATE, AudioDecodeError, decode_to_mono_16k


def write_wav(path: Path, rate: int, channels: int, seconds: float = 1.0) -> int:
    n = int(rate * seconds)
    frames = bytearray()
    for i in range(n):
        for ch in range(channels):
            v = int(12000 * math.sin(2 * math.pi * (440 + 220 * ch) * i / rate))
            frames += struct.pack("<h", v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return n


def test_16k_モノラルはサンプル数が一致し値が範囲内(tmp_path):
    p = tmp_path / "a.wav"
    n = write_wav(p, 16000, 1)
    audio = decode_to_mono_16k(p)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == n
    assert audio.min() >= -1.0 and audio.max() <= 1.0
    assert audio.max() > 0.2  # 無音ではない


def test_44k_は_16k_に変換される(tmp_path):
    p = tmp_path / "b.wav"
    n = write_wav(p, 44100, 1)
    audio = decode_to_mono_16k(p)
    assert abs(len(audio) - n * 16000 / 44100) <= 64
    assert audio.max() <= 1.0 and audio.min() >= -1.0


def test_ステレオはモノラルの1次元配列になる(tmp_path):
    p = tmp_path / "c.wav"
    write_wav(p, 16000, 2)
    audio = decode_to_mono_16k(p)
    assert audio.ndim == 1
    assert abs(len(audio) - 16000) <= 64


def test_存在しないファイルはパスを含む例外(tmp_path):
    p = tmp_path / "none.wav"
    with pytest.raises(AudioDecodeError) as e:
        decode_to_mono_16k(p)
    assert str(p) in str(e.value)


def test_壊れたファイルはパスを含む例外(tmp_path):
    p = tmp_path / "broken.wav"
    p.write_text("これは音声ではありません", encoding="utf-8")
    with pytest.raises(AudioDecodeError) as e:
        decode_to_mono_16k(p)
    assert str(p) in str(e.value)


def test_faster_whisper_impl_は_ndarray_を_transcribe_に渡す(tmp_path, monkeypatch):
    """退行防止：文字列のパスを渡すと faster-whisper 内部の decode_audio が呼ばれ、PyAV の版で壊れる。"""
    received = []

    class FakeSeg:
        start, end, text = 0.0, 0.5, "こんにちは"

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kw):
            received.append(audio)
            return iter([FakeSeg()]), types.SimpleNamespace(duration=999.0)  # 999 は使われない

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    from kijun.transcribe.faster_whisper_impl import FasterWhisperTranscriber

    p1, p2 = tmp_path / "1.wav", tmp_path / "2.wav"
    write_wav(p1, 16000, 1, 1.0)
    write_wav(p2, 16000, 1, 2.0)
    segs = FasterWhisperTranscriber(Config().transcribe).transcribe([p1, p2])
    assert len(received) == 2
    assert all(isinstance(a, np.ndarray) and a.dtype == np.float32 for a in received)
    # 2本目の時刻は、デコードした1本目の長さ（1秒）だけずれる。info.duration（999）は使わない。
    assert segs[1].start_sec == pytest.approx(1.0, abs=0.01)


def test_pyannote_impl_は_faster_whisper_を_import_しない(monkeypatch):
    src = (Path(__file__).parent.parent / "src/kijun/transcribe/pyannote_impl.py").read_text(encoding="utf-8")
    assert not re.search(r"faster[_-]whisper", src)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import すると ImportError になる
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", None)
    sys.modules.pop("kijun.transcribe.pyannote_impl", None)
    import importlib

    importlib.import_module("kijun.transcribe.pyannote_impl")


def test_pyannote_の波形は複数ファイルをつなぐ(tmp_path):
    from kijun.transcribe.pyannote_impl import PyannoteDiarizer

    class FakeTorch:
        @staticmethod
        def from_numpy(a):
            return types.SimpleNamespace(unsqueeze=lambda dim: a.shape)

    p1, p2 = tmp_path / "1.wav", tmp_path / "2.wav"
    write_wav(p1, 16000, 1, 1.0)
    write_wav(p2, 44100, 2, 2.0)
    d = object.__new__(PyannoteDiarizer)
    d._torch = FakeTorch
    out = d._load_waveform([p1, p2])
    assert out["sample_rate"] == SAMPLE_RATE
    assert abs(out["waveform"][0] - 48000) <= 128


def _fake_whisper(monkeypatch, seen_kwargs: list) -> None:
    class FakeSeg:
        start, end, text = 0.0, 0.5, "こんにちは"

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kw):
            seen_kwargs.append(kw)
            return iter([FakeSeg()]), types.SimpleNamespace(duration=1.0)

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)


def test_hotwords_は空白でつないだ1つの文字列として_transcribe_に渡す(tmp_path, monkeypatch):
    seen: list[dict] = []
    _fake_whisper(monkeypatch, seen)
    from kijun.transcribe.faster_whisper_impl import FasterWhisperTranscriber

    cfg = Config()
    cfg.transcribe.hotwords = ["箇条書き", " 感嘆符 ", ""]  # 前後の空白と空の語は取り除く
    p = tmp_path / "1.wav"
    write_wav(p, 16000, 1, 1.0)
    FasterWhisperTranscriber(cfg.transcribe).transcribe([p])
    assert seen[0]["hotwords"] == "箇条書き 感嘆符"


def test_hotwords_が空なら_None_を渡す(tmp_path, monkeypatch):
    seen: list[dict] = []
    _fake_whisper(monkeypatch, seen)
    from kijun.transcribe.faster_whisper_impl import FasterWhisperTranscriber

    assert Config().transcribe.hotwords == []
    p = tmp_path / "1.wav"
    write_wav(p, 16000, 1, 1.0)
    FasterWhisperTranscriber(Config().transcribe).transcribe([p])
    assert seen[0]["hotwords"] is None


def test_hotwords_は実在する_faster_whisper_の引数名である():
    # 偽のモデルでは引数名の綴り間違いを検出できないので、実物の署名で確かめる。
    # faster-whisper が入っていない環境では飛ばす。
    fw = pytest.importorskip("faster_whisper")
    import inspect

    assert "hotwords" in inspect.signature(fw.WhisperModel.transcribe).parameters
