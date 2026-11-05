from __future__ import annotations

import wave

import pytest

from kijun.ingest.audio import get_duration_sec


def write_wav(path, seconds: float, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def test_wav_の長さを秒で返す(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p, 2.5)
    assert get_duration_sec(p) == pytest.approx(2.5, abs=0.01)


def test_音声でないファイルは_None(tmp_path):
    p = tmp_path / "a.m4a"
    p.write_bytes(b"not audio")
    assert get_duration_sec(p) is None


def test_存在しないファイルは_None(tmp_path):
    assert get_duration_sec(tmp_path / "none.m4a") is None
