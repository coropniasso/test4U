"""NVIDIA の DLL ディレクトリ登録（transcribe/cuda_dlls.py）の検査。実機の Windows は使わない。"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

from kijun.transcribe import cuda_dlls
from kijun.transcribe import faster_whisper_impl as fw_impl


@pytest.fixture(autouse=True)
def _clear_registered():
    cuda_dlls._registered.clear()
    yield
    cuda_dlls._registered.clear()


def _spec_for(root: Path):
    spec = importlib.machinery.ModuleSpec("nvidia", None, is_package=True)
    spec.submodule_search_locations = [str(root)]
    return spec


def _make_nvidia(tmp_path: Path, with_dll=("cublas", "cudnn"), without_dll=()):
    root = tmp_path / "nvidia"
    for name in with_dll:
        (root / name / "bin").mkdir(parents=True)
        (root / name / "bin" / "dummy.dll").write_bytes(b"")
    for name in without_dll:
        (root / name / "bin").mkdir(parents=True)
    return root


@pytest.fixture
def as_windows(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cuda_dlls.os, "add_dll_directory", lambda p: calls.append(p), raising=False)
    return calls


def test_Windows以外では何もせず空のリストを返す(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert cuda_dlls.register_nvidia_dll_directories() == []


def test_bin_を持つ2つのディレクトリを登録する(tmp_path, monkeypatch, as_windows):
    root = _make_nvidia(tmp_path)
    monkeypatch.setattr(cuda_dlls.importlib.util, "find_spec", lambda name: _spec_for(root))
    result = cuda_dlls.register_nvidia_dll_directories()
    assert result == [root / "cublas" / "bin", root / "cudnn" / "bin"]
    assert sorted(as_windows) == sorted(str(p) for p in result)


def test_2回呼んでも各ディレクトリは1回しか登録しない(tmp_path, monkeypatch, as_windows):
    root = _make_nvidia(tmp_path)
    monkeypatch.setattr(cuda_dlls.importlib.util, "find_spec", lambda name: _spec_for(root))
    first = cuda_dlls.register_nvidia_dll_directories()
    second = cuda_dlls.register_nvidia_dll_directories()
    assert first == second
    assert len(as_windows) == 2


def test_dll_が無い_bin_は登録しない(tmp_path, monkeypatch, as_windows):
    root = _make_nvidia(tmp_path, with_dll=("cublas",), without_dll=("empty",))
    monkeypatch.setattr(cuda_dlls.importlib.util, "find_spec", lambda name: _spec_for(root))
    result = cuda_dlls.register_nvidia_dll_directories()
    assert result == [root / "cublas" / "bin"]
    assert as_windows == [str(root / "cublas" / "bin")]


def test_nvidia_パッケージが無ければ空のリストを返す(monkeypatch, as_windows, capsys):
    monkeypatch.setattr(cuda_dlls.importlib.util, "find_spec", lambda name: None)
    assert cuda_dlls.register_nvidia_dll_directories() == []
    assert as_windows == []
    assert "0 件" in capsys.readouterr().err


def test_faster_whisper_を_import_する前に登録を呼ぶ(monkeypatch):
    order: list[str] = []

    def fake_register():
        order.append("register")
        return []

    def fake_import(module, extra, purpose=""):
        order.append(f"import:{module}")
        return types.SimpleNamespace(WhisperModel=lambda *a, **k: object())

    monkeypatch.setattr(fw_impl, "register_nvidia_dll_directories", fake_register)
    monkeypatch.setattr(fw_impl, "import_optional", fake_import)
    from kijun.config import Config

    fw_impl.FasterWhisperTranscriber(Config().transcribe)
    assert order == ["register", "import:faster_whisper"]


def test_cublas_のエラーは対処を案内し_元の例外を残す(monkeypatch):
    dirs = [Path("X/cublas/bin")]

    def boom(*a, **k):
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    monkeypatch.setattr(fw_impl, "register_nvidia_dll_directories", lambda: dirs)
    monkeypatch.setattr(
        fw_impl, "import_optional", lambda *a, **k: types.SimpleNamespace(WhisperModel=boom)
    )
    from kijun.config import Config

    with pytest.raises(RuntimeError) as ei:
        fw_impl.FasterWhisperTranscriber(Config().transcribe)
    msg = str(ei.value)
    assert "uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12" in msg
    assert "cublas/bin" in msg
    assert isinstance(ei.value.__cause__, RuntimeError)
