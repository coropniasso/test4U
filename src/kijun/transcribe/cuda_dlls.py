"""Windows で、仮想環境内の NVIDIA の DLL のディレクトリを DLL の探索対象に登録する。

faster-whisper の計算エンジンである CTranslate2 は `cublas64_12.dll`（CUDA 12 の cuBLAS）と cuDNN を要求する。
torch 2.15.0+cu132 が同梱するのは `cublas64_13.dll` で名前が違うため、
`uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` で CUDA 12 版を併せて入れる。
その DLL は `<site-packages>/nvidia/<パッケージ名>/bin` に入る。PATH に手で足すとそのウィンドウの間しか
有効でなく、タスクスケジューラから起動する夜間バッチでは失敗するので、コードの中で登録する。

登録は2つ行う。両方必要である（2026-11-03 実機で確認）。
- `os.add_dll_directory()`: 足すのは LOAD_LIBRARY_SEARCH_USER_DIRS の対象で、LoadLibraryEx にそのフラグを
  付けた呼び出しにしか効かない。CTranslate2 は素の LoadLibrary 相当で読み込むため、これだけでは
  `cublas64_12.dll is not found or cannot be loaded` で落ちる。
- `os.environ["PATH"]` の先頭に足す: 素の LoadLibrary の既定の探索順には PATH が入る。これはプロセス内の
  環境変数なので、タスクスケジューラから起動した夜間バッチでも効く（PowerShell の `$env:PATH` と違い、
  ウィンドウに依存しない）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# 登録済みのディレクトリ。何度呼んでも同じディレクトリを二重に登録しないための記録。
_registered: set[Path] = set()


def _nvidia_roots() -> list[Path]:
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    if spec is None or spec.submodule_search_locations is None:
        return []
    return [Path(p) for p in spec.submodule_search_locations]


def _prepend_to_path(directory: Path) -> None:
    """PATH の先頭に directory を足す。既に PATH にあれば足さない（何度呼んでも重複しない）。"""
    entry = str(directory)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if entry in parts:
        return
    os.environ["PATH"] = os.pathsep.join([entry, *parts]) if parts else entry


def register_nvidia_dll_directories() -> list[Path]:
    """登録したディレクトリ（既に登録済みのものを含む）の一覧を返す。Windows 以外では空。"""
    if sys.platform != "win32":
        return []
    found: list[Path] = []
    for root in _nvidia_roots():
        if not root.is_dir():
            continue
        for sub in sorted(root.iterdir()):
            bin_dir = sub / "bin"
            if bin_dir.is_dir() and any(bin_dir.glob("*.dll")):
                found.append(bin_dir)
    for d in found:
        if d not in _registered:
            os.add_dll_directory(str(d))
            _registered.add(d)
        _prepend_to_path(d)
    if found:
        print("NVIDIA の DLL ディレクトリを登録: " + ", ".join(str(d) for d in found), file=sys.stderr)
    else:
        print(
            "NVIDIA の DLL ディレクトリは 0 件（nvidia パッケージが無いか、bin に .dll が無い）",
            file=sys.stderr,
        )
    return found
