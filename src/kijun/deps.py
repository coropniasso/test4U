"""任意依存（extra）の遅延 import。

torch、faster-whisper などは `uv sync` の既定では入らない。
各実装クラスの __init__ から import_optional() を呼び、入っていなければ
どの extra を入れればよいかを書いた例外にする。
モジュールのトップレベルでは重い依存を import しない。
"""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingDependencyError(ImportError):
    """任意依存が入っていないときの例外。入れ方をメッセージに含める。"""


def import_optional(module: str, extra: str, purpose: str = "") -> ModuleType:
    """module を import して返す。入っていなければ MissingDependencyError を出す。

    module:  import する名前（例: "faster_whisper"）
    extra:   pyproject.toml の optional-dependencies の名前（例: "transcribe"）
    purpose: エラーメッセージに入れる用途の説明
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        what = f"（{purpose}に必要）" if purpose else ""
        raise MissingDependencyError(
            f"Python パッケージ '{module}' {what}が入っていません。"
            f"次のコマンドで入れてください: uv sync --extra {extra}"
        ) from e
