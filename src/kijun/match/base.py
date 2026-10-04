"""埋め込みモデルのインターフェース（計画書 9-1）。

queries と passages を分けるのは、e5 系のモデルが入力に `query: ` / `passage: ` の接頭辞を
要求するため。接頭辞を付け忘れると類似度の精度が落ちる。インターフェースの段階で分けて、
呼び出し側が付け忘れられないようにする。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    model_name: str

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """検索する側の文（抽出結果の要約）をベクトル化する。形状は (len(texts), dim)。"""
        ...

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """検索される側の文（既存項目の check_text）をベクトル化する。形状は (len(texts), dim)。"""
        ...
