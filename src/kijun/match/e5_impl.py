"""sentence-transformers による埋め込み（multilingual-e5 など）。

モデルファイルのダウンロードが必要。動作確認は Windows の実機で行う。
sentence-transformers は `uv sync --extra match` で入る。
接頭辞は設定の query_prefix / passage_prefix から読む（ruri などに差し替えるときは設定で変える）。
"""

from __future__ import annotations

import numpy as np

from kijun.config import MatchConfig
from kijun.deps import import_optional


class E5Embedder:
    def __init__(self, cfg: MatchConfig) -> None:
        st = import_optional("sentence_transformers", "match", "埋め込みモデルによる照合")
        self._cfg = cfg
        self._model = st.SentenceTransformer(cfg.model, device=cfg.device)
        self.model_name = cfg.model
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            [prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._cfg.query_prefix)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._cfg.passage_prefix)
