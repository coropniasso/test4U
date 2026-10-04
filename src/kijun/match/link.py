"""コサイン類似度と閾値による分岐（計画書 9-2、CLAUDE.md 4-5）。

コサイン類似度は DuckDB の array_cosine_similarity ではなく numpy で計算する。
array_cosine_similarity は固定長の FLOAT[N] を要求し、埋め込みモデルを替えて次元数が変わると
スキーマを書き換える必要が出るため。ベクトルは可変長の FLOAT[] で embeddings テーブルに保存する。
テキストの編集に追従するため、元テキストのハッシュ（text_hash）も一緒に保存する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb
import numpy as np

from kijun.config import Config
from kijun.db import repo
from kijun.match.base import Embedder

ACTION_LINKED = "linked"  # 既存項目の根拠として紐付けた
ACTION_NEW_CANDIDATE = "new_candidate"  # 新規候補を作った


@dataclass(frozen=True)
class MatchResult:
    """抽出結果1件の照合結果。"""

    request_id: str
    request_summary: str
    best_item_id: str | None  # 最も類似度が高い既存項目。既存項目が無ければ None
    best_check_text: str | None
    similarity: float | None  # 既存項目が無ければ None
    action: str  # ACTION_LINKED または ACTION_NEW_CANDIDATE
    new_item_id: str | None = None  # 新規候補を作った場合の item_id（dry-run では None）


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a の各行と b の各行のコサイン類似度。形状は (len(a), len(b))。

    ベクトルの長さが 0 の行は、類似度を 0 とする。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = np.divide(a, a_norm, out=np.zeros_like(a), where=a_norm > 0)
    b_unit = np.divide(b, b_norm, out=np.zeros_like(b), where=b_norm > 0)
    return a_unit @ b_unit.T


def _vectors_with_cache(
    con: duckdb.DuckDBPyConnection,
    owner_type: str,
    owner_ids: list[str],
    texts: list[str],
    embedder: Embedder,
    encode,
    now: datetime,
    save: bool,
) -> np.ndarray:
    """owner_ids に対応するベクトルを返す。embeddings に保存済みのものは再計算しない。

    保存済みとみなすのは、model_name が embedder.model_name と一致し、かつ text_hash が現在の
    テキストと一致する行だけ。テキスト（check_text など）が編集されていれば再計算して上書きする。
    save が False のときは、新しく計算したベクトルを保存しない（dry-run 用）。
    """
    cached = repo.get_embeddings(
        con, owner_type, dict(zip(owner_ids, texts)), embedder.model_name
    )
    missing = [i for i, oid in enumerate(owner_ids) if oid not in cached]
    if missing:
        computed = encode([texts[i] for i in missing])
        new_vectors = {owner_ids[i]: computed[k] for k, i in enumerate(missing)}
        cached.update(new_vectors)
        if save:
            repo.put_embeddings(
                con,
                owner_type,
                embedder.model_name,
                {owner_ids[i]: (texts[i], new_vectors[owner_ids[i]]) for i in missing},
                now,
            )
    if not owner_ids:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.stack([cached[oid] for oid in owner_ids])


def link_requests(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    conversation_id: str,
    embedder: Embedder,
    model_name: str | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> list[MatchResult]:
    """会議の抽出結果を、既存のチェック項目と照合する。

    1. 対象は、会議の extracted_requests のうち reusable = true で、まだ item_evidence に
       紐付いていないもの。reusable = false は項目にならないので照合しない。
       model_name を指定すると、そのモデルの抽出結果だけを対象にする
       （Qwen と Claude の両方で抽出した場合に、同じ指摘から項目が二重にできるのを避けるため）。
    2. 既存項目は、status が candidate / approved / held のもの。rejected は照合しない。
    3. 類似度が similarity_threshold 以上なら、最も類似する既存項目に item_evidence で紐付ける。
       未満なら、knowledge_items に status = 'candidate'、version = 1 の行を作って紐付ける
       （similarity は NULL）。
    4. 既存項目が 0 件なら、全件が新規候補になる。
    5. dry_run が True のときは、DB に何も書かず、結果だけを返す。結果は類似度の降順
       （既存項目が無いものは末尾）に並べる。

    比較の対象は、この関数を呼んだ時点で存在する既存項目だけ。この呼び出しで作った新規候補とは
    照合しない（同じ会議の似た指摘は、別の候補として人が見て判断する）。
    """
    now = now or datetime.now()
    threshold = cfg.match.similarity_threshold
    save = not dry_run

    requests = [
        r
        for r in repo.get_requests(con, conversation_id, model_name)
        if r.reusable
    ]
    already_linked = repo.get_linked_request_ids(con, [r.request_id for r in requests])
    requests = [r for r in requests if r.request_id not in already_linked]
    if not requests:
        return []

    items = repo.list_knowledge_items(
        con, [repo.STATUS_CANDIDATE, repo.STATUS_APPROVED, repo.STATUS_HELD]
    )

    request_vectors = _vectors_with_cache(
        con,
        repo.OWNER_EXTRACTED_REQUEST,
        [r.request_id for r in requests],
        [r.request_summary for r in requests],
        embedder,
        embedder.encode_queries,
        now,
        save,
    )
    item_vectors = _vectors_with_cache(
        con,
        repo.OWNER_KNOWLEDGE_ITEM,
        [i.item_id for i in items],
        [i.check_text for i in items],
        embedder,
        embedder.encode_passages,
        now,
        save,
    )
    sims = cosine_similarity_matrix(request_vectors, item_vectors)

    results: list[MatchResult] = []
    for idx, req in enumerate(requests):
        if items:
            best = int(np.argmax(sims[idx]))
            similarity: float | None = float(sims[idx][best])
            best_item = items[best]
        else:
            similarity, best_item = None, None

        if best_item is not None and similarity is not None and similarity >= threshold:
            if save:
                repo.add_evidence(con, best_item.item_id, req.request_id, similarity, now)
            results.append(
                MatchResult(
                    request_id=req.request_id,
                    request_summary=req.request_summary,
                    best_item_id=best_item.item_id,
                    best_check_text=best_item.check_text,
                    similarity=similarity,
                    action=ACTION_LINKED,
                )
            )
            continue

        new_item_id = None
        if save:
            new_item_id = repo.next_item_id(con)
            repo.insert_knowledge_item(
                con,
                repo.KnowledgeItemRow(
                    item_id=new_item_id,
                    artifact_type=req.artifact_type,
                    check_text=req.pass_criterion or req.request_summary,
                    pass_criterion=req.pass_criterion,
                    # LLM が推測で埋めていないので、NULL のままになることが多い。それが正しい状態である。
                    reason=req.stated_reason,
                    applies_when=req.applies_when,
                    not_applies_when=None,
                    # 重要度の判定は Discord への投稿時に行う（第2弾）。この段階では NULL。
                    importance=None,
                    status=repo.STATUS_CANDIDATE,
                    version=1,
                    updated_at=now,
                ),
            )
            repo.add_evidence(con, new_item_id, req.request_id, None, now)
        results.append(
            MatchResult(
                request_id=req.request_id,
                request_summary=req.request_summary,
                best_item_id=best_item.item_id if best_item else None,
                best_check_text=best_item.check_text if best_item else None,
                similarity=similarity,
                action=ACTION_NEW_CANDIDATE,
                new_item_id=new_item_id,
            )
        )

    if dry_run:
        results.sort(key=lambda r: (r.similarity is None, -(r.similarity or 0.0)))
    return results
