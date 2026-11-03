"""矛盾の可能性がある組み合わせの検出(仕様 2-9)。

**この検出規則は CLAUDE.md に書かれていない。設計者(Claude)が考案したもので、推奨・未確定である。**
運用して見直す前提で、閾値はすべて設定値にしてある(discord.contradiction_min_similarity、
match.similarity_threshold、discord.max_contradictions_per_run)。

目的は矛盾を証明することではなく、同じ話題で基準が食い違っている組を人に見せること。
CLAUDE.md 4-6: 「多くは矛盾ではなく適用条件の違いであり、適用条件を言語化する材料になる」。

knowledge_items のうち status が approved または candidate の組 (A, B) で、次のすべてを満たすものを返す。
1. artifact_type が同じ。両方が NULL の場合は同じとみなさない(種類が分からないものどうしを同じ成果物と決めつけない)。
2. check_text のコサイン類似度が discord.contradiction_min_similarity 以上。
3. 同じ類似度が match.similarity_threshold 未満(以上は同一項目として統合される側なので、矛盾ではなく重複)。
4. pass_criterion が両方とも非 NULL で、文言が異なる(NFKC で正規化し、前後の空白を除いて比べる)。
5. まだ discord_posts に post_type = 'contradiction' として投稿されていない組(A と B の順序は問わない)。

ベクトルは既存の embeddings から取る。無い項目、または check_text が編集されてベクトルが古い項目は対象外にする
(この処理のために埋め込みモデルを読み込まない。原則2-2の VRAM の都合)。
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np

from kijun.config import Config
from kijun.db import repo
from kijun.db.repo import KnowledgeItemRow
from kijun.match.link import cosine_similarity_matrix


@dataclass(frozen=True)
class ContradictionPair:
    a: KnowledgeItemRow  # item_id が小さい方
    b: KnowledgeItemRow
    similarity: float


def _same_text(x: str, y: str) -> bool:
    return repo.text_hash(x.strip()) == repo.text_hash(y.strip())


def posted_pairs(con: duckdb.DuckDBPyConnection) -> set[frozenset[str]]:
    """既に contradiction として投稿した組。"""
    return {
        frozenset((p.item_id, p.pair_item_id))
        for p in repo.list_discord_posts(con, repo.POST_CONTRADICTION)
        if p.item_id and p.pair_item_id
    }


def find_contradictions(con: duckdb.DuckDBPyConnection, cfg: Config) -> list[ContradictionPair]:
    """条件を満たす組を、類似度の高い順に返す。件数の上限は呼び出し側(post)が適用する。"""
    items = repo.list_knowledge_items(con, [repo.STATUS_APPROVED, repo.STATUS_CANDIDATE])
    vectors = repo.get_embeddings(
        con, repo.OWNER_KNOWLEDGE_ITEM, {i.item_id: i.check_text for i in items}, cfg.match.model
    )
    usable = [i for i in items if i.item_id in vectors]
    if len(usable) < 2:
        return []
    sims = cosine_similarity_matrix(
        np.stack([vectors[i.item_id] for i in usable]), np.stack([vectors[i.item_id] for i in usable])
    )
    low = cfg.discord.contradiction_min_similarity
    high = cfg.match.similarity_threshold
    already = posted_pairs(con)

    pairs: list[ContradictionPair] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]  # list_knowledge_items は item_id 順なので a < b
            sim = float(sims[i][j])
            if not (low <= sim < high):
                continue
            if a.artifact_type is None or a.artifact_type != b.artifact_type:
                continue
            if a.pass_criterion is None or b.pass_criterion is None:
                continue
            if _same_text(a.pass_criterion, b.pass_criterion):
                continue
            if frozenset((a.item_id, b.item_id)) in already:
                continue
            pairs.append(ContradictionPair(a, b, sim))
    pairs.sort(key=lambda p: (-p.similarity, p.a.item_id, p.b.item_id))
    return pairs
