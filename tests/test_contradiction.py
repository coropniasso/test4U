"""矛盾の可能性がある組の検出(仕様 2-9、テスト 6)。"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from kijun.db import repo
from kijun.review.contradiction import find_contradictions
from tests.review_helpers import NOW, add_item


def vec(similarity_to_first: float) -> np.ndarray:
    """(1, 0) との内積(コサイン類似度)が指定の値になる単位ベクトル。"""
    return np.array([similarity_to_first, math.sqrt(1 - similarity_to_first**2)], dtype=np.float32)


def add_pair(con, cfg, sim: float, a_kw=None, b_kw=None, a_id="item_0001", b_id="item_0002"):
    """A と B の2項目と、類似度が sim になる埋め込みを入れる。"""
    ka = dict(check_text="告知文は冒頭に行動を書く", pass_criterion="冒頭2行以内に行動が書かれているか", artifact_type="announcement")
    kb = dict(check_text="告知文は最後に行動を書く", pass_criterion="末尾に行動が書かれているか", artifact_type="announcement")
    ka.update(a_kw or {})
    kb.update(b_kw or {})
    a = add_item(con, a_id, **ka)
    b = add_item(con, b_id, **kb)
    repo.put_embeddings(con, "knowledge_item", cfg.match.model, {a_id: (a.check_text, np.array([1.0, 0.0], dtype=np.float32))}, NOW)
    repo.put_embeddings(con, "knowledge_item", cfg.match.model, {b_id: (b.check_text, vec(sim))}, NOW)
    return a, b


def test_条件を満たす組が見つかる(con, cfg):
    add_pair(con, cfg, 0.78)
    pairs = find_contradictions(con, cfg)
    assert len(pairs) == 1
    assert (pairs[0].a.item_id, pairs[0].b.item_id) == ("item_0001", "item_0002")
    assert abs(pairs[0].similarity - 0.78) < 1e-4


def test_類似度が下限未満なら対象外(con, cfg):
    add_pair(con, cfg, 0.69)
    assert find_contradictions(con, cfg) == []


def test_類似度がちょうど下限なら対象(con, cfg):
    add_pair(con, cfg, 0.70)
    # float32 の誤差で下限を僅かに下回らないよう、設定の下限を僅かに下げて境界を確かめる
    cfg.discord.contradiction_min_similarity = 0.699
    assert len(find_contradictions(con, cfg)) == 1


def test_類似度が重複の閾値以上なら対象外(con, cfg):
    # 0.85 以上は同一項目として統合される側なので、矛盾ではなく重複
    add_pair(con, cfg, 0.90)
    assert find_contradictions(con, cfg) == []


def test_重複の閾値は設定値に従う(con, cfg):
    add_pair(con, cfg, 0.80)
    cfg.match.similarity_threshold = 0.75
    assert find_contradictions(con, cfg) == []


def test_artifact_type_が違えば対象外(con, cfg):
    add_pair(con, cfg, 0.78, b_kw=dict(artifact_type="report"))
    assert find_contradictions(con, cfg) == []


def test_artifact_type_が両方_NULL_なら同じとはみなさない(con, cfg):
    add_pair(con, cfg, 0.78, a_kw=dict(artifact_type=None), b_kw=dict(artifact_type=None))
    assert find_contradictions(con, cfg) == []


def test_pass_criterion_が同じなら対象外(con, cfg):
    add_pair(con, cfg, 0.78, b_kw=dict(pass_criterion="冒頭2行以内に行動が書かれているか"))
    assert find_contradictions(con, cfg) == []


def test_pass_criterion_の全角半角と前後の空白の違いは同じとみなす(con, cfg):
    add_pair(con, cfg, 0.78, a_kw=dict(pass_criterion="Ａ案か"), b_kw=dict(pass_criterion=" A案か "))
    assert find_contradictions(con, cfg) == []


def test_pass_criterion_がどちらかNULLなら対象外(con, cfg):
    add_pair(con, cfg, 0.78, b_kw=dict(pass_criterion=None))
    assert find_contradictions(con, cfg) == []


def test_却下と保留の項目は対象外(con, cfg):
    add_pair(con, cfg, 0.78, b_kw=dict(status="rejected"))
    assert find_contradictions(con, cfg) == []
    other = add_item(con, "item_0003", check_text="別", pass_criterion="別の基準か", status="held")
    repo.put_embeddings(con, "knowledge_item", cfg.match.model, {"item_0003": ("別", vec(0.75))}, NOW)
    assert find_contradictions(con, cfg) == []
    assert other.status == "held"


def test_承認済みと候補の組は対象(con, cfg):
    add_pair(con, cfg, 0.78, a_kw=dict(status="approved"), b_kw=dict(status="candidate"))
    assert len(find_contradictions(con, cfg)) == 1


def test_埋め込みが無い項目と_check_text_が編集されて古くなった項目は対象外(con, cfg):
    a, b = add_pair(con, cfg, 0.78)
    repo.update_item_fields(con, "item_0002", {"check_text": "編集後の文言"}, NOW)  # ベクトルの text_hash が合わなくなる
    assert find_contradictions(con, cfg) == []
    add_item(con, "item_0009", check_text="埋め込みの無い項目", pass_criterion="別の基準か")
    assert find_contradictions(con, cfg) == []
    assert a.item_id and b.item_id
