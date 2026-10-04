from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from kijun.db import repo
from kijun.extract.run import run_extraction
from kijun.match.link import cosine_similarity_matrix, link_requests
from tests.conftest import make_conversation, make_utterances
from tests.fakes import FakeEmbedder, make_item

CID = "20260920_1930_f2f_teirei"
NOW = datetime(2026, 9, 21, 0, 0)

SUMMARY_A = "告知文の冒頭に、読み手にしてほしい行動を書くこと"
SUMMARY_B = "資料のグラフには、単位と出典を必ず付けること"


def add_request(con, request_id: str, summary: str, reusable: bool = True, model: str = "m1", **kw) -> None:
    repo.insert_requests(
        con,
        [
            repo.RequestRow(
                request_id=request_id, conversation_id=CID, seq_from=1, seq_to=1,
                speaker_from_label="SPEAKER_00", speaker_to_label="SPEAKER_01", direction=None,
                counterpart_role=None, artifact_type=kw.get("artifact_type"), quote="引用",
                request_summary=summary, stated_reason=kw.get("reason"), applies_when=kw.get("applies_when"),
                reusable=reusable, pass_criterion=kw.get("criterion"),
                source_type="live_conversation", model_name=model, extracted_at=NOW,
            )
        ],
    )


def add_item(con, item_id: str, check_text: str, status: str = "approved") -> None:
    repo.insert_knowledge_item(
        con, repo.KnowledgeItemRow(item_id, None, check_text, None, None, None, None, 2, status, 1, NOW)
    )


@pytest.fixture
def conv(con):
    repo.insert_conversation(con, make_conversation(CID))
    return CID


# --- FakeEmbedder の性質 -------------------------------------------------------------------


def test_FakeEmbedder_同じ文字列には同じベクトル_似た文字列には似たベクトル():
    e = FakeEmbedder()
    a = e.encode_queries([SUMMARY_A])[0]
    a2 = e.encode_passages([SUMMARY_A])[0]
    similar = e.encode_passages([SUMMARY_A + "（再掲）"])[0]
    different = e.encode_passages([SUMMARY_B])[0]
    assert np.array_equal(a, a2)
    assert np.array_equal(e.encode_queries([SUMMARY_A])[0], a)  # 何度呼んでも同じ
    assert float(a @ similar) > 0.8
    assert float(a @ different) < 0.4
    assert float(a @ similar) > float(a @ different)


def test_FakeEmbedder_の形状():
    e = FakeEmbedder(dim=64)
    assert e.encode_queries(["a", "b", "c"]).shape == (3, 64)
    assert e.encode_passages([]).shape == (0, 64)


def test_cosine_similarity_matrix():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[3.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    m = cosine_similarity_matrix(a, b)
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1.0)  # 長さが違っても向きが同じなら 1
    assert m[1, 0] == pytest.approx(0.0)
    assert m[0, 1] == 0.0  # 長さ 0 のベクトルは 0
    assert m[0, 2] == pytest.approx(1 / np.sqrt(2))
    assert cosine_similarity_matrix(np.zeros((0, 2)), b).shape == (0, 3)


# --- 閾値による分岐 ------------------------------------------------------------------------


def test_閾値以上なら_item_evidence_に紐付く(con, cfg, conv):
    add_item(con, "item_0001", SUMMARY_A)
    add_request(con, "r1", SUMMARY_A)  # 同じ文面なので類似度 1.0
    results = link_requests(con, cfg, conv, FakeEmbedder(), now=NOW)

    assert [(r.action, r.best_item_id) for r in results] == [("linked", "item_0001")]
    assert results[0].similarity == pytest.approx(1.0)
    assert con.execute("SELECT item_id, request_id, similarity FROM item_evidence").fetchall() == [
        ("item_0001", "r1", pytest.approx(1.0))
    ]
    assert len(repo.list_knowledge_items(con)) == 1  # 新規候補は作らない


def test_閾値未満なら_candidate_の行ができる(con, cfg, conv):
    add_item(con, "item_0001", SUMMARY_B)
    add_request(con, "r1", SUMMARY_A, criterion="冒頭2行以内に、読み手にしてほしい行動が書かれているか",
                reason="読む人は忙しいから", applies_when="告知文のとき", artifact_type="announcement")
    results = link_requests(con, cfg, conv, FakeEmbedder(), now=NOW)

    assert results[0].action == "new_candidate"
    assert results[0].best_item_id == "item_0001"  # 最類似は分かるが、閾値に届かない
    assert results[0].similarity < 0.85
    new = {i.item_id: i for i in repo.list_knowledge_items(con)}["item_0002"]
    assert new.status == "candidate"
    assert new.version == 1
    assert new.importance is None
    assert new.check_text == "冒頭2行以内に、読み手にしてほしい行動が書かれているか"  # pass_criterion
    assert new.pass_criterion == new.check_text
    assert new.reason == "読む人は忙しいから"  # stated_reason をそのまま入れる
    assert new.applies_when == "告知文のとき"
    assert new.artifact_type == "announcement"
    assert new.updated_at == NOW
    assert con.execute("SELECT item_id, request_id, similarity FROM item_evidence").fetchall() == [
        ("item_0002", "r1", None)  # 新規候補の similarity は NULL
    ]


def test_pass_criterion_が無ければ_check_text_は要約になり_理由と条件は_NULL_のまま(con, cfg, conv):
    add_request(con, "r1", SUMMARY_A)
    link_requests(con, cfg, conv, FakeEmbedder(), now=NOW)
    item = repo.list_knowledge_items(con)[0]
    assert item.check_text == SUMMARY_A
    assert item.reason is None and item.applies_when is None  # 推測で埋めない


def test_knowledge_items_が0件の初回は全件が新規候補になる(con, cfg, conv):
    # 内容がほぼ同じ2件でも、既存項目が0件なので、どちらも新規候補になる
    add_request(con, "r1", SUMMARY_A)
    add_request(con, "r2", SUMMARY_A)
    add_request(con, "r3", SUMMARY_B)
    results = link_requests(con, cfg, conv, FakeEmbedder(), now=NOW)

    assert [r.action for r in results] == ["new_candidate"] * 3
    assert all(r.best_item_id is None and r.similarity is None for r in results)
    items = repo.list_knowledge_items(con)
    assert [i.item_id for i in items] == ["item_0001", "item_0002", "item_0003"]
    assert {i.status for i in items} == {"candidate"}
    assert con.execute("SELECT count(*) FROM item_evidence WHERE similarity IS NULL").fetchone()[0] == 3


def test_閾値ちょうどは紐付き_わずかに超えれば新規候補(con, cfg, conv):
    e = FakeEmbedder()
    item_text, summary = SUMMARY_A, SUMMARY_A + "（再掲）"
    exact = float(cosine_similarity_matrix(e.encode_queries([summary]), e.encode_passages([item_text]))[0, 0])
    add_item(con, "item_0001", item_text)
    add_request(con, "r1", summary)

    cfg.match.similarity_threshold = exact  # 閾値ちょうど（以上なので紐付く）
    r = link_requests(con, cfg, conv, FakeEmbedder(), dry_run=True, now=NOW)
    assert r[0].action == "linked"
    cfg.match.similarity_threshold = exact + 1e-6
    r = link_requests(con, cfg, conv, FakeEmbedder(), dry_run=True, now=NOW)
    assert r[0].action == "new_candidate"


def test_reusable_false_は照合されない(con, cfg, conv):
    add_item(con, "item_0001", SUMMARY_A)
    add_request(con, "r1", SUMMARY_A, reusable=False)
    embedder = FakeEmbedder()
    results = link_requests(con, cfg, conv, embedder, now=NOW)

    assert results == []
    assert con.execute("SELECT count(*) FROM item_evidence").fetchone()[0] == 0
    assert len(repo.list_knowledge_items(con)) == 1  # 新規候補も作らない
    assert embedder.query_texts == []  # ベクトル化もしない
    assert con.execute("SELECT count(*) FROM embeddings WHERE owner_id = 'r1'").fetchone()[0] == 0


def test_rejected_の項目とは照合しない(con, cfg, conv):
    add_item(con, "item_0001", SUMMARY_A, status="rejected")
    add_request(con, "r1", SUMMARY_A)
    results = link_requests(con, cfg, conv, FakeEmbedder(), now=NOW)
    assert results[0].action == "new_candidate"  # 同じ文面でも、却下済みには紐付けない
    assert results[0].best_item_id is None


@pytest.mark.parametrize("status", ["candidate", "approved", "held"])
def test_candidate_approved_held_の項目とは照合する(con, cfg, conv, status):
    add_item(con, "item_0001", SUMMARY_A, status=status)
    add_request(con, "r1", SUMMARY_A)
    assert link_requests(con, cfg, conv, FakeEmbedder(), now=NOW)[0].action == "linked"


# --- ベクトルの保存と再利用 ----------------------------------------------------------------


def test_ベクトルが_embeddings_に保存され_2回目は再計算しない(con, cfg, conv):
    add_item(con, "item_0001", SUMMARY_B)
    add_request(con, "r1", SUMMARY_A)
    embedder = FakeEmbedder()
    link_requests(con, cfg, conv, embedder, now=NOW)

    rows = con.execute(
        "SELECT owner_type, owner_id, model_name, dim FROM embeddings ORDER BY owner_type, owner_id"
    ).fetchall()
    assert rows == [
        ("extracted_request", "r1", "fake-embedder", 256),
        ("knowledge_item", "item_0001", "fake-embedder", 256),
    ]
    assert embedder.query_texts == [SUMMARY_A] and embedder.passage_texts == [SUMMARY_B]

    # 2回目: 抽出結果を未紐付けに戻して照合し直す。保存済みのベクトルがあるので、埋め込みは呼ばれない
    con.execute("DELETE FROM item_evidence")
    con.execute("DELETE FROM knowledge_items WHERE item_id != 'item_0001'")
    before = embedder.encoded_count
    link_requests(con, cfg, conv, embedder, now=NOW)
    assert embedder.encoded_count == before


def test_モデル名が違うベクトルは使わず再計算する(con, cfg, conv):
    add_item(con, "item_0001", SUMMARY_B)
    add_request(con, "r1", SUMMARY_A)
    link_requests(con, cfg, conv, FakeEmbedder(model_name="model-a"), now=NOW)
    con.execute("DELETE FROM item_evidence")
    con.execute("DELETE FROM knowledge_items WHERE item_id != 'item_0001'")

    other = FakeEmbedder(dim=32, model_name="model-b")
    link_requests(con, cfg, conv, other, now=NOW)
    assert other.query_texts == [SUMMARY_A] and other.passage_texts == [SUMMARY_B]
    dims = con.execute("SELECT model_name, dim FROM embeddings ORDER BY model_name, dim").fetchall()
    assert ("model-a", 256) in dims and ("model-b", 32) in dims


def test_紐付け済みの抽出結果は再実行で重複しない(con, cfg, conv):
    add_request(con, "r1", SUMMARY_A)
    link_requests(con, cfg, conv, FakeEmbedder(), now=NOW)
    assert link_requests(con, cfg, conv, FakeEmbedder(), now=NOW) == []
    assert len(repo.list_knowledge_items(con)) == 1


# --- dry-run・モデルの絞り込み ---------------------------------------------------------------


def test_dry_run_は何も書かず_類似度の降順に並べる(con, cfg, conv):
    add_item(con, "item_0001", SUMMARY_A)
    add_request(con, "r_low", SUMMARY_B)
    add_request(con, "r_high", SUMMARY_A)
    embedder = FakeEmbedder()
    results = link_requests(con, cfg, conv, embedder, dry_run=True, now=NOW)

    assert [r.request_id for r in results] == ["r_high", "r_low"]
    assert results[0].similarity > results[1].similarity
    assert [r.action for r in results] == ["linked", "new_candidate"]
    assert results[1].new_item_id is None
    assert con.execute("SELECT count(*) FROM item_evidence").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert len(repo.list_knowledge_items(con)) == 1


def test_dry_run_で既存項目が無い抽出結果は末尾に並ぶ(con, cfg, conv):
    add_request(con, "r1", SUMMARY_A)
    r = link_requests(con, cfg, conv, FakeEmbedder(), dry_run=True, now=NOW)
    assert r[0].similarity is None and r[0].action == "new_candidate"


def test_model_name_を指定すると_そのモデルの抽出結果だけを照合する(con, cfg, conv):
    add_request(con, "qwen#1", SUMMARY_A, model="qwen")
    add_request(con, "claude#1", SUMMARY_B, model="claude")
    results = link_requests(con, cfg, conv, FakeEmbedder(), model_name="qwen", now=NOW)
    assert [r.request_id for r in results] == ["qwen#1"]


def test_抽出から照合までつながる(con, cfg, conv):
    from tests.fakes import FakeExtractor

    repo.replace_utterances(con, CID, make_utterances(5))
    extractor = FakeExtractor({0: [make_item(1, quote="q", summary=SUMMARY_A, reusable=True),
                                   make_item(2, quote="r", summary=SUMMARY_B, reusable=False)]})
    run_extraction(con, cfg, CID, extractor, now=NOW)
    results = link_requests(con, cfg, CID, FakeEmbedder(), now=NOW)
    assert len(results) == 1 and results[0].action == "new_candidate"
