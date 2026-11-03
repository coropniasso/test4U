"""矛盾の可能性がある組の投稿(仕様 2-9 の条件5、1回の投稿数の上限)。検出そのものは test_contradiction.py。"""

from __future__ import annotations

import math

import numpy as np

from kijun.db import repo
from kijun.discord.post import run_post
from kijun.review.contradiction import find_contradictions
from tests.fakes import FakeDiscordClient
from tests.review_helpers import NOW, add_item
from tests.test_contradiction import add_pair, vec  # noqa: F401


def test_既に投稿した組は再投稿されない_順序を問わない(con, cfg):
    add_pair(con, cfg, 0.78)
    cfg.discord.review_channel_id, cfg.discord.contradictions_channel_id = "111", "222"
    client = FakeDiscordClient()

    first = run_post(con, cfg, client, None, NOW)
    assert first.posted.get("contradiction") == 1
    post = repo.list_discord_posts(con, "contradiction")[0]
    assert (post.item_id, post.pair_item_id, post.channel_id) == ("item_0001", "item_0002", "222")
    assert find_contradictions(con, cfg) == []

    # A と B を入れ替えて記録されていても、同じ組とみなす
    con.execute("UPDATE discord_posts SET item_id = 'item_0002', pair_item_id = 'item_0001'")
    assert find_contradictions(con, cfg) == []
    second = run_post(con, cfg, client, None, NOW)
    assert second.posted.get("contradiction") is None


def test_1回の投稿数は上限で制限され_残りは次回に回る(con, cfg):
    cfg.discord.review_channel_id, cfg.discord.contradictions_channel_id = "111", "222"
    cfg.discord.max_contradictions_per_run = 2
    # 4項目が互いに 0.78 前後 -> 6組。ここでは3項目で3組を作る
    items = {}
    for n, v in [(1, np.array([1.0, 0.0, 0.0])), (2, np.array([0.78, math.sqrt(1 - 0.78**2), 0.0])), (3, np.array([0.78, 0.2, math.sqrt(1 - 0.78**2 - 0.04)]))]:
        item = add_item(con, f"item_000{n}", check_text=f"文言{n}", pass_criterion=f"基準{n}か")
        repo.put_embeddings(con, "knowledge_item", cfg.match.model, {item.item_id: (item.check_text, v.astype(np.float32))}, NOW)
        items[n] = item
    assert len(find_contradictions(con, cfg)) == 3
    client = FakeDiscordClient()

    assert run_post(con, cfg, client, None, NOW).posted.get("contradiction") == 2
    assert run_post(con, cfg, client, None, NOW).posted.get("contradiction") == 1  # 残りが次回に出る
    assert run_post(con, cfg, client, None, NOW).posted.get("contradiction") is None


def test_類似度の高い組が先に投稿される(con, cfg):
    cfg.discord.review_channel_id, cfg.discord.contradictions_channel_id = "111", "222"
    cfg.discord.max_contradictions_per_run = 1
    # 1-2 の組は 0.72、5-6 の組は 0.82(ほかの組み合わせは 0.7 未満)
    for n, v in [(1, [1, 0, 0]), (2, [0.72, math.sqrt(1 - 0.72**2), 0]), (5, [0, 0, 1]), (6, [0, math.sqrt(1 - 0.82**2), 0.82])]:
        item = add_item(con, f"item_000{n}", check_text=f"文言{n}", pass_criterion=f"基準{n}か")
        repo.put_embeddings(con, "knowledge_item", cfg.match.model, {item.item_id: (item.check_text, np.array(v, dtype=np.float32))}, NOW)
    client = FakeDiscordClient()
    run_post(con, cfg, client, None, NOW)
    text = [c for c in client.contents() if c.startswith("【矛盾の可能性】")][0]
    assert "#0005 と #0006" in text  # 0.82 の組が先
