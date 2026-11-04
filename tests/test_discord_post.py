"""Discord への投稿(仕様 1-4、2-3、テスト 7)。"""

from __future__ import annotations

from datetime import datetime

from kijun.db import repo
from kijun.discord.post import count_candidates_needing_importance, run_post
from tests.conftest import make_conversation
from tests.fakes import FakeDiscordClient, FakeImportanceJudge
from tests.review_helpers import CID, NOW, add_item, add_request, candidate, setup_conversation


def configure(cfg, review="111", contra="222", max_candidates=20):
    cfg.discord.review_channel_id = review
    cfg.discord.contradictions_channel_id = contra
    cfg.discord.max_candidates_per_run = max_candidates
    return cfg


def candidate_messages(client: FakeDiscordClient) -> list[str]:
    return [c for c in client.contents() if c.startswith("【新規候補 #")]


def numbers(client: FakeDiscordClient) -> list[str]:
    """投稿した新規候補の項目番号を、投稿順に返す。"""
    out = []
    for c in client.contents():
        if c.startswith("【新規候補 #"):
            out.append(c.split("#")[1].split("】")[0])
    return out


def test_重要度の降順_同じ重要度なら新しい順に投稿する(con, cfg):
    configure(cfg)
    base = datetime(2026, 11, 3, 12, 0)
    candidate(con, 1, importance=1, linked_at=base)
    candidate(con, 2, importance=3, linked_at=base)
    candidate(con, 3, importance=3, linked_at=datetime(2026, 11, 3, 13, 0))  # 重要度3のうち新しい
    candidate(con, 4, importance=2, linked_at=base)
    client = FakeDiscordClient()

    run_post(con, cfg, client, None, NOW)

    assert numbers(client) == ["0003", "0002", "0004", "0001"]


def test_上限を超えた分は投稿されず_次回に回る(con, cfg):
    configure(cfg, max_candidates=2)
    for n, imp in [(1, 1), (2, 3), (3, 2), (4, 3)]:
        candidate(con, n, importance=imp)
    client = FakeDiscordClient()

    first = run_post(con, cfg, client, None, NOW)

    assert len(numbers(client)) == 2
    assert set(numbers(client)) == {"0002", "0004"}  # 重要度の高い2件が先に出る
    assert first.not_posted_candidates == 2  # 投稿していない件数を報告する
    assert {p.item_id for p in repo.list_discord_posts(con, "new_candidate")} == {"item_0002", "item_0004"}

    second = run_post(con, cfg, client, None, NOW)  # 次回に残りが出る。投稿済みは再投稿しない
    assert set(numbers(client)) == {"0001", "0002", "0003", "0004"} and len(numbers(client)) == 4
    assert second.not_posted_candidates == 0

    third = run_post(con, cfg, client, None, NOW)
    assert len(numbers(client)) == 4 and third.posted == {}


def test_投稿の記録にチャンネルIDが残る(con, cfg):
    configure(cfg, review="111", contra="222")
    candidate(con, 1, importance=2)
    run_post(con, cfg, FakeDiscordClient(), None, NOW)
    post = repo.list_discord_posts(con, "new_candidate")[0]
    assert post.channel_id == "111" and post.item_id == "item_0001" and post.processed_at is None
    assert post.message_id.startswith("m") and post.conversation_id == CID


def test_重要度が未判定の候補はLLMで判定して保存し_根拠を_review_log_に残す(con, cfg):
    configure(cfg)
    candidate(con, 1, importance=None, check_text="項目A")
    candidate(con, 2, importance=3, check_text="項目B")  # 判定済みは再判定しない
    judge = FakeImportanceJudge(importance=2)
    assert count_candidates_needing_importance(con) == 1

    run_post(con, cfg, FakeDiscordClient(), judge, NOW)

    assert [c[0] for c in judge.calls] == ["項目A"]
    assert judge.calls[0][1] == ["発言1"]  # 根拠の発言を渡す
    assert repo.get_knowledge_item(con, "item_0001").importance == 2
    log = repo.list_review_log(con, "item_0001", "importance_proposed")
    assert [(r.before, r.after) for r in log] == [(None, "根拠:項目A")]
    assert count_candidates_needing_importance(con) == 0


def test_重要度の判定に失敗しても投稿し_重要度案はなしと出る(con, cfg):
    configure(cfg)
    candidate(con, 1, importance=None, check_text="項目A")
    client = FakeDiscordClient()

    report = run_post(con, cfg, client, FakeImportanceJudge(fail_for={"項目A"}), NOW)

    assert "重要度案：なし" in candidate_messages(client)[0]
    assert repo.get_knowledge_item(con, "item_0001").importance is None
    assert not report.failed  # 判定の失敗は投稿の失敗ではない


def test_LLMを使えないときも投稿できる(con, cfg):
    configure(cfg)
    candidate(con, 1, importance=None)
    client = FakeDiscordClient()
    run_post(con, cfg, client, None, NOW)
    assert "重要度案：なし" in candidate_messages(client)[0]


def test_status_が候補でない項目は投稿しない(con, cfg):
    configure(cfg)
    candidate(con, 1, importance=2, status="approved")
    candidate(con, 2, importance=2, status="rejected")
    client = FakeDiscordClient()
    run_post(con, cfg, client, None, NOW)
    assert numbers(client) == []


def test_2000文字を超えて収まらない文面は投稿せず_失敗として報告する(con, cfg):
    configure(cfg)
    candidate(con, 1, importance=2, check_text="あ" * 2500)
    candidate(con, 2, importance=1)
    client = FakeDiscordClient()

    report = run_post(con, cfg, client, None, NOW)

    assert numbers(client) == ["0002"]  # 他の候補は投稿される
    assert report.failed and "上限" in report.failures[0] and "item_0001" in report.failures[0]
    assert report.not_posted_candidates == 1  # 投稿できなかった分は未投稿として残る


def test_1件の投稿の失敗で全体を止めない(con, cfg):
    configure(cfg)
    candidate(con, 1, importance=3, check_text="失敗させる項目")
    candidate(con, 2, importance=2)
    client = FakeDiscordClient(fail_post_when=lambda c: "失敗させる項目" in c)

    report = run_post(con, cfg, client, None, NOW)

    assert numbers(client) == ["0002"] and report.failed
    assert [p.item_id for p in repo.list_discord_posts(con, "new_candidate")] == ["item_0002"]  # 失敗は記録しない


def test_話者の対応付けは会議ごとに1回だけ投稿する(con, cfg):
    configure(cfg)
    setup_conversation(con)
    client = FakeDiscordClient()

    first = run_post(con, cfg, client, None, NOW)
    second = run_post(con, cfg, client, None, NOW)

    samples = [c for c in client.contents() if c.startswith("【話者の対応付け】")]
    assert len(samples) == 1 and "00=self 01=kouhai_a" in samples[0]
    assert first.posted == {"speaker_sample": 1} and second.posted == {}
    post = repo.list_discord_posts(con, "speaker_sample")[0]
    assert post.conversation_id == CID and post.channel_id == "111"


def test_self_が決まっている会議には話者の対応付けを投稿しない(con, cfg):
    configure(cfg)
    setup_conversation(con)
    repo.set_speaker_map(con, CID, "SPEAKER_00", "self")
    client = FakeDiscordClient()
    run_post(con, cfg, client, None, NOW)
    assert client.contents() == []


def test_束ね結果は複数ファイルの会議だけ投稿する(con, cfg):
    configure(cfg)
    repo.insert_conversation(con, make_conversation("single"))
    repo.insert_conversation(con, make_conversation("bundled", source_files=["a.m4a", "b.m4a"]))
    client = FakeDiscordClient()
    run_post(con, cfg, client, None, NOW)
    run_post(con, cfg, client, None, NOW)
    bundles = [c for c in client.contents() if c.startswith("【束ね結果】")]
    assert len(bundles) == 1 and "bundled" in bundles[0]


def test_チャンネルIDが空なら何もせず_設定が必要だと出す(con, cfg):
    candidate(con, 1, importance=2)
    client = FakeDiscordClient()
    report = run_post(con, cfg, client, None, NOW)
    assert client.posted == [] and report.skipped_for_config
    assert "設定が必要" in report.messages[0]


def test_review_だけ空なら矛盾だけ_contradictions_だけ空なら候補だけ投稿する(con, cfg):
    configure(cfg, review="", contra="222")
    candidate(con, 1, importance=2)
    client = FakeDiscordClient()
    report = run_post(con, cfg, client, None, NOW)
    assert numbers(client) == [] and any("review_channel_id が空" in m for m in report.messages)

    configure(cfg, review="111", contra="")
    report = run_post(con, cfg, client, None, NOW)
    assert numbers(client) == ["0001"] and any("contradictions_channel_id が空" in m for m in report.messages)


# --- 文字起こしが読めなかった抽出結果（CLAUDE.md 7章の案A） ---------------------------------
#
# これらは reusable = False なので照合の対象から外れ、knowledge_items にならない。投稿するのは、
# 発話したのが利用者本人であり、壊れた引用を見れば自分が何と言ったか分かるためである。
# 投稿しなければ、同じ発話が繰り返し失われ続ける（「箇条書き」「感嘆符」は頻出語である）。

UNCLEAR_HEAD = "【文字起こしが読めませんでした】"


def unclear_messages(client: FakeDiscordClient) -> list[str]:
    return [c for c in client.contents() if c.startswith(UNCLEAR_HEAD)]


def add_unclear(con, request_id: str, quote: str, model: str, **kw):
    return add_request(
        con, request_id, quote=quote, model=model, reusable=False,
        transcription_unclear=True, **kw,
    )


def test_文字起こしが読めなかった抽出結果を_review_に投稿する(con, cfg):
    configure(cfg)
    cfg.extract.strata.model = "m"
    setup_conversation(con)
    add_unclear(con, "r1", "3つ目簡単フォーイスなんか減らしてください", "m",
                summary="「簡単フォーイス」を減らすことを求めた")
    client = FakeDiscordClient()

    run_post(con, cfg, client, None, NOW)

    posted = unclear_messages(client)
    assert len(posted) == 1
    # 引用は直さずそのまま出す。直すのは利用者で、そのための投稿である
    assert "簡単フォーイス" in posted[0]
    # チェック項目案は書かない。書けないことがこの投稿の内容である
    assert "チェック項目案" not in posted[0]


def test_投稿は_request_id_で記録され_次回は投稿しない(con, cfg):
    # discord_posts に item_id では記録できない（knowledge_items にならないため）。
    configure(cfg)
    cfg.extract.strata.model = "m"
    setup_conversation(con)
    add_unclear(con, "r1", "過剰書きにして", "m")
    client = FakeDiscordClient()

    run_post(con, cfg, client, None, NOW)
    rows = repo.list_discord_posts(con, repo.POST_TRANSCRIPTION_UNCLEAR)
    assert [p.request_id for p in rows] == ["r1"]
    assert rows[0].item_id is None and rows[0].conversation_id == CID

    run_post(con, cfg, client, None, NOW)  # 2回目
    assert len(unclear_messages(client)) == 1  # 増えない


def test_上限の件数までしか投稿しない(con, cfg):
    configure(cfg)
    cfg.extract.strata.model = "m"
    cfg.discord.max_transcription_unclear_per_run = 2
    setup_conversation(con)
    for i in range(5):
        add_unclear(con, f"r{i}", f"壊れた引用{i}", "m")
    client = FakeDiscordClient()

    run_post(con, cfg, client, None, NOW)

    assert len(unclear_messages(client)) == 2


def test_別のモデルで抽出した分は投稿しない(con, cfg):
    # 設定を変えて2回抽出した場合、いまの設定が指すモデルの分だけを投稿する。
    configure(cfg)
    cfg.extract.strata.model = "m"
    setup_conversation(con)
    add_unclear(con, "r1", "いまの設定のモデル", "m")
    add_unclear(con, "r2", "別の設定のモデル", "other")
    client = FakeDiscordClient()

    run_post(con, cfg, client, None, NOW)

    posted = unclear_messages(client)
    assert len(posted) == 1 and "いまの設定のモデル" in posted[0]


def test_文字起こしが読めた抽出結果は投稿しない(con, cfg):
    configure(cfg)
    cfg.extract.strata.model = "m"
    setup_conversation(con)
    add_request(con, "r1", quote="結論を先に書いて", model="m")  # transcription_unclear は False
    client = FakeDiscordClient()

    run_post(con, cfg, client, None, NOW)

    assert unclear_messages(client) == []


def test_チェック項目として使える行は_読めなかった側には投稿しない(con, cfg):
    """二重投稿を止める（2026-11-04 実機で見つかった不具合）。

    「文字起こしが壊れていること」と「チェック項目として使えること」は独立している。実機では、
    「箇条書き」が「過剰書き」に化けた発話に対して、モデルが transcription_unclear = true と
    judge したうえで、文脈から趣旨を取り pass_criterion まで作った（reusable = true）。
    両方の経路に流すと、同じ発話が #review に2回出る。そして新規候補の文面には壊れた引用が
    入っているので、利用者は既にそれを見てスレッドで直せる。この種別の投稿は何も足さない。
    """
    configure(cfg)
    cfg.extract.strata.model = "m"
    setup_conversation(con)
    add_request(
        con, "r1", quote="過剰書きでいいんで見出しのすぐ下に書いて", model="m",
        reusable=True, transcription_unclear=True,
    )
    client = FakeDiscordClient()

    run_post(con, cfg, client, None, NOW)

    assert unclear_messages(client) == []
    assert repo.list_discord_posts(con, repo.POST_TRANSCRIPTION_UNCLEAR) == []


def test_読めずチェック項目にもならない行だけを投稿する(con, cfg):
    configure(cfg)
    cfg.extract.strata.model = "m"
    setup_conversation(con)
    add_request(con, "r1", quote="過剰書きで書いて", model="m",
                reusable=True, transcription_unclear=True)   # 新規候補の側に出る
    add_request(con, "r2", quote="3つ目簡単フォーイスなんか減らして", model="m",
                reusable=False, transcription_unclear=True)  # こちらだけ
    client = FakeDiscordClient()

    run_post(con, cfg, client, None, NOW)

    posted = unclear_messages(client)
    assert len(posted) == 1 and "簡単フォーイス" in posted[0]
