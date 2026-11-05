"""Discord への議事録の投稿。DiscordClient は tests/fakes.py の FakeDiscordClient で差し替える。"""

from __future__ import annotations

from datetime import datetime

import pytest
from typer.testing import CliRunner

from kijun.db import repo
from kijun.discord.base import DiscordError
from kijun.discord.post import CONTINUED_NOTICE, DISCORD_LIMIT, build_post_text, run_post
from kijun.discord.run import TOKEN_ENV, post_command
from kijun.minutes.generate import generate_minutes
from kijun.minutes.models import ChunkMinutes, Overview, Todo, Topic
from tests.conftest import make_conversation, make_utterances
from tests.fakes import FakeDiscordClient, FakeJsonChat

NOW = datetime(2026, 11, 6, 9, 0)
CID = "20260920_1930_f2f_teirei"


def handler(system, user, model_cls, name):
    if model_cls is Overview:
        return Overview(overview="定例の日程を決めた会議。")
    return ChunkMinutes(
        topics=[Topic(title="日程", points=["水曜日の午後"])],
        decisions=["定例は水曜日の午後にする"],
        todos=[Todo(task="資料を用意する", owner="SPEAKER_01", due="金曜日"), Todo(task="会場を探す")],
        open_issues=["会場"],
    )


def make_minutes(con, cfg, cid=CID, chat_handler=handler):
    repo.insert_conversation(con, make_conversation(cid, transcribed_at=datetime(2026, 9, 21), duration_sec=600.0))
    repo.replace_utterances(con, cid, make_utterances(10))
    return generate_minutes(con, cfg, cid, FakeJsonChat(chat_handler), NOW, lambda m: None)


def markdown(cfg, cid=CID):
    return (cfg.paths.minutes / f"{cid}.md").read_text(encoding="utf-8")


# --- 本文 ----------------------------------------------------------------------------------------


def test_本文は見出し_概要_決定事項_ToDo(con, cfg):
    make_minutes(con, cfg)
    text = build_post_text(markdown(cfg), CID)
    assert text.startswith(f"**議事録 {CID}**（2026-09-20 19:30）")
    assert "**概要**\n定例の日程を決めた会議。" in text
    assert "**決定事項**\n- 定例は水曜日の午後にする" in text
    # Discord は Markdown の表を表示しないので、ToDo は箇条書きに直す
    assert "- 資料を用意する（担当: SPEAKER_01、期限: 金曜日）" in text
    assert "- 会場を探す（担当: 記載なし、期限: 記載なし）" in text
    assert "|" not in text
    # 議題・未決事項・文字起こしは本文に入れない(添付ファイルにある)
    assert "日程" not in text.split("**概要**")[0] and "未決" not in text and "発話0" not in text
    assert len(text) <= DISCORD_LIMIT


def test_2000文字を超えたら末尾を切って_続きは添付ファイルと書く(con, cfg):
    long_handler = lambda s, u, m, n: (  # noqa: E731
        Overview(overview="あ" * 300) if m is Overview else ChunkMinutes(decisions=[f"決定{i}" + "い" * 60 for i in range(60)])
    )
    make_minutes(con, cfg, chat_handler=long_handler)
    text = build_post_text(markdown(cfg), CID)
    assert len(text) <= DISCORD_LIMIT
    assert text.endswith(CONTINUED_NOTICE) and "（続きは添付ファイル）" in text
    assert text.startswith(f"**議事録 {CID}**")  # 先頭(見出し・概要)は残る


def test_ちょうど上限なら切らない():
    md = "# 議事録 x\n\n- 日時: 2026-09-20 19:30\n\n## 概要\n\n" + "あ" * 100 + "\n\n## 決定事項\n\nなし\n\n## ToDo\n\nなし\n"
    text = build_post_text(md, "x")
    assert CONTINUED_NOTICE not in text
    cut = build_post_text(md, "x", limit=len(text) - 1)
    assert cut.endswith(CONTINUED_NOTICE) and len(cut) <= len(text) - 1


def test_節が_なし_のときも本文を作れる():
    md = "# 議事録 x\n\n- 日時: 2026-09-20 19:30\n\n## 概要\n\n概要です\n\n## 決定事項\n\nなし\n\n## ToDo\n\nなし\n\n## 付録: 文字起こし\n"
    text = build_post_text(md, "x")
    assert "**決定事項**\nなし" in text and "**ToDo**\nなし" in text


# --- 投稿 ----------------------------------------------------------------------------------------


def test_議事録を本文とファイル1つで投稿し_投稿済みにする(con, cfg):
    cfg.discord.minutes_channel_id = "111"
    make_minutes(con, cfg)
    client = FakeDiscordClient()

    report = run_post(con, cfg, client, NOW)

    assert report.posted == [CID] and not report.failed
    channel, content, message_id = client.posted[0]
    assert channel == "111" and content == build_post_text(markdown(cfg), CID)
    filename, data = client.files[message_id]
    assert filename == f"{CID}.md" and data.decode("utf-8") == markdown(cfg)  # Markdown 全体を添付する
    row = repo.get_minutes(con, CID)
    assert row.discord_message_id == message_id and row.posted_at == NOW


def test_投稿済みは再投稿しない(con, cfg):
    cfg.discord.minutes_channel_id = "111"
    make_minutes(con, cfg)
    client = FakeDiscordClient()
    run_post(con, cfg, client, NOW)
    run_post(con, cfg, client, NOW)
    assert len(client.posted) == 1


def test_議事録を作り直しても自動では再投稿しない(con, cfg):
    cfg.discord.minutes_channel_id = "111"
    make_minutes(con, cfg)
    client = FakeDiscordClient()
    run_post(con, cfg, client, NOW)
    generate_minutes(con, cfg, CID, FakeJsonChat(handler), datetime(2026, 11, 7), lambda m: None)
    run_post(con, cfg, client, NOW)
    assert len(client.posted) == 1


def test_1件の投稿に失敗しても残りは投稿し_失敗した議事録は未投稿のまま残す(con, cfg):
    cfg.discord.minutes_channel_id = "111"
    make_minutes(con, cfg, "20260920_1930_f2f_teirei")
    make_minutes(con, cfg, "20260921_1930_f2f_teirei")
    client = FakeDiscordClient(fail_post_when=lambda c: "20260920_1930" in c)

    report = run_post(con, cfg, client, NOW)

    assert report.posted == ["20260921_1930_f2f_teirei"]
    assert len(report.failures) == 1 and "20260920_1930_f2f_teirei" in report.failures[0] and report.failed
    assert [m.conversation_id for m in repo.list_unposted_minutes(con)] == ["20260920_1930_f2f_teirei"]
    # 次回の実行で再試行される
    report2 = run_post(con, cfg, FakeDiscordClient(), NOW)
    assert report2.posted == ["20260920_1930_f2f_teirei"]


def test_議事録のファイルが無ければ失敗として残し_他は続ける(con, cfg):
    cfg.discord.minutes_channel_id = "111"
    make_minutes(con, cfg, "20260920_1930_f2f_teirei")
    make_minutes(con, cfg, "20260921_1930_f2f_teirei")
    (cfg.paths.minutes / "20260920_1930_f2f_teirei.md").unlink()
    report = run_post(con, cfg, FakeDiscordClient(), NOW)
    assert report.posted == ["20260921_1930_f2f_teirei"]
    assert "読めません" in report.failures[0]


def test_チャンネルIDが空なら何も投稿せず設定が必要と伝える(con, cfg):
    make_minutes(con, cfg)
    client = FakeDiscordClient()
    report = run_post(con, cfg, client, NOW)
    assert client.posted == [] and report.skipped_for_config and "minutes_channel_id" in report.messages[0]
    assert repo.get_minutes(con, CID).posted_at is None


# --- post_command ----------------------------------------------------------------------------------


def test_post_command_は設定が空なら_0_を返す_トークンも要らない(con, cfg, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    make_minutes(con, cfg)
    out: list[str] = []
    assert post_command(con, cfg, echo=out.append) == 0
    assert any("minutes_channel_id" in m for m in out)


def test_post_command_は未投稿が無ければトークンが無くても_0(con, cfg, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    cfg.discord.minutes_channel_id = "111"
    out: list[str] = []
    assert post_command(con, cfg, echo=out.append) == 0
    assert any("未投稿の議事録はありません" in m for m in out)


def test_post_command_は未投稿があってトークンが無ければ案内して失敗する(con, cfg, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    cfg.discord.minutes_channel_id = "111"
    make_minutes(con, cfg)
    with pytest.raises(RuntimeError, match=TOKEN_ENV):
        post_command(con, cfg, echo=lambda m: None)


def test_post_command_は投稿に失敗したものがあれば_1(con, cfg):
    cfg.discord.minutes_channel_id = "111"
    make_minutes(con, cfg)
    out: list[str] = []
    code = post_command(con, cfg, FakeDiscordClient(fail_post_when=lambda c: True), out.append, NOW)
    assert code == 1 and any(m.startswith("失敗:") for m in out)
    code_ok = post_command(con, cfg, FakeDiscordClient(), out.append, NOW)
    assert code_ok == 0 and any("投稿: 1 件" in m for m in out)


def test_DiscordError_がクライアントから漏れても_1_を返す(con, cfg):
    class Broken:
        def post_file(self, *a):
            raise DiscordError("ログインできません")

    cfg.discord.minutes_channel_id = "111"
    make_minutes(con, cfg)
    out: list[str] = []
    assert post_command(con, cfg, Broken(), out.append, NOW) == 1
    assert any("ログインできません" in m for m in out)


# --- CLI -----------------------------------------------------------------------------------------------


def test_discord_post_コマンドと_minutes_コマンド(tmp_path, monkeypatch):
    from kijun import cli
    from kijun.llm import chat as chat_module

    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text(
        f'[paths]\ndb = "{(tmp_path / "k.duckdb").as_posix()}"\nminutes = "{(tmp_path / "md").as_posix()}"\n',
        encoding="utf-8",
    )
    base = ["--config", str(cfg_path)]
    runner = CliRunner()
    assert runner.invoke(cli.app, [*base, "db", "init"]).exit_code == 0
    con = repo.connect(tmp_path / "k.duckdb")
    repo.insert_conversation(con, make_conversation(CID, transcribed_at=datetime(2026, 9, 21), duration_sec=600.0))
    repo.replace_utterances(con, CID, make_utterances(10))
    con.close()
    monkeypatch.setattr(chat_module, "OllamaJsonChat", lambda cfg: FakeJsonChat(handler))

    r = runner.invoke(cli.app, [*base, "minutes", "--pending"])
    assert r.exit_code == 0, r.output
    assert "議事録を保存しました" in r.output and (tmp_path / "md" / f"{CID}.md").exists()
    r = runner.invoke(cli.app, [*base, "minutes", "--pending"])
    assert "対象の会議がありません" in r.output  # 2回目は議事録のある会議を作り直さない
    r = runner.invoke(cli.app, [*base, "minutes", "--conversation", CID])
    assert r.exit_code == 0 and "議事録を保存しました" in r.output  # 指定すれば作り直せる

    monkeypatch.setattr(chat_module, "OllamaJsonChat", lambda cfg: FakeJsonChat(lambda *a: __import__("kijun.llm.chat", fromlist=["x"]).LlmError("接続拒否")))
    r = runner.invoke(cli.app, [*base, "minutes", "--conversation", CID])
    assert r.exit_code == 1 and "議事録を作れませんでした" in r.output

    r = runner.invoke(cli.app, [*base, "discord", "post"])  # チャンネルIDが空
    assert r.exit_code == 0 and "minutes_channel_id" in r.output
