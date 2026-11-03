"""discord read / post / derive-direction のコマンド(仕様 2-1、2-8)。"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from kijun.cli import app
from kijun.db import migrate, repo
from kijun.discord import run as discord_run
from kijun.review.reply import RoutedEntry
from tests.fakes import FakeDiscordClient, FakeReplyRouter
from tests.review_helpers import CID, NOW, add_request, candidate, r, reply, setup_conversation, APPROVE

runner = CliRunner()


def write_config(tmp_path: Path, channels: bool = True) -> Path:
    path = tmp_path / "config.toml"
    discord = '[discord]\nreview_channel_id = "111"\ncontradictions_channel_id = "222"\n' if channels else ""
    path.write_text(
        f"""
[paths]
db = "{(tmp_path / 'data' / 'kijun.duckdb').as_posix()}"
{discord}
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def db(tmp_path):
    cfg_path = write_config(tmp_path)
    runner.invoke(app, ["--config", str(cfg_path), "db", "init"])
    con = repo.connect(tmp_path / "data" / "kijun.duckdb")
    yield con, cfg_path
    con.close()


def invoke(cfg_path, *args):
    return runner.invoke(app, ["--config", str(cfg_path), *args])


def test_help_に_discord_と_derive_direction_が出る():
    out = runner.invoke(app, ["--help"]).output
    assert "discord" in out and "derive-direction" in out
    assert "read" in runner.invoke(app, ["discord", "--help"]).output
    assert "post" in runner.invoke(app, ["discord", "--help"]).output
    assert "--all" in runner.invoke(app, ["derive-direction", "--help"]).output


def test_DB_が未適用なら_db_init_を促す(tmp_path):
    cfg_path = write_config(tmp_path)
    r_ = invoke(cfg_path, "discord", "post")
    assert r_.exit_code == 1 and "db init" in r_.output


def test_チャンネルIDが空なら何もせず設定が必要だと出して正常終了する(tmp_path, monkeypatch):
    cfg_path = write_config(tmp_path, channels=False)
    invoke(cfg_path, "db", "init")
    monkeypatch.setattr(discord_run, "build_client", lambda cfg: pytest.fail("接続してはいけない"))
    for sub in ("read", "post"):
        result = invoke(cfg_path, "discord", sub)
        assert result.exit_code == 0 and "設定が必要" in result.output


def test_トークンが無ければ環境変数の設定方法を示して終了する(db, monkeypatch):
    _con, cfg_path = db
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    result = invoke(cfg_path, "discord", "post")
    assert result.exit_code == 1 and "DISCORD_BOT_TOKEN" in result.output


def test_discord_py_が無ければ入れ方を示す(db, monkeypatch):
    import sys

    _con, cfg_path = db
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "x")
    monkeypatch.setitem(sys.modules, "discord", None)
    result = invoke(cfg_path, "discord", "post")
    assert result.exit_code == 1 and "uv sync --extra discord" in result.output


def test_post_は投稿して件数と未投稿の件数を出す(db, monkeypatch):
    con, cfg_path = db
    candidate(con, 1, importance=2)
    candidate(con, 2, importance=3)
    con.close()
    client = FakeDiscordClient()
    monkeypatch.setattr(discord_run, "build_client", lambda cfg: client)
    monkeypatch.setattr(discord_run, "build_llm", lambda cfg, transport=None: discord_run.LlmParts())

    result = invoke(cfg_path, "discord", "post")

    assert result.exit_code == 0, result.output
    assert "new_candidate 2 件" in result.output and "まだ投稿していない新規候補: 0 件" in result.output
    assert len([c for c in client.contents() if c.startswith("【新規候補")]) == 2


def test_post_は投稿に失敗したものがあれば終了コード_1(db, monkeypatch):
    con, cfg_path = db
    candidate(con, 1, importance=2)
    con.close()
    monkeypatch.setattr(discord_run, "build_client", lambda cfg: FakeDiscordClient(fail_post_when=lambda c: True))
    monkeypatch.setattr(discord_run, "build_llm", lambda cfg, transport=None: discord_run.LlmParts())
    result = invoke(cfg_path, "discord", "post")
    assert result.exit_code == 1 and "失敗:" in result.output


def test_read_はリアクションを反映して件数を出す(db, monkeypatch):
    con, cfg_path = db
    candidate(con, 1, importance=2)
    repo.insert_discord_post(con, repo.DiscordPostRow("m1", "new_candidate", "item_0001", CID, NOW, None, "111", None))
    con.close()
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE)
    client.replies["m1"] = [reply("理由はこれ")]
    router = FakeReplyRouter({"理由はこれ": [RoutedEntry(column="reason", text="これ")]})
    monkeypatch.setattr(discord_run, "build_client", lambda cfg: client)
    monkeypatch.setattr(discord_run, "build_llm", lambda cfg, transport=None: discord_run.LlmParts(router=router))

    result = invoke(cfg_path, "discord", "read")

    assert result.exit_code == 0, result.output
    assert "status の更新 1 件" in result.output and "返信の振り分け 1 件" in result.output
    con = repo.connect(Path(cfg_path).parent / "data" / "kijun.duckdb")
    assert repo.get_knowledge_item(con, "item_0001").status == "approved"
    con.close()


def test_read_は判定できないリアクションを会議IDと項目IDつきで出す(db, monkeypatch):
    con, cfg_path = db
    candidate(con, 1, importance=2)
    repo.insert_discord_post(con, repo.DiscordPostRow("m1", "new_candidate", "item_0001", CID, NOW, None, "111", None))
    con.close()
    client = FakeDiscordClient()
    client.reactions["m1"] = r(APPROVE, "❌")
    monkeypatch.setattr(discord_run, "build_client", lambda cfg: client)
    monkeypatch.setattr(discord_run, "build_llm", lambda cfg, transport=None: discord_run.LlmParts())
    result = invoke(cfg_path, "discord", "read")
    assert result.exit_code == 0
    assert "判定できませんでした" in result.output and CID in result.output and "item_0001" in result.output


# --- derive-direction ----------------------------------------------------------------------------


def test_derive_direction_は会議を指定して導出する(db):
    con, cfg_path = db
    setup_conversation(con)
    add_request(con, "r1")
    repo.set_speaker_map(con, CID, "SPEAKER_00", "self")
    con.close()
    result = invoke(cfg_path, "derive-direction", "--conversation", CID)
    assert result.exit_code == 0, result.output
    assert "導出した" in result.output
    con = repo.connect(Path(cfg_path).parent / "data" / "kijun.duckdb")
    assert repo.get_requests(con, CID)[0].direction == "self_to_other"
    con.close()


def test_derive_direction_の_all_は_speaker_map_のある会議すべて(db):
    con, cfg_path = db
    for cid in ("c1", "c2"):
        setup_conversation(con, cid)
        add_request(con, f"{cid}#1", conversation_id=cid)
    repo.set_speaker_map(con, "c1", "SPEAKER_00", "self")  # c2 には self が無い
    con.close()
    result = invoke(cfg_path, "derive-direction", "--all")
    assert result.exit_code == 0 and "c1" in result.output and "c2" not in result.output


def test_derive_direction_は_self_が無い会議で導出しないと出す(db):
    con, cfg_path = db
    setup_conversation(con)
    add_request(con, "r1")
    con.close()
    result = invoke(cfg_path, "derive-direction", "--conversation", CID)
    assert "導出しない" in result.output and "self" in result.output


def test_derive_direction_は引数をどちらか一方に限る(db):
    _con, cfg_path = db
    assert invoke(cfg_path, "derive-direction").exit_code == 1
    assert invoke(cfg_path, "derive-direction", "--all", "--conversation", "x").exit_code == 1
