from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kijun.cli import app
from kijun.db import repo
from kijun.ingest.run import collect_bundles, format_bundles, register_bundles
from tests.conftest import make_conversation

runner = CliRunner()


def touch(dir_: Path, *names: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dir_ / n).write_bytes(b"x")


def durations(table: dict[str, float | None]):
    return lambda path: table.get(Path(path).name)


# --- ingest ------------------------------------------------------------------------------


def test_inbox_を走査して束ねる(cfg):
    touch(cfg.paths.inbox, "20260920_1930_f2f_teirei.m4a", "20260920_2055_f2f_teirei.m4a", "memo.m4a")
    result = collect_bundles(
        cfg, warn=lambda m: None,
        duration_fn=durations({"20260920_1930_f2f_teirei.m4a": 3600, "20260920_2055_f2f_teirei.m4a": 1200}),
    )
    assert len(result.bundles) == 1
    assert result.bundles[0].file_names == ["20260920_1930_f2f_teirei.m4a", "20260920_2055_f2f_teirei.m4a"]
    assert [n for n, _ in result.failed] == ["memo.m4a"]
    assert (cfg.paths.inbox / "memo.m4a").exists()  # 解析できなくても動かさない


def test_長さが取れなかったら警告を出す(cfg):
    touch(cfg.paths.inbox, "20260920_1930_f2f_teirei.m4a")
    warnings: list[str] = []
    result = collect_bundles(cfg, warn=warnings.append, duration_fn=durations({}))
    assert any("長さ" in w for w in warnings)
    assert result.bundles[0].has_unknown_duration
    assert "長さ不明" in format_bundles(result.bundles)


def test_実在のファイルでも長さ取得に失敗したら_None_になり落ちない(cfg):
    # 中身が音声でないファイル。mutagen が読めず None が返る
    touch(cfg.paths.inbox, "20260920_1930_f2f_teirei.m4a")
    result = collect_bundles(cfg, warn=lambda m: None)
    assert result.bundles[0].total_duration_sec is None


def test_登録_更新_スキップ(con, cfg):
    touch(cfg.paths.inbox, "20260920_1930_f2f_teirei.m4a")
    first = collect_bundles(cfg, warn=lambda m: None, duration_fn=durations({"20260920_1930_f2f_teirei.m4a": 600}))
    register_bundles(con, first, now=datetime(2026, 9, 21))
    assert first.registered == ["20260920_1930_f2f_teirei"]
    conv = repo.get_conversation(con, "20260920_1930_f2f_teirei")
    assert conv.source_files == ["20260920_1930_f2f_teirei.m4a"] and conv.duration_sec == 600
    assert conv.setting_format == "f2f" and conv.setting_kind == "teirei"

    # 続きのファイルが届いた（文字起こし前）: ファイル一覧を更新する
    touch(cfg.paths.inbox, "20260920_1945_f2f_teirei.m4a")
    second = collect_bundles(
        cfg, warn=lambda m: None,
        duration_fn=durations({"20260920_1930_f2f_teirei.m4a": 600, "20260920_1945_f2f_teirei.m4a": 300}),
    )
    register_bundles(con, second)
    assert second.updated == ["20260920_1930_f2f_teirei"] and second.registered == []
    assert len(repo.get_conversation(con, "20260920_1930_f2f_teirei").source_files) == 2

    # 変化が無ければ何もしない
    third = collect_bundles(
        cfg, warn=lambda m: None,
        duration_fn=durations({"20260920_1930_f2f_teirei.m4a": 600, "20260920_1945_f2f_teirei.m4a": 300}),
    )
    register_bundles(con, third)
    assert third.updated == [] and third.registered == [] and third.skipped == []

    # 文字起こし済みの会議には触らない
    repo.mark_transcribed(con, "20260920_1930_f2f_teirei", "t.json", 900.0, datetime(2026, 9, 21))
    fourth = collect_bundles(cfg, warn=lambda m: None, duration_fn=durations({}))
    register_bundles(con, fourth)
    assert fourth.skipped == ["20260920_1930_f2f_teirei"]


# --- CLI ----------------------------------------------------------------------------------


def write_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f"""
[paths]
inbox = "{(tmp_path / 'inbox').as_posix()}"
processed = "{(tmp_path / 'processed').as_posix()}"
transcripts = "{(tmp_path / 'transcripts').as_posix()}"
db = "{(tmp_path / 'data' / 'kijun.duckdb').as_posix()}"
""",
        encoding="utf-8",
    )
    return cfg_path


def test_kijun_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("db", "ingest", "transcribe", "restore-transcript", "extract", "compare", "match", "purge-audio"):
        assert cmd in result.output


@pytest.mark.parametrize(
    "args",
    [["db"], ["db", "init"], ["ingest"], ["transcribe"], ["restore-transcript"], ["extract"], ["compare"], ["match"], ["purge-audio"]],
)
def test_各サブコマンドの_help(args):
    result = runner.invoke(app, [*args, "--help"])
    assert result.exit_code == 0, result.output


def test_db_init_と_ingest_dry_run(tmp_path):
    cfg_path = write_config(tmp_path)
    touch(tmp_path / "inbox", "20260920_1930_f2f_teirei.m4a", "bad name.m4a")
    base = ["--config", str(cfg_path)]

    r = runner.invoke(app, [*base, "db", "init"])
    assert r.exit_code == 0 and "テーブル 10 個" in r.output
    assert (tmp_path / "data" / "kijun.duckdb").exists()

    r = runner.invoke(app, [*base, "ingest", "--dry-run"])
    assert r.exit_code == 0
    assert "20260920_1930_f2f_teirei" in r.output
    assert "bad name.m4a" in r.output  # 解析できなかったファイルが一覧に出る
    assert "DB には登録していません" in r.output
    assert (tmp_path / "inbox" / "bad name.m4a").exists()

    r = runner.invoke(app, [*base, "ingest"])
    assert r.exit_code == 0 and "新規登録 1 件" in r.output
    r = runner.invoke(app, [*base, "ingest"])
    assert "新規登録 0 件" in r.output  # 2回目は重複登録しない


def test_スキーマ未適用なら_db_init_を促す(tmp_path):
    cfg_path = write_config(tmp_path)
    r = runner.invoke(app, ["--config", str(cfg_path), "extract", "--conversation", "x"])
    assert r.exit_code == 1 and "db init" in r.output


def test_重い依存が無いときは入れ方を示して終了する(tmp_path):
    cfg_path = write_config(tmp_path)
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    r = runner.invoke(app, ["--config", str(cfg_path), "transcribe", "--audio", str(audio)])
    assert r.exit_code == 1
    assert "uv sync --extra transcribe" in r.output


def test_transcribe_は_conversation_と_audio_のどちらか一方を要求する(tmp_path):
    cfg_path = write_config(tmp_path)
    r = runner.invoke(app, ["--config", str(cfg_path), "transcribe"])
    assert r.exit_code == 1 and "どちらか一方" in r.output
    r = runner.invoke(app, ["--config", str(cfg_path), "transcribe", "--conversation", "a", "--audio", "b"])
    assert r.exit_code == 1


def test_purge_audio_は既定では削除せず_execute_で削除する(tmp_path):
    cfg_path = write_config(tmp_path)
    base = ["--config", str(cfg_path)]
    runner.invoke(app, [*base, "db", "init"])
    from kijun.config import load_config

    cfg = load_config(cfg_path)
    touch(cfg.paths.processed, "old.m4a")
    con = repo.connect(cfg.paths.db)
    repo.insert_conversation(
        con, make_conversation("old", ["old.m4a"], transcribed_at=datetime.now() - timedelta(days=40))
    )
    con.close()

    r = runner.invoke(app, [*base, "purge-audio"])
    assert r.exit_code == 0 and "old.m4a" in r.output and "--execute" in r.output
    assert (cfg.paths.processed / "old.m4a").exists()  # 既定では削除しない

    r = runner.invoke(app, [*base, "purge-audio", "--execute"])
    assert r.exit_code == 0 and "削除しました" in r.output
    assert not (cfg.paths.processed / "old.m4a").exists()
