"""夜間バッチの手順(仕様 3-1、3-2、テスト 9)。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from kijun.batch.run import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, format_summary, plan_lines, run_batch, write_log
from kijun.db import repo
from tests.conftest import make_conversation, make_utterances
from tests.fakes import FakeChatTransport, FakeCommandRunner
from tests.review_helpers import add_request, candidate

CLOCK = lambda: datetime(2026, 11, 4, 23, 30)  # noqa: E731


class Clock:
    """sleep で進む偽の時計。待ち時間のテストで本当に待たない。"""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def sleep(self, sec: float) -> None:
        self.sleeps.append(sec)
        self.t += sec

    def monotonic(self) -> float:
        return self.t


def run(cfg, runner, transport=None, clock=None):
    clock = clock or Clock()
    return run_batch(
        cfg, None, runner, transport, clock=CLOCK, sleep=clock.sleep, monotonic=clock.monotonic, echo=lambda m: None
    )


def steps(result):
    return [(s.step, s.status) for s in result.steps]


def add_untranscribed(con, *ids):
    for i in ids:
        repo.insert_conversation(con, make_conversation(i))


def mark_transcribed(con, cid):
    repo.mark_transcribed(con, cid, "t.json", 60.0, datetime(2026, 11, 4, 23, 40))
    repo.replace_utterances(con, cid, make_utterances(3))


class World:
    """子プロセスの副作用を DB に再現する偽の世界。transcribe が成功した会議は文字起こし済みになる。"""

    def __init__(self, cfg, fail: set[str] = frozenset(), fail_exact: set[str] = frozenset()) -> None:
        self.cfg, self.fail, self.fail_exact = cfg, set(fail), set(fail_exact)
        self.runner = FakeCommandRunner(self.handle)

    def handle(self, command):
        if isinstance(command, str):
            return None
        args = command[3:]
        if args[:1] == ["--config"]:
            args = args[2:]
        text = " ".join(args)
        if text in self.fail_exact or any(text.startswith(f) for f in self.fail):
            return (1, f"エラー: {text} が失敗")
        if args[:1] == ["transcribe"]:
            con = repo.connect(self.cfg.paths.db)
            mark_transcribed(con, args[2])
            con.close()
        return None


# --- 順序 -------------------------------------------------------------------------------------


def test_仕様の順序でコマンドを実行する(con, cfg):
    add_untranscribed(con, "20260920_1930_f2f_teirei")
    con.close()
    world = World(cfg)

    result = run(cfg, world.runner)

    assert world.runner.kijun_calls() == [
        "discord read",
        "ingest",
        "transcribe --conversation 20260920_1930_f2f_teirei",
        "extract --conversation 20260920_1930_f2f_teirei",
        "discord post",  # 抽出の副作用は偽なので match の対象は無い
    ]
    assert result.ok


def test_dry_run_は仕様の表の順序でコマンドを出す(cfg):
    lines = plan_lines(cfg, None)
    commands = [l.split("  ", 1)[1].strip() for l in lines if l.startswith("手順")]
    assert [l.split()[0] for l in lines if l.startswith("手順")] == [
        "手順1", "手順2", "手順3-4", "手順5", "手順6", "手順7", "手順8", "手順9",
    ]
    assert commands[0] == "uv run kijun discord read"
    assert commands[1] == "uv run kijun ingest"
    assert commands[2] == "uv run kijun transcribe --conversation <ID>"
    assert commands[5] == "uv run kijun match --conversation <ID>"
    assert commands[6] == "uv run kijun discord post"
    text = "\n".join(lines)
    assert "uv run kijun extract --conversation <ID>" in text
    assert text.index("transcribe") < text.index("extract") < text.index("match") < text.index("discord post")


def test_dry_run_は設定した_start_stop_を表示し_config_を引き継ぐ(cfg, tmp_path):
    cfg.extract.strata.start_command = "start-strata.bat"
    cfg.extract.strata.stop_command = "stop-strata.bat"
    text = "\n".join(plan_lines(cfg, tmp_path / "my.toml"))
    assert "start-strata.bat" in text and "stop-strata.bat" in text
    assert f"uv run kijun --config {tmp_path / 'my.toml'} ingest" in text


def test_dry_run_は何も実行せず_DB_にも触れない(cfg, tmp_path):
    from typer.testing import CliRunner

    from kijun.cli import app

    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text(f'[paths]\ndb = "{(tmp_path / "none" / "x.duckdb").as_posix()}"\n', encoding="utf-8")
    out = CliRunner().invoke(app, ["--config", str(cfg_path), "batch", "run", "--dry-run"])
    assert out.exit_code == 0 and "何も実行していません" in out.output
    assert not (tmp_path / "none").exists()


# --- 失敗の扱い --------------------------------------------------------------------------------


def test_手順1が失敗しても続ける(con, cfg):
    con.close()
    world = World(cfg, fail_exact={"discord read"})
    result = run(cfg, world.runner)
    assert world.runner.kijun_calls()[:2] == ["discord read", "ingest"]
    assert not result.ok and ("1", STATUS_FAILED) in steps(result)


def test_手順2が失敗したら止める(con, cfg):
    add_untranscribed(con, "20260920_1930_f2f_teirei")
    con.close()
    world = World(cfg, fail_exact={"ingest"})

    result = run(cfg, world.runner)

    assert world.runner.kijun_calls() == ["discord read", "ingest"]  # 以降は何も起動しない
    assert not result.ok
    assert steps(result) == [
        ("1", STATUS_OK), ("2", STATUS_FAILED), ("3-4", STATUS_SKIPPED), ("6", STATUS_SKIPPED),
        ("7", STATUS_SKIPPED), ("8", STATUS_SKIPPED),
    ]
    assert "手順2" in result.steps[2].note


def test_手順3_4が1つの会議で失敗しても次の会議に進む(con, cfg):
    add_untranscribed(con, "20260920_1930_f2f_teirei", "20260921_1930_f2f_teirei", "20260922_1930_f2f_teirei")
    con.close()
    world = World(cfg, fail={"transcribe --conversation 20260921"})

    result = run(cfg, world.runner)

    calls = world.runner.kijun_calls()
    assert [c for c in calls if c.startswith("transcribe")] == [
        "transcribe --conversation 20260920_1930_f2f_teirei",
        "transcribe --conversation 20260921_1930_f2f_teirei",
        "transcribe --conversation 20260922_1930_f2f_teirei",  # 失敗した会議の次も実行する
    ]
    # 失敗した会議は文字起こし済みにならないので、抽出の対象にならない
    assert [c for c in calls if c.startswith("extract")] == [
        "extract --conversation 20260920_1930_f2f_teirei",
        "extract --conversation 20260922_1930_f2f_teirei",
    ]
    assert calls[-1] == "discord post"  # 最後まで進む
    failed = [s for s in result.steps if s.status == STATUS_FAILED]
    assert [s.name for s in failed] == ["文字起こし 20260921_1930_f2f_teirei"] and not result.ok


def test_抽出が1つの会議で失敗しても次の会議に進む(con, cfg):
    add_untranscribed(con, "c1", "c2")
    con.close()
    world = World(cfg, fail={"extract --conversation c1"})
    result = run(cfg, world.runner)
    assert [c for c in world.runner.kijun_calls() if c.startswith("extract")] == [
        "extract --conversation c1", "extract --conversation c2",
    ]
    assert not result.ok


def test_抽出が済んでいない会議は翌晩に再び対象になる(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg, fail={"extract"})
    run(cfg, world.runner)
    second = World(cfg)
    run(cfg, second.runner)
    assert "extract --conversation c1" in second.runner.kijun_calls()


def test_抽出が0件で済んだ会議は翌晩に抽出し直さない(con, cfg):
    from kijun.extract.run import ExtractionRun, extract_log_path, write_extract_log

    add_untranscribed(con, "c1")
    mark_transcribed(con, "c1")
    write_extract_log(
        extract_log_path(cfg, "c1", cfg.extract.strata.model), "c1",
        ExtractionRun(model_name=cfg.extract.strata.model, chunk_count=2), datetime(2026, 11, 3),
    )
    con.close()
    world = World(cfg)
    run(cfg, world.runner)
    assert not any(c.startswith("extract") for c in world.runner.kijun_calls())


def test_すべてのチャンクが失敗した抽出は翌晩に再試行する(con, cfg):
    from kijun.extract.base import ExtractionError  # noqa: F401
    from kijun.extract.run import ChunkFailure, ExtractionRun, extract_log_path, write_extract_log

    add_untranscribed(con, "c1")
    mark_transcribed(con, "c1")
    model = cfg.extract.strata.model
    failed_run = ExtractionRun(model_name=model, chunk_count=2, failures=[ChunkFailure(0, 0, 1, "x"), ChunkFailure(1, 1, 2, "y")])
    write_extract_log(extract_log_path(cfg, "c1", model), "c1", failed_run, datetime(2026, 11, 3))
    con.close()
    world = World(cfg)
    run(cfg, world.runner)
    assert "extract --conversation c1" in world.runner.kijun_calls()


def test_照合は未紐付けの抽出結果がある会議だけ実行する(con, cfg):
    candidate(con, 1)  # c: 抽出結果は紐付け済み
    from tests.review_helpers import CID

    add_request(con, "u1", conversation_id=CID, model=cfg.extract.strata.model)  # 紐付いていない
    mark_transcribed(con, CID)
    con.close()
    world = World(cfg)
    run(cfg, world.runner)
    assert f"match --conversation {CID}" in world.runner.kijun_calls()


def test_対象が無ければ_その手順は飛ばす(con, cfg):
    con.close()
    world = World(cfg)
    result = run(cfg, world.runner)
    assert world.runner.kijun_calls() == ["discord read", "ingest", "discord post"]
    assert [(s.step, s.status) for s in result.steps if s.status == STATUS_SKIPPED] == [
        ("3-4", STATUS_SKIPPED), ("6", STATUS_SKIPPED), ("7", STATUS_SKIPPED),
    ]
    assert result.ok


def test_DB_がまだ無くても落ちずに手順2を実行する(cfg):
    world = World(cfg)
    result = run(cfg, world.runner)
    assert world.runner.kijun_calls()[:2] == ["discord read", "ingest"] and result.ok


# --- 手順6: Strata の起動と停止 ---------------------------------------------------------------------


def with_strata(cfg):
    cfg.extract.strata.start_command = "start-strata"
    cfg.extract.strata.stop_command = "stop-strata"
    cfg.batch.strata_ready_timeout_sec = 100
    return cfg


def test_手順6は_start_抽出_stop_の順に挟む(con, cfg):
    with_strata(cfg)
    add_untranscribed(con, "c1", "c2")
    con.close()
    world = World(cfg)
    transport = FakeChatTransport(["x"])

    run(cfg, world.runner, transport)

    commands = world.runner.commands()
    start, stop = commands.index("start-strata"), commands.index("stop-strata")
    assert commands[start + 1 : stop] == ["kijun extract --conversation c1", "kijun extract --conversation c2"]
    assert commands.index("kijun transcribe --conversation c2") < start  # 文字起こしが先
    assert transport.gets == ["http://localhost:8080/v1/models"]  # 応答を確認してから抽出する
    assert commands.count("start-strata") == 1  # 会議ごとには起動しない


def test_手順6が失敗しても_stop_command_が実行される(con, cfg):
    with_strata(cfg)
    add_untranscribed(con, "c1", "c2")
    con.close()
    world = World(cfg, fail={"extract"})

    result = run(cfg, world.runner, FakeChatTransport(["x"]))

    commands = world.runner.commands()
    assert "stop-strata" in commands
    assert commands.index("stop-strata") > commands.index("kijun extract --conversation c2")  # 全部試してから停止
    assert any(s.step == "6:stop" and s.status == STATUS_OK for s in result.steps)


def test_start_command_が失敗したら抽出は飛ばすが_stop_command_は実行する(con, cfg):
    with_strata(cfg)
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg)
    world.runner = FakeCommandRunner(lambda c: (1, "起動できません") if c == "start-strata" else world.handle(c))

    result = run(cfg, world.runner, FakeChatTransport(["x"]))

    commands = world.runner.commands()
    assert "kijun extract --conversation c1" not in commands
    assert "stop-strata" in commands  # 起動が中途半端に残らないよう必ず停止する
    assert any(s.step == "6" and s.status == STATUS_SKIPPED and "起動できなかった" in s.note for s in result.steps)
    assert not result.ok


def test_Strata_が応答しなければ待ってから失敗にして_stop_command_を実行する(con, cfg):
    with_strata(cfg)
    cfg.batch.strata_ready_timeout_sec = 12
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg)
    transport = FakeChatTransport(["x"])
    transport.get_error = Exception  # 使われない。下で上書き
    from kijun.transport import TransportError

    transport.get_error = TransportError("接続拒否")
    clock = Clock()

    result = run(cfg, world.runner, transport, clock)

    assert "kijun extract --conversation c1" not in world.runner.commands()
    assert "stop-strata" in world.runner.commands()
    assert clock.t >= 12 and len(transport.gets) >= 3  # 待ちながら何度も確認した
    ready = next(s for s in result.steps if s.step == "6:ready")
    assert ready.status == STATUS_FAILED and "応答しませんでした" in ready.output_tail


def test_Strata_の起動に時間がかかっても_応答したら抽出に進む(con, cfg):
    with_strata(cfg)
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg)
    transport = FakeChatTransport(["x"])
    transport.get_ready_after = 3  # 3回目までは応答しない
    run(cfg, world.runner, transport, Clock())
    assert "kijun extract --conversation c1" in world.runner.commands()
    assert len(transport.gets) == 4


def test_start_command_の実行には待ち時間の上限を渡す(con, cfg):
    with_strata(cfg)
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg)
    run(cfg, world.runner, FakeChatTransport(["x"]))
    index = world.runner.calls.index("start-strata")
    assert world.runner.timeouts[index] == 100  # 戻らない start_command を打ち切るため


def test_start_command_が空なら起動も停止もしない(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg)
    run(cfg, world.runner)
    assert all(isinstance(c, list) for c in world.runner.calls)  # シェルのコマンドは1つも無い


def test_手順1は未処理の投稿があるときだけ_Strata_を挟む(con, cfg):
    with_strata(cfg)
    cfg.discord.review_channel_id = "111"
    con.close()
    world = World(cfg)
    run(cfg, world.runner, FakeChatTransport(["x"]))
    assert "start-strata" not in world.runner.commands()  # 未処理の投稿が無いので LLM は要らない

    con = repo.connect(cfg.paths.db)
    candidate(con, 1)
    repo.insert_discord_post(con, repo.DiscordPostRow("m1", "new_candidate", "item_0001", None, datetime(2026, 11, 3), None, "111", None))
    con.close()
    world2 = World(cfg)
    run(cfg, world2.runner, FakeChatTransport(["x"]))
    commands = world2.runner.commands()
    assert commands[:4] == ["start-strata", "kijun discord read", "stop-strata", "kijun ingest"]


def test_手順8は重要度が未判定の候補があるときだけ_Strata_を挟む(con, cfg):
    with_strata(cfg)
    cfg.discord.review_channel_id = "111"
    candidate(con, 1, importance=3)  # 判定済み
    con.close()
    world = World(cfg)
    run(cfg, world.runner, FakeChatTransport(["x"]))
    assert world.runner.commands()[-1] == "kijun discord post"  # 投稿の前後に Strata を挟まない

    con = repo.connect(cfg.paths.db)
    candidate(con, 2, importance=None)
    con.close()
    world2 = World(cfg)
    run(cfg, world2.runner, FakeChatTransport(["x"]))
    commands = world2.runner.commands()
    assert commands[-3:] == ["start-strata", "kijun discord post", "stop-strata"]


def test_手順8は_Strata_が起動できなくても投稿は実行する(con, cfg):
    with_strata(cfg)
    cfg.discord.review_channel_id = "111"
    candidate(con, 1, importance=None)
    con.close()
    world = World(cfg)
    world.runner = FakeCommandRunner(lambda c: (1, "x") if c == "start-strata" else world.handle(c))
    result = run(cfg, world.runner, FakeChatTransport(["x"]))
    assert "kijun discord post" in world.runner.commands()  # LLM が無くても投稿できる
    assert "stop-strata" in world.runner.commands() and not result.ok


# --- 記録 ---------------------------------------------------------------------------------------------


def test_全手順の記録が_JSON_で保存される(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg, fail_exact={"discord read"})
    result = run(cfg, world.runner)

    path = write_log(cfg, result)

    assert path.parent == cfg.batch.log_dir and path.name == "20261104_233000.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ok"] is False and data["started_at"] == "2026-11-04T23:30:00"
    first = data["steps"][0]
    assert (first["step"], first["status"], first["exit_code"]) == ("1", "failed", 1)
    assert "discord read" in first["command"] and "エラー: discord read が失敗" in first["output_tail"]
    assert {"started_at", "ended_at", "seconds"} <= set(first)


def test_最後の表に全手順の成否が出る(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    result = run(cfg, World(cfg, fail_exact={"discord read"}).runner)
    table = format_summary(result)
    assert "失敗" in table and "成功" in table and "文字起こし c1" in table and "終了コード" in table


def test_中断は握りつぶさず_そのまま伝える(con, cfg):
    con.close()

    class Boom(FakeCommandRunner):
        def run(self, command, timeout_sec=None):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(cfg, Boom())


def test_子プロセスは現在の_Python_で_kijun_cli_を起動する(con, cfg, tmp_path):
    import sys

    con.close()
    world = World(cfg)
    run_batch(cfg, tmp_path / "my.toml", world.runner, None, clock=CLOCK, echo=lambda m: None)
    first = world.runner.calls[0]
    assert first[:3] == [sys.executable, "-m", "kijun.cli"] and first[3:5] == ["--config", str(tmp_path / "my.toml")]


# --- 実際のプロセスの起動(クラウド環境でも動く。python 自身を起動する) -----------------------------------------


def test_SubprocessRunner_は終了コードと日本語の出力の末尾を返す():
    import sys

    from kijun.batch.runner import OUTPUT_TAIL_CHARS, SubprocessRunner

    ok = SubprocessRunner().run([sys.executable, "-c", "import sys; print('日本語の出力'); print('エラー', file=sys.stderr)"])
    assert ok.exit_code == 0 and "日本語の出力" in ok.output_tail and "エラー" in ok.output_tail  # 標準エラーも残る

    ng = SubprocessRunner().run([sys.executable, "-c", "import sys; print('x' * 5000); sys.exit(3)"])
    assert ng.exit_code == 3 and len(ng.output_tail) == OUTPUT_TAIL_CHARS


def test_SubprocessRunner_はシェルの文字列も実行し_時間切れを打ち切る():
    import sys

    from kijun.batch.runner import SubprocessRunner

    assert SubprocessRunner().run("echo hello").output_tail.strip() == "hello"
    slow = SubprocessRunner().run([sys.executable, "-c", "import time; time.sleep(30)"], timeout_sec=0.5)
    assert slow.timed_out and slow.exit_code is None


def test_SubprocessRunner_は起動できないコマンドを失敗として返す():
    from kijun.batch.runner import SubprocessRunner

    result = SubprocessRunner().run(["/nonexistent/command"])
    assert result.exit_code is None and "起動できませんでした" in result.output_tail


def test_batch_run_コマンドは記録を保存し_失敗があれば終了コード_1(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from kijun import cli
    from kijun.batch import runner as runner_module
    from kijun.cli import app

    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text(
        f'[paths]\ndb = "{(tmp_path / "d" / "k.duckdb").as_posix()}"\n[batch]\nlog_dir = "{(tmp_path / "logs").as_posix()}"\n',
        encoding="utf-8",
    )
    fake = FakeCommandRunner(lambda c: 1 if isinstance(c, list) and c[-1] == "ingest" else None)
    monkeypatch.setattr(runner_module, "SubprocessRunner", lambda: fake)

    result = CliRunner().invoke(app, ["--config", str(cfg_path), "batch", "run"])

    assert result.exit_code == 1, result.output
    assert "終了コード" in result.output and "記録を保存しました" in result.output
    logs = list((tmp_path / "logs").glob("*.json"))
    assert len(logs) == 1 and json.loads(logs[0].read_text(encoding="utf-8"))["ok"] is False
    assert cli  # import の確認
