"""夜間バッチの手順(ingest -> transcribe -> minutes -> discord post)。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from kijun.batch.run import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, format_summary, plan_lines, run_batch, write_log
from kijun.db import repo
from tests.conftest import make_conversation, make_utterances
from tests.fakes import FakeCommandRunner, FakeProcessLister

CLOCK = lambda: datetime(2026, 11, 4, 23, 30)  # noqa: E731


class Clock:
    """sleep で進む偽の時計。待ち時間のテストで本当に待たない。"""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def sleep(self, sec: float) -> None:
        self.sleeps.append(sec)
        self.t += sec


def run(cfg, runner, lister=None, clock=None):
    clock = clock or Clock()
    return run_batch(cfg, None, runner, lister, clock=CLOCK, sleep=clock.sleep, echo=lambda m: None)


def steps(result):
    return [(s.step, s.status) for s in result.steps]


def add_untranscribed(con, *ids):
    for i in ids:
        repo.insert_conversation(con, make_conversation(i))


def mark_transcribed(con, cid):
    repo.mark_transcribed(con, cid, "t.json", 60.0, datetime(2026, 11, 4, 23, 40))
    repo.replace_utterances(con, cid, make_utterances(3))


class World:
    """子プロセスの副作用を DB に再現する偽の世界。transcribe が成功した会議は文字起こし済みに、
    minutes が成功した会議は議事録ありになる。"""

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
        if args[:1] == ["minutes"]:
            con = repo.connect(self.cfg.paths.db)
            repo.upsert_minutes(con, args[2], f"{args[2]}.md", "m", datetime(2026, 11, 4, 23, 50))
            con.close()
        return None


# --- 順序 -------------------------------------------------------------------------------------


def test_順序どおりにコマンドを実行する(con, cfg):
    add_untranscribed(con, "20260920_1930_f2f_teirei")
    con.close()
    world = World(cfg)

    result = run(cfg, world.runner)

    assert world.runner.kijun_calls() == [
        "ingest",
        "transcribe --conversation 20260920_1930_f2f_teirei",
        "minutes --conversation 20260920_1930_f2f_teirei",
        "discord post",
    ]
    assert result.ok


def test_dry_run_は順序どおりにコマンドを出す(cfg):
    lines = plan_lines(cfg, None)
    assert [l.split()[0] for l in lines if l.startswith("手順")] == ["手順1", "手順2", "手順3", "手順4"]
    commands = [l.split("  ", 1)[1].strip() for l in lines if l.startswith("手順")]
    assert commands == [
        "uv run kijun ingest",
        "uv run kijun transcribe --conversation <ID>",
        "uv run kijun minutes --conversation <ID>",
        "uv run kijun discord post",
    ]


def test_dry_run_は_config_を引き継ぐ(cfg, tmp_path):
    text = "\n".join(plan_lines(cfg, tmp_path / "my.toml"))
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


def test_手順1が失敗したら止める(con, cfg):
    add_untranscribed(con, "20260920_1930_f2f_teirei")
    con.close()
    world = World(cfg, fail_exact={"ingest"})

    result = run(cfg, world.runner)

    assert world.runner.kijun_calls() == ["ingest"]  # 以降は何も起動しない
    assert not result.ok
    assert steps(result) == [("1", STATUS_FAILED), ("2", STATUS_SKIPPED), ("3", STATUS_SKIPPED), ("4", STATUS_SKIPPED)]
    assert "手順1" in result.steps[1].note


def test_文字起こしが1つの会議で失敗しても次の会議に進む(con, cfg):
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
    # 失敗した会議は文字起こし済みにならないので、議事録の対象にならない
    assert [c for c in calls if c.startswith("minutes")] == [
        "minutes --conversation 20260920_1930_f2f_teirei",
        "minutes --conversation 20260922_1930_f2f_teirei",
    ]
    assert calls[-1] == "discord post"  # 最後まで進む
    failed = [s for s in result.steps if s.status == STATUS_FAILED]
    assert [s.name for s in failed] == ["文字起こし 20260921_1930_f2f_teirei"] and not result.ok


def test_議事録が1つの会議で失敗しても次の会議に進み_Discord_にも投稿する(con, cfg):
    add_untranscribed(con, "c1", "c2")
    con.close()
    world = World(cfg, fail={"minutes --conversation c1"})
    result = run(cfg, world.runner)
    calls = world.runner.kijun_calls()
    assert [c for c in calls if c.startswith("minutes")] == ["minutes --conversation c1", "minutes --conversation c2"]
    assert calls[-1] == "discord post"
    assert not result.ok


def test_議事録を作れなかった会議は翌晩に再び対象になる(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    run(cfg, World(cfg, fail={"minutes"}).runner)
    second = World(cfg)
    run(cfg, second.runner)
    assert "minutes --conversation c1" in second.runner.kijun_calls()


def test_議事録のある会議は翌晩に作り直さない(con, cfg):
    add_untranscribed(con, "c1")
    mark_transcribed(con, "c1")
    repo.upsert_minutes(con, "c1", "c1.md", "m", datetime(2026, 11, 3))
    con.close()
    world = World(cfg)
    run(cfg, world.runner)
    assert not any(c.startswith("minutes") for c in world.runner.kijun_calls())


def test_対象が無ければ_その手順は飛ばす(con, cfg):
    con.close()
    world = World(cfg)
    result = run(cfg, world.runner)
    assert world.runner.kijun_calls() == ["ingest", "discord post"]
    assert [(s.step, s.status) for s in result.steps if s.status == STATUS_SKIPPED] == [
        ("2", STATUS_SKIPPED), ("3", STATUS_SKIPPED),
    ]
    assert result.ok


def test_DB_がまだ無くても落ちずに手順1を実行する(cfg):
    world = World(cfg)
    result = run(cfg, world.runner)
    assert world.runner.kijun_calls()[:1] == ["ingest"] and result.ok


def test_LLM_の起動と停止のコマンドは実行しない(con, cfg):
    # keep_alive = "0" なので応答のたびに VRAM から下りる。
    add_untranscribed(con, "c1", "c2")
    con.close()
    world = World(cfg)
    run(cfg, world.runner)
    assert all(isinstance(c, list) for c in world.runner.calls)  # シェルのコマンドは1つも無い
    assert cfg.llm.keep_alive == "0"


# --- ゲームの実行中は延期する(batch/defer.py) -------------------------------------------------------------


def with_game(cfg, names=("F1_25.exe",), interval_min=30, max_hours=4):
    cfg.batch.defer_if_running = list(names)
    cfg.batch.defer_check_interval_min = interval_min
    cfg.batch.defer_max_hours = max_hours
    return cfg


def test_defer_if_running_が空なら検出そのものを行わない(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    lister = FakeProcessLister([{"f1_25.exe"}])
    clock = Clock()
    run(cfg, World(cfg).runner, lister, clock)
    assert lister.calls == 0 and clock.sleeps == []


def test_ゲームが動いていなければ待たずに全手順を実行する(con, cfg):
    with_game(cfg)
    add_untranscribed(con, "c1")
    con.close()
    lister = FakeProcessLister([{"chrome.exe"}])
    clock = Clock()
    world = World(cfg)
    result = run(cfg, world.runner, lister, clock)
    assert clock.sleeps == []
    assert lister.calls >= 2  # GPU を使う手順(文字起こし、議事録)の直前ごとに確かめた
    assert not any(s.step.endswith(":defer") for s in result.steps)
    assert "transcribe --conversation c1" in world.runner.kijun_calls() and result.ok


def test_ゲームが消えたら待ちをやめて再開し_待った時間と原因を記録する(con, cfg):
    with_game(cfg, names=("F1_25.exe", "LeagueClient.exe"))
    add_untranscribed(con, "c1")
    con.close()
    # 文字起こしの直前: 動いている → 動いている → 消えた。以降は動いていない。
    lister = FakeProcessLister([{"f1_25.exe"}, {"f1_25.exe"}, set()])
    clock = Clock()
    world = World(cfg)

    result = run(cfg, world.runner, lister, clock)

    assert clock.sleeps == [1800, 1800]  # 30 分ごとに再確認した
    defer = next(s for s in result.steps if s.step == "2:defer")
    assert defer.status == STATUS_OK
    assert "F1_25.exe" in defer.note and "60 分" in defer.note  # 設定に書いた表記で、待った時間を残す
    index = [s.step for s in result.steps].index("2:defer")
    assert result.steps[index + 1].step == "2" and result.steps[index + 1].status == STATUS_OK  # 待った後に実行
    assert "transcribe --conversation c1" in world.runner.kijun_calls() and result.ok


def test_プロセス名の大文字小文字は区別しない(con, cfg):
    with_game(cfg, names=("LEAGUEclient.EXE",))
    add_untranscribed(con, "c1")
    con.close()
    lister = FakeProcessLister([{"leagueclient.exe"}, set()])
    clock = Clock()
    run(cfg, World(cfg).runner, lister, clock)
    assert clock.sleeps == [1800]


def test_延期の上限に達したら_残りの_GPU_を使う手順を飛ばす_ingest_は実行する(con, cfg):
    with_game(cfg, interval_min=30, max_hours=1)
    add_untranscribed(con, "c1", "c2")
    con.close()
    lister = FakeProcessLister([{"f1_25.exe"}])  # 消えない
    clock = Clock()
    world = World(cfg)

    result = run(cfg, world.runner, lister, clock)

    assert clock.sleeps == [1800, 1800]  # 上限(1時間)で諦める。以降の手順では待たない
    # GPU を使うコマンド(文字起こし)は起動しない。投稿は GPU を使わないので実行する。
    assert world.runner.kijun_calls() == ["ingest", "discord post"]
    skipped = [s for s in result.steps if s.status == STATUS_SKIPPED]
    assert [(s.step, s.name) for s in skipped] == [
        ("2", "文字起こし c1"), ("2", "文字起こし c2"),
        ("3", "議事録"),  # 文字起こし済みの会議が無いので、議事録の対象も無い
    ]
    assert "延期の上限に達した" in skipped[0].note and "F1_25.exe" in skipped[0].note
    defer = [s for s in result.steps if s.step.endswith(":defer")]
    assert len(defer) == 1 and defer[0].step == "2:defer" and "延期の上限に達した" in defer[0].note
    assert result.ok  # 飛ばしただけで、失敗ではない


def test_上限に達した後は_ゲームを再確認しない(con, cfg):
    with_game(cfg, interval_min=30, max_hours=1)
    add_untranscribed(con, "c1")
    con.close()
    lister = FakeProcessLister([{"f1_25.exe"}])
    run(cfg, World(cfg).runner, lister, Clock())
    # 諦めた時点までに確認した回数(最初 + 30 分後 + 60 分後)だけ。以降の手順では検出しない。
    assert lister.calls == 3


def test_待った時間はバッチ全体で合計する(con, cfg):
    with_game(cfg, interval_min=30, max_hours=1)
    add_untranscribed(con, "c1")
    mark_transcribed(con, "c1")  # 文字起こし済み。議事録の対象
    add_untranscribed(con, "c2")  # 文字起こし前。文字起こしの対象
    con.close()
    # 文字起こしの直前に 30 分待ってゲームが消え、議事録の直前に再びゲームが動いている。
    lister = FakeProcessLister([{"f1_25.exe"}, set(), {"f1_25.exe"}])
    clock = Clock()
    world = World(cfg)

    result = run(cfg, world.runner, lister, clock)

    # 上限は 1 時間。文字起こしの前に 30 分使ったので、議事録の前に待てるのは残り 30 分だけ。
    assert clock.sleeps == [1800, 1800]
    calls = world.runner.kijun_calls()
    assert "transcribe --conversation c2" in calls
    assert not any(c.startswith("minutes") for c in calls)  # 上限に達したので飛ばした
    assert any(s.step == "3" and s.status == STATUS_SKIPPED and "延期の上限" in s.note for s in result.steps)


def test_残りの時間より長くは待たない(con, cfg):
    with_game(cfg, interval_min=30, max_hours=0.75)  # 45 分
    add_untranscribed(con, "c1")
    con.close()
    clock = Clock()
    run(cfg, World(cfg).runner, FakeProcessLister([{"f1_25.exe"}]), clock)
    assert clock.sleeps == [1800, 900]


def test_延期の上限が_0_なら待たずに飛ばす(con, cfg):
    with_game(cfg, max_hours=0)
    add_untranscribed(con, "c1")
    con.close()
    clock = Clock()
    world = World(cfg)
    run(cfg, world.runner, FakeProcessLister([{"f1_25.exe"}]), clock)
    assert clock.sleeps == []
    assert "transcribe --conversation c1" not in world.runner.kijun_calls()


def test_ゲームで飛ばした会議は翌晩に未処理として拾われる(con, cfg):
    from kijun.batch.pending import pending_minutes, pending_transcription

    with_game(cfg, max_hours=0)
    add_untranscribed(con, "c1", "c2")  # 文字起こし前
    add_untranscribed(con, "c3")
    mark_transcribed(con, "c3")  # 文字起こし済み・議事録前
    con.close()

    run(cfg, World(cfg).runner, FakeProcessLister([{"f1_25.exe"}]), Clock())  # 1 晩目: ゲームで全部飛ばす

    con = repo.connect(cfg.paths.db)
    try:
        assert pending_transcription(con) == ["c1", "c2"]  # DB は 1 晩目で何も変わっていない
        assert pending_minutes(con) == ["c3"]
    finally:
        con.close()

    second = World(cfg)
    run(cfg, second.runner, FakeProcessLister([set()]), Clock())  # 2 晩目: ゲームが動いていない
    calls = second.runner.kijun_calls()
    assert "transcribe --conversation c1" in calls and "transcribe --conversation c2" in calls
    assert "minutes --conversation c3" in calls


def test_ゲームの検出に失敗したら_警告して動いていないものとして続ける(con, cfg):
    from kijun.batch.defer import ProcessListError

    with_game(cfg)
    add_untranscribed(con, "c1")
    con.close()
    messages: list[str] = []
    world = World(cfg)
    result = run_batch(
        cfg, None, world.runner, FakeProcessLister([ProcessListError("tasklist が失敗")]),
        clock=CLOCK, sleep=lambda s: None, echo=messages.append,
    )
    assert "transcribe --conversation c1" in world.runner.kijun_calls() and result.ok
    assert any("ゲームの実行を確認できませんでした" in m for m in messages)


def test_dry_run_は開始時刻とゲームの待機を表示する(cfg):
    with_game(cfg, names=("F1_25.exe", "LeagueClient.exe"), interval_min=20, max_hours=3)
    cfg.batch.task_time = "22:40"
    text = "\n".join(plan_lines(cfg, None))
    assert "22:40" in text
    assert "F1_25.exe, LeagueClient.exe" in text and "20 分ごと" in text and "3 時間" in text
    assert "start_command" not in text and "Ollama" in text


def test_dry_run_はゲーム検出が空なら検出しないと表示する(cfg):
    assert "検出は行わない" in "\n".join(plan_lines(cfg, None))


# --- 記録 ---------------------------------------------------------------------------------------------


def test_全手順の記録が_JSON_で保存される(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg, fail_exact={"discord post"})
    result = run(cfg, world.runner)

    path = write_log(cfg, result)

    assert path.parent == cfg.batch.log_dir and path.name == "20261104_233000.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ok"] is False and data["started_at"] == "2026-11-04T23:30:00"
    last = data["steps"][-1]
    assert (last["step"], last["status"], last["exit_code"]) == ("4", "failed", 1)
    assert "discord post" in last["command"] and "エラー: discord post が失敗" in last["output_tail"]
    assert {"started_at", "ended_at", "seconds"} <= set(last)


def test_最後の表に全手順の成否が出る(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    result = run(cfg, World(cfg, fail_exact={"discord post"}).runner)
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
    run_batch(cfg, tmp_path / "my.toml", world.runner, clock=CLOCK, echo=lambda m: None)
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


def test_ゲームの上限に達しても_GPU_を使わない_discord_post_と_ingest_は実行する(con, cfg):
    with_game(cfg, max_hours=0)
    add_untranscribed(con, "c1")
    con.close()
    world = World(cfg)
    result = run(cfg, world.runner, FakeProcessLister([{"f1_25.exe"}]), Clock())
    assert world.runner.kijun_calls() == ["ingest", "discord post"]
    assert ("2", STATUS_SKIPPED) in steps(result) and result.ok


# --- keep_alive が "0" 以外のとき: モデルを下ろす要求 ---------------------------------------------------------


class _UnloadTransport:
    """POST /api/generate だけを受け付ける偽の HttpTransport。呼ばれた順を共有のログに残す。"""

    def __init__(self, log: list[str], status: int = 200, error: Exception | None = None) -> None:
        self.log, self.status, self.error = log, status, error
        self.posts: list[dict] = []

    def post(self, url, payload, timeout_sec):
        from kijun.transport import HttpResponse

        self.posts.append({"url": url, "payload": payload})
        self.log.append("unload")
        if self.error is not None:
            raise self.error
        return HttpResponse(self.status, "{}", 0.1)

    def get(self, url, timeout_sec):  # pragma: no cover
        raise AssertionError("使わない")


def _run_with_unload(cfg, transport, log, world_cls=World):
    world = world_cls(cfg)
    inner = world.handle

    def handler(command):
        if isinstance(command, list):
            args = command[3:]
            if args[:1] == ["--config"]:
                args = args[2:]
            log.append(" ".join(args))
        return inner(command)

    world.runner = FakeCommandRunner(handler)
    result = run_batch(cfg, None, world.runner, clock=CLOCK, sleep=lambda s: None, echo=lambda m: None, transport=transport)
    return result, world


def test_keep_alive_が_0_ならモデルを下ろす要求を送らない(con, cfg):
    add_untranscribed(con, "c1")
    con.close()
    log: list[str] = []
    transport = _UnloadTransport(log)
    assert cfg.llm.keep_alive == "0"
    _run_with_unload(cfg, transport, log)
    assert transport.posts == []


def test_keep_alive_が_0_以外なら_手順3の後にモデルを下ろす(con, cfg):
    cfg.llm.keep_alive = "5m"
    add_untranscribed(con, "c1", "c2")
    con.close()
    log: list[str] = []
    transport = _UnloadTransport(log)

    result, _ = _run_with_unload(cfg, transport, log)

    # 議事録の後(Discord への投稿の前)に下ろす。文字起こしは議事録より前なので、LLM が VRAM に残らない。
    assert log == [
        "ingest",
        "transcribe --conversation c1", "transcribe --conversation c2",
        "minutes --conversation c1", "minutes --conversation c2", "unload",
        "discord post",
    ]
    assert transport.posts[0]["url"] == "http://127.0.0.1:11434/api/generate"
    assert transport.posts[0]["payload"] == {"model": "qwen3.5:9b", "keep_alive": 0}
    assert [s.step for s in result.steps if s.step.endswith(":unload")] == ["3:unload"]
    assert result.ok


def test_議事録の対象が無ければ_手順3の後には下ろさない(con, cfg):
    cfg.llm.keep_alive = "5m"
    con.close()
    log: list[str] = []
    transport = _UnloadTransport(log)
    _run_with_unload(cfg, transport, log)
    assert log.count("unload") == 0


def test_モデルを下ろす要求が失敗しても_バッチは続け_失敗として記録する(con, cfg):
    from kijun.transport import TransportError

    cfg.llm.keep_alive = "5m"
    add_untranscribed(con, "c1")
    con.close()
    log: list[str] = []
    result, world = _run_with_unload(cfg, _UnloadTransport(log, error=TransportError("接続拒否")), log)
    assert "transcribe --conversation c1" in log  # 続ける
    unload = next(s for s in result.steps if s.step == "3:unload")
    assert unload.status == STATUS_FAILED and "接続拒否" in unload.note and not result.ok

    con = repo.connect(cfg.paths.db)
    add_untranscribed(con, "c2")  # 1 回目で c1 の議事録は記録済み。新しい会議で 2 回目を試す
    con.close()
    log2: list[str] = []
    result2, _ = _run_with_unload(cfg, _UnloadTransport(log2, status=500), log2)
    assert any(s.status == STATUS_FAILED and "HTTP 500" in s.note for s in result2.steps)


def test_dry_run_は_keep_alive_に応じてアンロードの有無を表示する(cfg):
    assert "アンロードの要求は送らない" in "\n".join(plan_lines(cfg, None))
    cfg.llm.keep_alive = "5m"
    assert "モデルを下ろす要求を送る" in "\n".join(plan_lines(cfg, None))
