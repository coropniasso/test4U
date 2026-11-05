"""夜間バッチの手順(仕様 3-1、3-2、テスト 9)。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from kijun.batch.run import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, format_summary, plan_lines, run_batch, write_log
from kijun.db import repo
from tests.conftest import make_conversation, make_utterances
from tests.fakes import FakeCommandRunner, FakeProcessLister
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
        extract_log_path(cfg, "c1", cfg.extract.model_name()), "c1",
        ExtractionRun(model_name=cfg.extract.model_name(), chunk_count=2), datetime(2026, 11, 3),
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
    model = cfg.extract.model_name()
    failed_run = ExtractionRun(model_name=model, chunk_count=2, failures=[ChunkFailure(0, 0, 1, "x"), ChunkFailure(1, 1, 2, "y")])
    write_extract_log(extract_log_path(cfg, "c1", model), "c1", failed_run, datetime(2026, 11, 3))
    con.close()
    world = World(cfg)
    run(cfg, world.runner)
    assert "extract --conversation c1" in world.runner.kijun_calls()


def test_照合は未紐付けの抽出結果がある会議だけ実行する(con, cfg):
    candidate(con, 1)  # c: 抽出結果は紐付け済み
    from tests.review_helpers import CID

    add_request(con, "u1", conversation_id=CID, model=cfg.extract.model_name())  # 紐付いていない
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


# --- 手順6: LLM(Ollama)は起動も停止もしない ----------------------------------------------------------


def test_LLM_の起動と停止のコマンドは実行しない(con, cfg):
    # 2026-11-05 に Strata から Ollama へ移った。keep_alive = "0" なので応答のたびに VRAM から下りる。
    add_untranscribed(con, "c1", "c2")
    con.close()
    world = World(cfg)
    run(cfg, world.runner)
    assert all(isinstance(c, list) for c in world.runner.calls)  # シェルのコマンドは1つも無い
    assert cfg.extract.ollama.keep_alive == "0"


def test_手順1は未処理の投稿があるときだけゲームを確かめる(con, cfg):
    cfg.batch.defer_if_running = ["Game.exe"]
    cfg.discord.review_channel_id = "111"
    con.close()
    lister = FakeProcessLister([set()])
    run(cfg, World(cfg).runner, lister)
    # 未処理の投稿が無いので LLM は要らない。手順1の前には確認しない(後の手順は DB が無く、対象も無い)。
    assert lister.calls == 0

    con = repo.connect(cfg.paths.db)
    candidate(con, 1)
    repo.insert_discord_post(con, repo.DiscordPostRow("m1", "new_candidate", "item_0001", None, datetime(2026, 11, 3), None, "111", None))
    con.close()
    lister2 = FakeProcessLister([set()])
    world2 = World(cfg)
    run(cfg, world2.runner, lister2)
    assert lister2.calls >= 1  # 返信の振り分けに LLM を使うので、手順1の前に確かめた
    assert world2.runner.kijun_calls()[0] == "discord read"


def test_手順8は重要度が未判定の候補があるときだけゲームを確かめる(con, cfg):
    from tests.review_helpers import CID

    cfg.batch.defer_if_running = ["Game.exe"]
    cfg.discord.review_channel_id = "111"
    cfg.extract.ollama.model = "m"  # candidate() が作る抽出結果のモデル名。抽出済みとみなされる
    candidate(con, 1, importance=3)  # 判定済み。抽出・照合も済み
    mark_transcribed(con, CID)  # 文字起こし済み。手順8以外に GPU を使う手順が残らない
    con.close()
    judged = FakeProcessLister([set()])
    run(cfg, World(cfg).runner, judged)
    assert judged.calls == 0  # LLM は要らない

    con = repo.connect(cfg.paths.db)
    con.execute("UPDATE knowledge_items SET importance = NULL")
    con.close()
    unjudged = FakeProcessLister([set()])
    run(cfg, World(cfg).runner, unjudged)
    assert unjudged.calls == 1  # 重要度の判定に LLM を使うので、投稿の前に確かめた


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
    assert lister.calls >= 2  # GPU を使う手順(文字起こし、抽出)の直前ごとに確かめた
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
    defer = next(s for s in result.steps if s.step == "3-4:defer")
    assert defer.status == STATUS_OK
    assert "F1_25.exe" in defer.note and "60 分" in defer.note  # 設定に書いた表記で、待った時間を残す
    index = [s.step for s in result.steps].index("3-4:defer")
    assert result.steps[index + 1].step == "3-4" and result.steps[index + 1].status == STATUS_OK  # 待った後に実行
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
    # GPU を使うコマンド(文字起こし)は起動しない。投稿は LLM が要らない(重要度が未判定の候補が無い)ので実行する。
    assert world.runner.kijun_calls() == ["discord read", "ingest", "discord post"]
    skipped = [s for s in result.steps if s.status == STATUS_SKIPPED]
    assert [(s.step, s.name) for s in skipped] == [
        ("3-4", "文字起こし c1"), ("3-4", "文字起こし c2"),
        ("6", "抽出"),  # 文字起こし済みの会議が無いので、抽出の対象も無い
        ("7", "照合"),
    ]
    assert "延期の上限に達した" in skipped[0].note and "F1_25.exe" in skipped[0].note
    defer = [s for s in result.steps if s.step.endswith(":defer")]
    assert len(defer) == 1 and defer[0].step == "3-4:defer" and "延期の上限に達した" in defer[0].note
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
    mark_transcribed(con, "c1")  # 文字起こし済み。抽出の対象
    add_untranscribed(con, "c2")  # 文字起こし前。文字起こしの対象
    con.close()
    # 文字起こしの直前に 30 分待ってゲームが消え、抽出の直前に再びゲームが動いている。
    lister = FakeProcessLister([{"f1_25.exe"}, set(), {"f1_25.exe"}])
    clock = Clock()
    world = World(cfg)

    result = run(cfg, world.runner, lister, clock)

    # 上限は 1 時間。文字起こしの前に 30 分使ったので、抽出の前に待てるのは残り 30 分だけ。
    assert clock.sleeps == [1800, 1800]
    calls = world.runner.kijun_calls()
    assert "transcribe --conversation c2" in calls
    assert not any(c.startswith("extract") for c in calls)  # 上限に達したので飛ばした
    assert any(s.step == "6" and s.status == STATUS_SKIPPED and "延期の上限" in s.note for s in result.steps)


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
    from kijun.batch.pending import pending_extraction, pending_transcription

    with_game(cfg, max_hours=0)
    add_untranscribed(con, "c1", "c2")  # 文字起こし前
    add_untranscribed(con, "c3")
    mark_transcribed(con, "c3")  # 文字起こし済み・抽出前
    con.close()

    run(cfg, World(cfg).runner, FakeProcessLister([{"f1_25.exe"}]), Clock())  # 1 晩目: ゲームで全部飛ばす

    con = repo.connect(cfg.paths.db)
    try:
        assert pending_transcription(con) == ["c1", "c2"]  # DB は 1 晩目で何も変わっていない
        assert pending_extraction(con, cfg) == ["c3"]
    finally:
        con.close()

    second = World(cfg)
    run(cfg, second.runner, FakeProcessLister([set()]), Clock())  # 2 晩目: ゲームが動いていない
    calls = second.runner.kijun_calls()
    assert "transcribe --conversation c1" in calls and "transcribe --conversation c2" in calls
    assert "extract --conversation c3" in calls


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


def test_ゲームの上限に達したら_LLM_が要る手順8は飛ばし_要らない手順8は実行する(con, cfg):
    from tests.review_helpers import CID

    with_game(cfg, max_hours=0)
    cfg.discord.review_channel_id = "111"
    cfg.extract.ollama.model = "m"
    candidate(con, 1, importance=None)  # 重要度が未判定。投稿の前に LLM が要る
    mark_transcribed(con, CID)
    con.close()
    world = World(cfg)
    result = run(cfg, world.runner, FakeProcessLister([{"f1_25.exe"}]), Clock())
    assert "discord post" not in world.runner.kijun_calls()
    assert ("8", STATUS_SKIPPED) in steps(result) and result.ok

    con = repo.connect(cfg.paths.db)
    con.execute("UPDATE knowledge_items SET importance = 2")
    con.close()
    world2 = World(cfg)
    run(cfg, world2.runner, FakeProcessLister([{"f1_25.exe"}]), Clock())
    assert "discord post" in world2.runner.kijun_calls()  # LLM が要らないので、ゲームが動いていても投稿する


def test_ゲームの上限に達したら_LLM_が要る手順1は飛ばすが_ingest_は実行する(con, cfg):
    with_game(cfg, max_hours=0)
    cfg.discord.review_channel_id = "111"
    candidate(con, 1)
    repo.insert_discord_post(con, repo.DiscordPostRow("m1", "new_candidate", "item_0001", None, datetime(2026, 11, 3), None, "111", None))
    con.close()
    world = World(cfg)
    result = run(cfg, world.runner, FakeProcessLister([{"f1_25.exe"}]), Clock())
    calls = world.runner.kijun_calls()
    assert "discord read" not in calls and "ingest" in calls
    assert ("1", STATUS_SKIPPED) in steps(result)


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
    assert cfg.extract.ollama.keep_alive == "0"
    _run_with_unload(cfg, transport, log)
    assert transport.posts == []


def test_keep_alive_が_0_以外なら_手順1と手順6の後にモデルを下ろす(con, cfg):
    cfg.extract.ollama.keep_alive = "5m"
    add_untranscribed(con, "c1", "c2")
    con.close()
    log: list[str] = []
    transport = _UnloadTransport(log)

    result, _ = _run_with_unload(cfg, transport, log)

    # 手順1の後(文字起こしの前)と、手順6の後(照合の前)に下ろす。文字起こしの時点で LLM が VRAM に残らないようにするため。
    assert log == [
        "discord read", "unload", "ingest",
        "transcribe --conversation c1", "transcribe --conversation c2",
        "extract --conversation c1", "extract --conversation c2", "unload",
        "discord post",
    ]
    assert transport.posts[0]["url"] == "http://127.0.0.1:11434/api/generate"
    assert transport.posts[0]["payload"] == {"model": "qwen3.5:9b", "keep_alive": 0}
    assert [s.step for s in result.steps if s.step.endswith(":unload")] == ["1:unload", "6:unload"]
    assert result.ok


def test_抽出の対象が無ければ_手順6の後には下ろさない(con, cfg):
    cfg.extract.ollama.keep_alive = "5m"
    con.close()
    log: list[str] = []
    transport = _UnloadTransport(log)
    _run_with_unload(cfg, transport, log)
    assert log.count("unload") == 1  # 手順1の後だけ


def test_モデルを下ろす要求が失敗しても_バッチは続け_失敗として記録する(con, cfg):
    from kijun.transport import TransportError

    cfg.extract.ollama.keep_alive = "5m"
    add_untranscribed(con, "c1")
    con.close()
    log: list[str] = []
    result, world = _run_with_unload(cfg, _UnloadTransport(log, error=TransportError("接続拒否")), log)
    assert "transcribe --conversation c1" in log  # 続ける
    unload = next(s for s in result.steps if s.step == "1:unload")
    assert unload.status == STATUS_FAILED and "接続拒否" in unload.note and not result.ok

    log2: list[str] = []
    result2, _ = _run_with_unload(cfg, _UnloadTransport(log2, status=500), log2)
    assert any(s.status == STATUS_FAILED and "HTTP 500" in s.note for s in result2.steps)


def test_dry_run_は_keep_alive_に応じてアンロードの有無を表示する(cfg):
    assert "アンロードの要求は送らない" in "\n".join(plan_lines(cfg, None))
    cfg.extract.ollama.keep_alive = "5m"
    assert "モデルを下ろす要求を送る" in "\n".join(plan_lines(cfg, None))
