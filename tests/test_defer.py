"""ゲームの実行中の延期(batch/defer.py)の単体テスト。バッチ全体での挙動は tests/test_batch.py にある。"""

from __future__ import annotations

import subprocess
from datetime import datetime

import pytest

from kijun.batch.defer import (
    GameDeferrer, ProcessListError, TasklistProcessLister, parse_tasklist_csv,
)
from kijun.config import BatchConfig
from tests.fakes import FakeProcessLister

SAMPLE = (
    '"System Idle Process","0","Services","0","8 K"\n'
    '"F1_25.exe","12345","Console","1","3,456,789 K"\n'
    '"LeagueClient.exe","222","Console","1","100 K"\n'
)


def deferrer(lister, names=("F1_25.exe",), interval=30, max_hours=4):
    cfg = BatchConfig(defer_if_running=list(names), defer_check_interval_min=interval, defer_max_hours=max_hours)
    sleeps: list[float] = []
    warnings: list[str] = []
    d = GameDeferrer(cfg, lister, sleep=sleeps.append, clock=lambda: datetime(2026, 11, 5, 23, 30), warn=warnings.append)
    return d, sleeps, warnings


def test_tasklist_の_CSV_から1列目を小文字で取り出す():
    assert parse_tasklist_csv(SAMPLE) == {"system idle process", "f1_25.exe", "leagueclient.exe"}
    assert parse_tasklist_csv("") == set()


def test_TasklistProcessLister_は_UTF8_でも_cp932_でも読む(monkeypatch):
    def fake_run(output):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run(SAMPLE.encode("utf-8")))
    assert "f1_25.exe" in TasklistProcessLister().running_names()
    # cp932 の日本語を含み、UTF-8 では読めない出力
    monkeypatch.setattr(subprocess, "run", fake_run('"ゲーム.exe","1","Console","1","1 K"\n"F1_25.exe","2","C","1","1 K"\n'.encode("cp932")))
    assert "f1_25.exe" in TasklistProcessLister().running_names()


def test_TasklistProcessLister_は失敗を_ProcessListError_にする(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout=b"", stderr=b"x"))
    with pytest.raises(ProcessListError, match="終了コード 1"):
        TasklistProcessLister().running_names()

    def boom(*a, **k):
        raise FileNotFoundError("tasklist")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(ProcessListError, match="実行できません"):
        TasklistProcessLister().running_names()


def test_空の設定では検出しない():
    lister = FakeProcessLister([{"f1_25.exe"}])
    d, sleeps, _ = deferrer(lister, names=())
    r = d.gate()
    assert r.proceed and lister.calls == 0 and sleeps == [] and not d.enabled


def test_動いていなければそのまま進む():
    d, sleeps, _ = deferrer(FakeProcessLister([{"chrome.exe"}]))
    r = d.gate()
    assert r.proceed and r.waited_sec == 0 and sleeps == [] and r.blocked_by == []


def test_消えるまで間隔ごとに待つ():
    d, sleeps, _ = deferrer(FakeProcessLister([{"f1_25.exe"}] * 3 + [set()]), interval=20)
    r = d.gate()
    assert r.proceed and sleeps == [1200, 1200, 1200] and r.waited_sec == 3600
    assert r.blocked_by == ["F1_25.exe"] and r.started_at is not None
    assert d.total_waited_sec == 3600 and not d.gave_up


def test_上限に達したら諦め_以降は検出も待ちもしない():
    lister = FakeProcessLister([{"f1_25.exe"}])
    d, sleeps, _ = deferrer(lister, interval=30, max_hours=1)
    first = d.gate()
    assert not first.proceed and first.gave_up_now and sleeps == [1800, 1800] and first.waited_sec == 3600
    calls = lister.calls
    again = d.gate()
    assert not again.proceed and not again.gave_up_now and again.waited_sec == 0
    assert again.blocked_by == ["F1_25.exe"] and lister.calls == calls and sleeps == [1800, 1800]


def test_複数のプロセスのうち動いているものを原因に挙げる():
    d, _, _ = deferrer(FakeProcessLister([{"leagueclient.exe"}, set()]), names=("F1_25.exe", "LeagueClient.exe"))
    assert d.gate().blocked_by == ["LeagueClient.exe"]


def test_最後の待機で消えたなら進む_ちょうど上限でも進める():
    # 1 時間の上限に対し、2 回目の待機(合計 60 分)の直後に消えた場合は、上限に達していても進める。
    d, sleeps, _ = deferrer(FakeProcessLister([{"f1_25.exe"}, {"f1_25.exe"}, set()]), interval=30, max_hours=1)
    r = d.gate()
    assert r.proceed and sleeps == [1800, 1800]


def test_一覧を取得できなければ警告して動いていないものとして進む():
    d, sleeps, warnings = deferrer(FakeProcessLister([ProcessListError("tasklist が失敗")]))
    r = d.gate()
    assert r.proceed and sleeps == []
    assert len(warnings) == 1 and "ゲームの実行を確認できませんでした" in warnings[0]
