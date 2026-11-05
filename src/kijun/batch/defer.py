"""ゲームの実行中は、GPU を使う手順を延期する(CLAUDE.md 4-8、docs/history/implementation-plan.md 15-2)。

夜間バッチの GPU を使う手順(文字起こし、LLM)は VRAM を使う。同じ PC でゲームが動いていると
競合するので、設定 batch.defer_if_running に挙げた実行ファイルのどれかが動いている間は、その手順を
始めずに待つ。

- 待った時間はバッチ全体で合計する。batch.defer_max_hours を超えたら、その夜の残りの GPU を使う手順は
  すべて飛ばす(以降は検出もしない)。飛ばした会議は、翌晩に batch/pending.py が未処理として拾う。
- batch.defer_if_running が空なら、検出そのものを行わない。
- プロセスの一覧は tasklist の出力から読む。新しい依存(psutil)は足さない。
  tasklist が失敗して一覧が得られないときは、警告を出して「動いていない」とみなす。検出の失敗で
  バッチが毎晩止まるより、検出できなかったことを警告して続ける方を取った。
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from kijun.config import BatchConfig


class ProcessListError(Exception):
    """プロセスの一覧を取得できなかったとき。"""


class ProcessLister(Protocol):
    def running_names(self) -> set[str]:
        """いま動いているプロセスの実行ファイル名を、小文字で返す。取得できなければ ProcessListError。"""
        ...


def parse_tasklist_csv(text: str) -> set[str]:
    """`tasklist /FO CSV /NH` の出力から、1列目(イメージ名)を小文字で取り出す。"""
    names: set[str] = set()
    for row in csv.reader(io.StringIO(text)):
        if row and row[0].strip():
            names.add(row[0].strip().lower())
    return names


class TasklistProcessLister:
    """Windows の tasklist を実行して一覧を得る。"""

    def running_names(self) -> set[str]:
        try:
            proc = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            raise ProcessListError(f"tasklist を実行できませんでした: {e}") from e
        if proc.returncode != 0:
            raise ProcessListError(f"tasklist が終了コード {proc.returncode} で終わりました")
        # 日本語 Windows のコンソールの文字コードは cp932。UTF-8 で読めなければ cp932 で読む。
        try:
            text = proc.stdout.decode("utf-8")
        except UnicodeDecodeError:
            text = proc.stdout.decode("cp932", errors="replace")
        return parse_tasklist_csv(text)


@dataclass
class GateResult:
    """GPU を使う手順を始める直前の確認 1 回の結果。"""

    proceed: bool
    waited_sec: float = 0.0  # この確認で待った時間
    blocked_by: list[str] = field(default_factory=list)  # 原因のプロセス名(設定に書いた表記)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    # 上限に達してこの確認で初めて諦めた(以前の確認で諦めていたのではない)
    gave_up_now: bool = False


class GameDeferrer:
    def __init__(
        self,
        cfg: BatchConfig,
        lister: ProcessLister | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = datetime.now,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._lister = lister
        self._sleep, self._clock = sleep, clock
        self._warn = warn or (lambda m: print(m, file=sys.stderr))
        # 小文字の名前 -> 設定に書いた表記
        self._watched = {n.strip().lower(): n.strip() for n in cfg.defer_if_running}
        self.total_waited_sec = 0.0
        self.gave_up = False
        self.gave_up_by: list[str] = []

    @property
    def enabled(self) -> bool:
        return bool(self._watched)

    def _running(self) -> list[str]:
        """いま動いている、監視対象のプロセス名(設定の表記)。一覧を取れなければ警告を出して空。"""
        lister = self._lister or TasklistProcessLister()
        self._lister = lister
        try:
            names = lister.running_names()
        except ProcessListError as e:
            self._warn(f"警告: ゲームの実行を確認できませんでした。動いていないものとして続けます: {e}")
            return []
        return [original for lowered, original in self._watched.items() if lowered in names]

    def gate(self) -> GateResult:
        """GPU を使う手順を始めてよいか確認する。動いていれば、消えるか上限に達するまで待つ。"""
        if not self.enabled:
            return GateResult(True)
        if self.gave_up:
            return GateResult(False, blocked_by=list(self.gave_up_by))
        blocking = self._running()
        if not blocking:
            return GateResult(True)
        started = self._clock()
        observed = list(blocking)
        waited = 0.0
        max_sec = self._cfg.defer_max_hours * 3600
        interval_sec = self._cfg.defer_check_interval_min * 60
        while blocking:
            remaining = max_sec - self.total_waited_sec
            if remaining <= 0:
                self.gave_up, self.gave_up_by = True, observed
                return GateResult(False, waited, observed, started, self._clock(), gave_up_now=True)
            nap = min(interval_sec, remaining)
            self._sleep(nap)
            self.total_waited_sec += nap
            waited += nap
            blocking = self._running()
            observed += [n for n in blocking if n not in observed]
        return GateResult(True, waited, observed, started, self._clock())
