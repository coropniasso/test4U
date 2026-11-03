"""Strata の起動・停止の窓(仕様 3-1 の手順6、docs/implementation-plan.md 2-2)。

faster-whisper・pyannote と Strata(Qwen3.8-Flash-Next)を同時に VRAM に載せないため、Strata を使う手順の前後で
設定の extract.strata.start_command と stop_command を実行する。Strata に Ollama の keep_alive: 0 に
相当するアンロードの手段があるかは未確認なので、プロセスごと止める。

- start_command は、Strata を起動したらすぐに戻るコマンド(バックグラウンドで起動するもの)を前提にする。
  戻らなければ batch.strata_ready_timeout_sec で打ち切って失敗にする。
- start_command が戻ったあと、Strata が応答するまで待つ。125B の MoE の読み込みには時間がかかり、
  応答する前に抽出を始めると接続できず毎晩失敗するため。応答の確認は doctor と同じ /v1/models への GET。
- stop_command は、start が失敗しても、窓の中の処理が失敗しても、必ず実行する(Strata が VRAM を保持したまま
  残らないようにするため)。
- 空文字のコマンドは何もしない。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from kijun.batch.runner import CommandRunner
from kijun.config import Config
from kijun.transport import HttpTransport, TransportError

# Strata の応答を確認する間隔(秒)
READY_POLL_INTERVAL_SEC = 5.0


@dataclass(frozen=True)
class WindowEvent:
    """窓の中で実行したコマンド1つ(start / ready / stop)の結果。"""

    name: str
    command: str
    ok: bool
    exit_code: int | None
    output_tail: str
    started_at: datetime
    ended_at: datetime


class StrataWindow:
    """with 文で使う。入ったときに起動して応答を待ち、出るときに必ず停止する。

    `with window as ok:` の ok が False なら起動に失敗している(窓の中の処理は飛ばす)。
    """

    def __init__(
        self,
        cfg: Config,
        runner: CommandRunner,
        transport: HttpTransport,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._cfg = cfg
        self._runner = runner
        self._transport = transport
        self._sleep, self._monotonic, self._now = sleep, monotonic, now
        self.events: list[WindowEvent] = []

    def _record(self, name: str, command: str, result) -> bool:
        ok = result.exit_code == 0
        self.events.append(
            WindowEvent(name, command, ok, result.exit_code, result.output_tail, result.started_at, result.ended_at)
        )
        return ok

    def _wait_ready(self) -> bool:
        strata = self._cfg.extract.strata
        url = f"{strata.base_url}/v1/models"
        timeout = self._cfg.batch.strata_ready_timeout_sec
        started_at, deadline = self._now(), self._monotonic() + timeout
        last_error = "応答がありません"
        while True:
            try:
                response = self._transport.get(url, timeout_sec=30)
                if response.ok:
                    self.events.append(WindowEvent("ready", url, True, 0, "応答しました", started_at, self._now()))
                    return True
                last_error = f"HTTP {response.status_code}"
            except TransportError as e:
                last_error = str(e)
            if self._monotonic() >= deadline:
                break
            self._sleep(READY_POLL_INTERVAL_SEC)
        self.events.append(
            WindowEvent(
                "ready", url, False, None,
                f"{timeout} 秒待っても Strata が応答しませんでした({last_error})", started_at, self._now(),
            )
        )
        return False

    def __enter__(self) -> bool:
        command = self._cfg.extract.strata.start_command
        if not command:
            return True
        result = self._runner.run(command, timeout_sec=self._cfg.batch.strata_ready_timeout_sec)
        if not self._record("start", command, result):
            return False
        return self._wait_ready()

    def __exit__(self, exc_type, exc, tb) -> None:
        command = self._cfg.extract.strata.stop_command
        if command:
            self._record("stop", command, self._runner.run(command))
