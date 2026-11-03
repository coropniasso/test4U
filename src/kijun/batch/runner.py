"""外部コマンドの実行の境界(原則2-1)。

夜間バッチは、既存の CLI(kijun ingest など)を別プロセスとして起動する。原則2-2(プロセスの境界を VRAM の境界にする)
を守るためで、1つの Python プロセスの中で全部を呼ぶと、faster-whisper と Strata が同時に VRAM を使う危険が出る。
テストでは tests/fakes.py の FakeCommandRunner を差し込み、実際のプロセスを起動しない。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

# 記録に残す出力の末尾の文字数。全文を残すとログが大きくなり、末尾にエラーの原因が出ることが多い。
OUTPUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class CommandResult:
    exit_code: int | None  # 時間切れで終了コードが得られなかった場合は None
    output_tail: str  # 標準出力と標準エラー出力をまとめた末尾
    started_at: datetime
    ended_at: datetime
    timed_out: bool = False


class CommandRunner(Protocol):
    def run(self, command: list[str] | str, timeout_sec: float | None = None) -> CommandResult:
        """コマンドを実行して終わるまで待つ。list はそのまま起動し、str はシェルで実行する(設定の start_command など)。"""
        ...


class SubprocessRunner:
    def run(self, command: list[str] | str, timeout_sec: float | None = None) -> CommandResult:
        started = datetime.now()
        # 子プロセスの出力は UTF-8 に固定する。Windows の既定(cp932)のままだと、日本語の出力が文字化けして記録される。
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.run(
                command,
                shell=isinstance(command, str),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # エラーは標準エラーに出るので、まとめて末尾に残す
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            partial = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return CommandResult(None, partial[-OUTPUT_TAIL_CHARS:], started, datetime.now(), timed_out=True)
        except OSError as e:
            return CommandResult(None, f"起動できませんでした: {e}", started, datetime.now())
        return CommandResult(proc.returncode, (proc.stdout or "")[-OUTPUT_TAIL_CHARS:], started, datetime.now())
