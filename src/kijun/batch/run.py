"""夜間バッチの本体(仕様 3-1、3-2、CLAUDE.md 4-8)。

既存の CLI を subprocess で別プロセスとして順に起動する。ここには処理の中身を書かない
(原則2-2。1つの Python プロセスの中で全部を呼ぶと faster-whisper と Strata が同時に VRAM を使う危険が出る)。

| 手順 | 実行するもの | 失敗したとき |
|---|---|---|
| 1 | kijun discord read | 続ける |
| 2 | kijun ingest | **止める** |
| 3-4 | kijun transcribe --conversation <ID>(会議ごと) | その会議だけ飛ばす |
| 5 | (手順3-4のプロセスの終了。OS が GPU メモリを解放する) | - |
| 6 | start_command -> kijun extract --conversation <ID> -> stop_command | その会議だけ飛ばす |
| 7 | kijun match --conversation <ID>(会議ごと) | その会議だけ飛ばす |
| 8 | kijun discord post | 続ける |
| 9 | (処理済み音声の移動は transcribe が既に行う) | - |

仕様の表と違う点が1つある(docs/implementation-plan.md 12節に理由を残した)。仕様は Strata を起動するのを手順6だけに
しているが、手順1(read: 返信の振り分け)と手順8(post: 重要度の判定)も LLM を使う。手順6の終わりに
stop_command で Strata を止める以上、手順1と8の時点で Strata は動いていない。そのため、手順1と8も
Strata が要るときだけ(未処理の投稿がある・重要度が未判定の候補がある)start_command と stop_command で挟む。
start_command が空なら、どの手順でも Strata を起動も停止もしない(手で起動しておく運用)。
手順1と8は LLM が使えなくても動く(返信は原文だけ記録して未処理に残り、重要度は NULL で投稿される)ので、
Strata が起動しなくても、その手順は実行する。
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from kijun.batch.pending import (
    open_db_if_ready, pending_extraction, pending_match, pending_transcription,
)
from kijun.batch.runner import CommandRunner
from kijun.batch.strata import StrataWindow, WindowEvent
from kijun.config import Config
from kijun.discord.post import count_candidates_needing_importance
from kijun.discord.read import count_unprocessed
from kijun.table import format_table
from kijun.transport import HttpTransport

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# 画面と記録に出す、CLI の呼び出し方。実際の起動は現在の Python(sys.executable -m kijun.cli)で行う。
# タスクスケジューラの環境では uv が PATH に無いことがあり、同じ仮想環境の Python を直接使う方が確実なため。
DISPLAY_PREFIX = "uv run kijun"


@dataclass
class StepRecord:
    """バッチの1手順(または Strata の起動・停止)の記録。"""

    step: str  # 仕様 3-1 の手順番号。Strata の起動・停止は "6:start" のように付ける
    name: str
    command: str
    status: str
    exit_code: int | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    output_tail: str = ""
    note: str = ""

    @property
    def seconds(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class BatchResult:
    started_at: datetime
    ended_at: datetime | None = None
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(s.status == STATUS_FAILED for s in self.steps)

    def to_json(self) -> str:
        def default(o):
            if isinstance(o, datetime):
                return o.isoformat(timespec="seconds")
            raise TypeError(type(o))

        payload = {
            "started_at": self.started_at, "ended_at": self.ended_at, "ok": self.ok,
            "steps": [{**asdict(s), "seconds": s.seconds} for s in self.steps],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, default=default)


def config_args(config_path: Path | None) -> list[str]:
    return ["--config", str(config_path)] if config_path else []


def kijun_command(config_path: Path | None, *args: str) -> list[str]:
    """子プロセスとして起動するコマンド。現在の Python で kijun.cli を実行する。"""
    return [sys.executable, "-m", "kijun.cli", *config_args(config_path), *args]


def display_command(config_path: Path | None, *args: str) -> str:
    return " ".join([DISPLAY_PREFIX, *config_args(config_path), *args])


# --- --dry-run ---------------------------------------------------------------------


def plan_lines(cfg: Config, config_path: Path | None) -> list[str]:
    """--dry-run で出す、実行する順序と各コマンド。何も実行しない。"""
    d = lambda *a: display_command(config_path, *a)  # noqa: E731
    start = cfg.extract.strata.start_command or "(未設定: 何もしない)"
    stop = cfg.extract.strata.stop_command or "(未設定: 何もしない)"
    return [
        f"手順1    {d('discord', 'read')}",
        "         失敗しても続ける。未処理の投稿があり start_command が設定されていれば、前後で Strata を起動・停止する",
        f"手順2    {d('ingest')}",
        "         失敗したら止める(会議が登録できなければ以降が何もできない)",
        f"手順3-4  {d('transcribe', '--conversation', '<ID>')}",
        "         文字起こし前の会議ごとに実行する。失敗した会議だけ飛ばして次に進む",
        "手順5    (手順3-4のプロセスの終了で OS が GPU メモリを解放する)",
        f"手順6    {start}",
        f"         -> {d('extract', '--conversation', '<ID>')}  (抽出前の会議ごと。失敗した会議だけ飛ばす)",
        f"         -> {stop}  (手順6が失敗しても必ず実行する)",
        f"手順7    {d('match', '--conversation', '<ID>')}",
        "         未紐付けの抽出結果がある会議ごとに実行する。失敗した会議だけ飛ばす",
        f"手順8    {d('discord', 'post')}",
        "         失敗しても続ける。重要度が未判定の候補があり start_command が設定されていれば、前後で Strata を起動・停止する",
        "手順9    (処理済み音声を processed へ移す処理は、手順3-4の transcribe が既に行う)",
    ]


# --- 実行 ----------------------------------------------------------------------------


class _Batch:
    def __init__(
        self,
        cfg: Config,
        config_path: Path | None,
        runner: CommandRunner,
        transport: HttpTransport,
        clock: Callable[[], datetime],
        sleep: Callable[[float], None],
        monotonic: Callable[[], float],
        echo: Callable[[str], None],
    ) -> None:
        self.cfg, self.config_path, self.runner, self.transport = cfg, config_path, runner, transport
        self.clock, self.sleep, self.monotonic, self.echo = clock, sleep, monotonic, echo
        self.result = BatchResult(started_at=clock())

    # --- 記録 ---
    def _add(self, record: StepRecord) -> StepRecord:
        self.result.steps.append(record)
        mark = {"ok": "完了", "failed": "失敗", "skipped": "飛ばした"}[record.status]
        self.echo(f"[{mark}] 手順{record.step} {record.name}")
        return record

    def _skip(self, step: str, name: str, note: str) -> None:
        self._add(StepRecord(step, name, "", STATUS_SKIPPED, note=note))

    def _from_events(self, step: str, events: list[WindowEvent]) -> list[StepRecord]:
        label = {"start": "Strata の起動", "ready": "Strata の応答待ち", "stop": "Strata の停止"}
        return [
            StepRecord(
                f"{step}:{e.name}", label[e.name], e.command, STATUS_OK if e.ok else STATUS_FAILED,
                e.exit_code, e.started_at, e.ended_at, e.output_tail,
            )
            for e in events
        ]

    def _run_kijun(self, step: str, name: str, *args: str) -> bool:
        result = self.runner.run(kijun_command(self.config_path, *args))
        ok = result.exit_code == 0
        self._add(
            StepRecord(
                step, name, display_command(self.config_path, *args), STATUS_OK if ok else STATUS_FAILED,
                result.exit_code, result.started_at, result.ended_at, result.output_tail,
            )
        )
        return ok

    # --- Strata の窓 ---
    def _in_strata_window(self, step: str, body: Callable[[bool], None]) -> None:
        """Strata を起動して body(起動できたか)を実行し、必ず停止する。起動・停止の記録は body の前後に差し込む。"""
        window = StrataWindow(self.cfg, self.runner, self.transport, self.sleep, self.monotonic, self.clock)
        index = len(self.result.steps)
        with window as ok:
            started = list(window.events)
            body(ok)
        # 起動の記録は body の記録より前に、停止の記録は後ろに置く(実行した順に並べる)
        self.result.steps[index:index] = self._from_events(step, started)
        self.result.steps.extend(self._from_events(step, window.events[len(started):]))
        for r in [*self._from_events(step, window.events)]:
            if r.status == STATUS_FAILED:
                self.echo(f"[失敗] 手順{r.step} {r.name}")

    # --- 手順 ---
    def step1_read(self) -> None:
        with open_db_if_ready(self.cfg) as con:
            needs_llm = (
                con is not None
                and bool(self.cfg.extract.strata.start_command)
                and count_unprocessed(con) > 0
            )
        if needs_llm:
            self._in_strata_window("1", lambda _ok: self._run_kijun("1", "Discord の読み取り", "discord", "read"))
        else:
            self._run_kijun("1", "Discord の読み取り", "discord", "read")

    def step2_ingest(self) -> bool:
        return self._run_kijun("2", "inbox の取り込み", "ingest")

    def step3_transcribe(self) -> None:
        with open_db_if_ready(self.cfg) as con:
            targets = pending_transcription(con) if con else []
        if not targets:
            self._skip("3-4", "文字起こし", "文字起こし前の会議がありません")
        for cid in targets:
            self._run_kijun("3-4", f"文字起こし {cid}", "transcribe", "--conversation", cid)
        # 手順5: transcribe のプロセスが終了した時点で、OS が GPU メモリを解放している

    def step6_extract(self) -> None:
        with open_db_if_ready(self.cfg) as con:
            targets = pending_extraction(con, self.cfg) if con else []
        if not targets:
            self._skip("6", "抽出", "抽出前の会議がありません")
            return

        def body(ok: bool) -> None:
            for cid in targets:
                if not ok:
                    self._skip("6", f"抽出 {cid}", "Strata を起動できなかったため")
                    continue
                self._run_kijun("6", f"抽出 {cid}", "extract", "--conversation", cid)

        self._in_strata_window("6", body)

    def step7_match(self) -> None:
        with open_db_if_ready(self.cfg) as con:
            targets = pending_match(con, self.cfg) if con else []
        if not targets:
            self._skip("7", "照合", "照合する抽出結果がありません")
        for cid in targets:
            self._run_kijun("7", f"照合 {cid}", "match", "--conversation", cid)

    def step8_post(self) -> None:
        with open_db_if_ready(self.cfg) as con:
            needs_llm = (
                con is not None
                and bool(self.cfg.extract.strata.start_command)
                and bool(self.cfg.discord.review_channel_id)
                and count_candidates_needing_importance(con) > 0
            )
        if needs_llm:
            self._in_strata_window("8", lambda _ok: self._run_kijun("8", "Discord への投稿", "discord", "post"))
        else:
            self._run_kijun("8", "Discord への投稿", "discord", "post")

    def run(self) -> BatchResult:
        try:
            self.step1_read()
            if not self.step2_ingest():
                # 会議が登録できなければ、以降は何もできない
                for step, name in (("3-4", "文字起こし"), ("6", "抽出"), ("7", "照合"), ("8", "Discord への投稿")):
                    self._skip(step, name, "手順2(ingest)が失敗したため止めました")
                return self.result
            self.step3_transcribe()
            self.step6_extract()
            self.step7_match()
            self.step8_post()
            return self.result
        finally:
            self.result.ended_at = self.clock()


def run_batch(
    cfg: Config,
    config_path: Path | None,
    runner: CommandRunner,
    transport: HttpTransport,
    clock: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    echo: Callable[[str], None] = print,
) -> BatchResult:
    """バッチを実行する。1つの会議の失敗では止めず、手順2が失敗したときだけ止める。"""
    return _Batch(cfg, config_path, runner, transport, clock, sleep, monotonic, echo).run()


def write_log(cfg: Config, result: BatchResult) -> Path:
    """全手順の記録を batch.log_dir に <日時>.json として保存する(翌朝どの段で失敗したか分かるように)。"""
    cfg.batch.log_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.batch.log_dir / f"{result.started_at:%Y%m%d_%H%M%S}.json"
    path.write_text(result.to_json(), encoding="utf-8")
    return path


def format_summary(result: BatchResult) -> str:
    """全手順の成否の表。"""
    label = {STATUS_OK: "成功", STATUS_FAILED: "失敗", STATUS_SKIPPED: "飛ばした"}
    rows = [
        [
            s.step, s.name, label[s.status],
            "-" if s.exit_code is None else str(s.exit_code),
            "-" if s.seconds is None else f"{s.seconds:.0f}秒",
            s.note,
        ]
        for s in result.steps
    ]
    return format_table(["手順", "内容", "結果", "終了コード", "所要", "備考"], rows)
