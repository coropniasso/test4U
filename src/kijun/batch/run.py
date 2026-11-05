"""夜間バッチの本体(CLAUDE.md 夜間バッチ)。

既存の CLI を subprocess で別プロセスとして順に起動する。ここには処理の中身を書かない
(原則2-2。1つの Python プロセスの中で全部を呼ぶと faster-whisper と LLM が同時に VRAM を使う危険が出る)。

| 手順 | 実行するもの | 失敗したとき |
|---|---|---|
| 1 | kijun ingest | **止める** |
| 2 | kijun transcribe --conversation <ID>(会議ごと。話者分離を含む) | その会議だけ飛ばす |
| 3 | kijun minutes --conversation <ID>(会議ごと) | その会議だけ飛ばす |
| 4 | kijun discord post | 続ける(最後の手順) |

処理済み音声の processed への移動は、手順2の transcribe が行う。手順2のプロセスが終了すると OS が
GPU メモリを解放するので、手順3の LLM と VRAM を取り合わない。

LLM は Ollama で、設定 llm.keep_alive が "0" なら応答のたびに VRAM から下ろされる。したがって手順2の時点で
LLM は VRAM に載っておらず、バッチは LLM の起動も停止もしない。keep_alive を "0" 以外にした場合は、
手順3の後に、モデルを下ろす要求(POST /api/generate、keep_alive 0)を送る。

GPU を使う手順(2、3)の直前には、ゲームが動いていないかを確認する(batch/defer.py)。動いていれば待ち、
待った時間が batch.defer_max_hours を超えたら、残りの GPU を使う手順をすべて飛ばして終える。
飛ばした会議は、翌晩に batch/pending.py が未処理として拾う。手順4(Discord への投稿)は GPU を使わない。
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from kijun.batch.defer import GameDeferrer, GateResult, ProcessLister
from kijun.batch.pending import open_db_if_ready, pending_minutes, pending_transcription
from kijun.batch.runner import CommandRunner
from kijun import ollama_api
from kijun.config import Config
from kijun.table import format_table
from kijun.transport import HttpTransport, HttpxTransport

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# 画面と記録に出す、CLI の呼び出し方。実際の起動は現在の Python(sys.executable -m kijun.cli)で行う。
# タスクスケジューラの環境では uv が PATH に無いことがあり、同じ仮想環境の Python を直接使う方が確実なため。
DISPLAY_PREFIX = "uv run kijun"


@dataclass
class StepRecord:
    """バッチの1手順(またはゲームによる待機)の記録。"""

    step: str  # 仕様 3-1 の手順番号。ゲームによる待機は "3-4:defer" のように付ける
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
    watched = ", ".join(cfg.batch.defer_if_running)
    if watched:
        gate = (
            f"GPU を使う手順の直前に、{watched} が動いていないかを確認する。動いていれば "
            f"{cfg.batch.defer_check_interval_min} 分ごとに確認して待つ(合計 {cfg.batch.defer_max_hours:g} 時間まで。"
            "超えたら残りの GPU を使う手順を飛ばす)"
        )
    else:
        gate = "ゲームの検出は行わない(batch.defer_if_running が空)"
    return [
        f"開始時刻  {cfg.batch.task_time}(kijun batch install-task の --time を省略したときの値)",
        f"ゲーム    {gate}",
        f"手順1    {d('ingest')}",
        "         失敗したら止める(会議が登録できなければ以降が何もできない)",
        f"手順2    {d('transcribe', '--conversation', '<ID>')}",
        "         文字起こし前の会議ごとに、ゲームの確認を挟んで実行する(話者分離を含む)。失敗した会議だけ飛ばして次に進む。"
        "処理済みの音声の processed への移動もこの手順が行う。プロセスの終了で OS が GPU メモリを解放する",
        f"手順3    {d('minutes', '--conversation', '<ID>')}",
        "         議事録の無い、文字起こし済みの会議ごとに、ゲームの確認を挟んで実行する。失敗した会議だけ飛ばす。"
        f"LLM は Ollama(keep_alive = {cfg.llm.keep_alive!r})で、バッチは起動も停止もしない"
        + (
            '。keep_alive が "0" ではないので、手順3の後にモデルを下ろす要求を送る'
            if ollama_api.keeps_model_loaded(cfg.llm)
            else "。応答のたびに VRAM から下りるので、アンロードの要求は送らない"
        ),
        f"手順4    {d('discord', 'post')}",
        "         未投稿の議事録を Discord に投稿する。失敗しても、記録して終わる(GPU を使わないので、ゲームの確認は挟まない)",
    ]


# --- 実行 ----------------------------------------------------------------------------


class _Batch:
    def __init__(
        self,
        cfg: Config,
        config_path: Path | None,
        runner: CommandRunner,
        deferrer: GameDeferrer,
        clock: Callable[[], datetime],
        echo: Callable[[str], None],
        transport: HttpTransport | None,
    ) -> None:
        self.cfg, self.config_path, self.runner, self.deferrer = cfg, config_path, runner, deferrer
        self.clock, self.echo, self.transport = clock, echo, transport
        self.result = BatchResult(started_at=clock())

    # --- 記録 ---
    def _add(self, record: StepRecord) -> StepRecord:
        self.result.steps.append(record)
        mark = {"ok": "完了", "failed": "失敗", "skipped": "飛ばした"}[record.status]
        self.echo(f"[{mark}] 手順{record.step} {record.name}")
        return record

    def _skip(self, step: str, name: str, note: str) -> None:
        self._add(StepRecord(step, name, "", STATUS_SKIPPED, note=note))

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

    # --- LLM のアンロード ---
    def _unload_llm(self, step: str) -> None:
        """keep_alive が "0" 以外のときだけ、モデルを VRAM から下ろす要求を送る。

        "0" なら Ollama が応答のたびに下ろすので、何もしない。下ろせなくても処理は止めない
        (失敗として記録する。Ollama が動いていなければ、そもそも読み込まれていない)。
        """
        ollama = self.cfg.llm
        if not ollama_api.keeps_model_loaded(ollama):
            return
        transport = self.transport or HttpxTransport()
        started = self.clock()
        ok, detail = ollama_api.unload_model(transport, ollama)
        self._add(
            StepRecord(
                f"{step}:unload", "LLM のアンロード", ollama_api.url_of(ollama, ollama_api.GENERATE_PATH),
                STATUS_OK if ok else STATUS_FAILED, started_at=started, ended_at=self.clock(), note=detail,
            )
        )

    # --- ゲームによる延期 ---
    def _gate(self, step: str, name: str) -> bool:
        """GPU を使う手順の直前の確認。始めてよければ True。上限に達していれば飛ばした記録を残して False。

        待ったときは、待った時間と原因のプロセス名を StepRecord("<手順>:defer")に残す。
        """
        result: GateResult = self.deferrer.gate()
        if result.waited_sec > 0:
            names = "、".join(result.blocked_by)
            note = f"{names} が動いていたため {result.waited_sec / 60:.0f} 分待った"
            if not result.proceed:
                note += "(延期の上限に達した)"
            self._add(
                StepRecord(
                    f"{step}:defer", "ゲームの実行中のため待機", "", STATUS_OK,
                    started_at=result.started_at, ended_at=result.ended_at, note=note,
                )
            )
        if not result.proceed:
            names = "、".join(result.blocked_by)
            self._skip(step, name, f"ゲームの実行中のため延期の上限に達した({names})")
        return result.proceed

    # --- 手順 ---
    def step1_ingest(self) -> bool:
        return self._run_kijun("1", "inbox の取り込み", "ingest")

    def step2_transcribe(self) -> None:
        with open_db_if_ready(self.cfg) as con:
            targets = pending_transcription(con) if con else []
        if not targets:
            self._skip("2", "文字起こし", "文字起こし前の会議がありません")
        for cid in targets:
            if self._gate("2", f"文字起こし {cid}"):
                self._run_kijun("2", f"文字起こし {cid}", "transcribe", "--conversation", cid)
        # transcribe のプロセスが終了した時点で、OS が GPU メモリを解放している

    def step3_minutes(self) -> None:
        with open_db_if_ready(self.cfg) as con:
            targets = pending_minutes(con) if con else []
        if not targets:
            self._skip("3", "議事録", "議事録の無い、文字起こし済みの会議がありません")
        for cid in targets:
            if self._gate("3", f"議事録 {cid}"):
                self._run_kijun("3", f"議事録 {cid}", "minutes", "--conversation", cid)
        if targets:
            self._unload_llm("3")

    def step4_post(self) -> None:
        self._run_kijun("4", "Discord への投稿", "discord", "post")

    def run(self) -> BatchResult:
        try:
            if not self.step1_ingest():
                # 会議が登録できなければ、以降は何もできない
                for step, name in (("2", "文字起こし"), ("3", "議事録"), ("4", "Discord への投稿")):
                    self._skip(step, name, "手順1(ingest)が失敗したため止めました")
                return self.result
            self.step2_transcribe()
            self.step3_minutes()
            self.step4_post()
            return self.result
        finally:
            self.result.ended_at = self.clock()


def run_batch(
    cfg: Config,
    config_path: Path | None,
    runner: CommandRunner,
    lister: ProcessLister | None = None,
    clock: Callable[[], datetime] = datetime.now,
    sleep: Callable[[float], None] = time.sleep,
    echo: Callable[[str], None] = print,
    transport: HttpTransport | None = None,
) -> BatchResult:
    """バッチを実行する。1つの会議の失敗では止めず、手順1が失敗したときだけ止める。

    lister はゲームの検出(batch.defer_if_running が空なら使われない)。sleep は延期中の待機で、
    テストでは実時間を待たない関数を渡す。transport はモデルを下ろす要求に使い、
    llm.keep_alive が "0" なら使われない。
    """
    deferrer = GameDeferrer(cfg.batch, lister, sleep, clock, warn=echo)
    return _Batch(cfg, config_path, runner, deferrer, clock, echo, transport).run()


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
