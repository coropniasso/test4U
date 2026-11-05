"""kijun のサブコマンド。

GPU・ネットワークに触る実装（faster-whisper、pyannote、Ollama、discord.py）は、
サブコマンドの実行時に初めて import する。`kijun --help` は、これらが入っていなくても動く。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from kijun import ollama_api
from kijun.config import Config, ConfigError, load_config
from kijun.db import migrate, repo
from kijun.deps import MissingDependencyError

app = typer.Typer(
    help="会話の録音から議事録を作る仕組み。",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="DuckDB の操作。", no_args_is_help=True)
app.add_typer(db_app, name="db")
discord_app = typer.Typer(
    help="Discord への議事録の投稿(bot は常駐しない。夜間バッチが post を起動する)。",
    no_args_is_help=True,
)
app.add_typer(discord_app, name="discord")
batch_app = typer.Typer(
    help="夜間バッチ(CLAUDE.md 夜間バッチ)。既存の CLI を別プロセスとして順に起動する。",
    no_args_is_help=True,
)
app.add_typer(batch_app, name="batch")


ConversationOpt = Annotated[
    str, typer.Option("--conversation", help="会議ID（例: 20260920_1930_f2f_teirei）")
]
@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="設定ファイルのパス。省略すると ./config.toml（無ければ既定値）。"),
    ] = None,
) -> None:
    ctx.obj = config


def _fail(message: str) -> typer.Exit:
    typer.echo(f"エラー: {message}", err=True)
    return typer.Exit(code=1)


@contextmanager
def _errors() -> Iterator[None]:
    """想定している失敗（設定の誤り、依存の不足、対象が無い等）を、トレースバックなしで報告する。"""
    try:
        yield
    except typer.Exit:
        raise  # typer.Exit は RuntimeError の子クラスなので、先に通す
    except (ConfigError, MissingDependencyError, LookupError, FileNotFoundError, RuntimeError) as e:
        raise _fail(str(e)) from e


def _config(ctx: typer.Context) -> Config:
    return load_config(ctx.obj)


def _open_db(cfg: Config):
    """DB に接続する。スキーマが未適用なら、`kijun db init` を促して終了する。"""
    con = repo.connect(cfg.paths.db)
    if migrate.ensure_current(con) is None:
        con.close()
        raise _fail("DB のスキーマが未適用です。先に `uv run kijun db init` を実行してください。")
    return con


def _build_transcriber(cfg: Config):
    if cfg.transcribe.backend != "faster_whisper":
        raise _fail(f"未対応の transcribe.backend です: {cfg.transcribe.backend}")
    from kijun.transcribe.faster_whisper_impl import FasterWhisperTranscriber

    return FasterWhisperTranscriber(cfg.transcribe)


def _build_diarizer(cfg: Config):
    if cfg.diarize.backend != "pyannote":
        raise _fail(f"未対応の diarize.backend です: {cfg.diarize.backend}")
    from kijun.transcribe.pyannote_impl import PyannoteDiarizer

    return PyannoteDiarizer(cfg.diarize)


# --- db ----------------------------------------------------------------------


@db_app.command("init")
def db_init(ctx: typer.Context) -> None:
    """スキーマを適用する（何度実行しても安全）。"""
    with _errors():
        cfg = _config(ctx)
        con = repo.connect(cfg.paths.db)
        version = migrate.apply_schema(con)
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE' ORDER BY table_name"
            ).fetchall()
        ]
        con.close()
        typer.echo(f"DB: {cfg.paths.db}（スキーマバージョン {version}）")
        typer.echo(f"テーブル {len(tables)} 個: {', '.join(tables)}")


# --- ingest ------------------------------------------------------------------


@app.command()
def ingest(
    ctx: typer.Context,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="DB に登録せず、束ねた結果を表で出す。")] = False,
) -> None:
    """inbox を走査し、会議単位に束ねて conversations に登録する。"""
    from kijun.ingest.run import collect_bundles, format_bundles, register_bundles

    with _errors():
        cfg = _config(ctx)
        result = collect_bundles(cfg)
        if result.failed:
            typer.echo("解析できなかったファイル（削除も移動もしていません。ファイル名を直してください）:")
            for name, reason in result.failed:
                typer.echo(f"  {name}: {reason}")
            typer.echo("")
        if not result.bundles:
            typer.echo(f"処理対象の音声ファイルがありません: {cfg.paths.inbox}")
            return
        typer.echo(format_bundles(result.bundles))
        if dry_run:
            typer.echo("\n--dry-run のため、DB には登録していません。")
            return
        con = _open_db(cfg)
        register_bundles(con, result)
        con.close()
        typer.echo(
            f"\n新規登録 {len(result.registered)} 件、更新 {len(result.updated)} 件、"
            f"文字起こし済みのため対象外 {len(result.skipped)} 件。"
        )


# --- transcribe --------------------------------------------------------------


def _echo_dropped(dropped: list) -> None:
    """幻聴の疑いで除いた区間の件数と内容を表示する。内容は文字起こしの JSON の dropped_segments にも残る。"""
    if not dropped:
        return
    typer.echo("")
    typer.echo(f"幻聴の疑いで除いた区間: {len(dropped)} 件（JSON の dropped_segments に残してあります）")
    for d in dropped:
        typer.echo(f"  {d.start_sec:8.1f}s  {d.text}  [{d.reason}]")


@app.command()
def transcribe(
    ctx: typer.Context,
    conversation: Annotated[
        str | None, typer.Option("--conversation", help="登録済みの会議ID。utterances に書く。")
    ] = None,
    audio: Annotated[
        Path | None,
        typer.Option("--audio", help="単発の試し実行。DB には書かず、標準出力と JSON に出す。"),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="--audio のときの JSON の保存先。省略すると transcripts 配下。")
    ] = None,
) -> None:
    """文字起こしと話者分離を行う。"""
    from kijun.transcribe.run import transcribe_conversation, transcribe_single_audio

    if (conversation is None) == (audio is None):
        raise _fail("--conversation と --audio のどちらか一方を指定してください。")
    with _errors():
        cfg = _config(ctx)
        started = time.monotonic()
        if audio is not None:
            if not audio.exists():
                raise _fail(f"音声ファイルが見つかりません: {audio}")
            transcriber = _build_transcriber(cfg)
            diarizer = _build_diarizer(cfg)
            out_path = out or (cfg.paths.transcripts / f"{audio.stem}.json")
            result = transcribe_single_audio(
                audio,
                out_path,
                transcriber,
                diarizer,
                cfg.transcribe.min_utterance_sec,
                cfg.transcribe.max_utterance_sec,
                drop_max_speaker_overlap=cfg.transcribe.drop_max_speaker_overlap,
                drop_max_avg_logprob=cfg.transcribe.drop_max_avg_logprob,
            )
            for u in result.utterances:
                typer.echo(
                    f"[{u.seq}] {u.start_sec:8.1f}-{u.end_sec:8.1f} {u.speaker_label or 'UNKNOWN'}: {u.text}"
                )
            _echo_dropped(result.dropped)
            typer.echo(f"\nJSON を保存しました: {out_path}")
        else:
            assert conversation is not None
            con = _open_db(cfg)
            transcriber = _build_transcriber(cfg)
            diarizer = _build_diarizer(cfg)
            result = transcribe_conversation(con, cfg, conversation, transcriber, diarizer)
            con.close()
            typer.echo(f"{conversation}: 発話 {len(result.utterances)} 件を保存しました。")
            _echo_dropped(result.dropped)
        elapsed = time.monotonic() - started
        typer.echo(
            f"処理時間: {elapsed:.1f} 秒（モデルの読み込みを含む）。"
            "VRAM の使用量は、実行中に別の端末で `nvidia-smi` を実行して記録してください。"
        )


# --- restore-transcript ------------------------------------------------------


@app.command("restore-transcript")
def restore_transcript_cmd(
    ctx: typer.Context,
    conversation: Annotated[
        str | None, typer.Option("--conversation", help="復元する会議ID。")
    ] = None,
    all_: Annotated[
        bool, typer.Option("--all", help="[paths].transcripts の JSON をすべて復元する。")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="conversations に同じ会議が既にあっても上書きする。")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="DB に書かず、復元される内容だけを出す。")
    ] = False,
) -> None:
    """文字起こしの JSON から、conversations と utterances を復元する。"""
    from kijun.transcribe.restore import (
        STATUS_ERROR,
        STATUS_OVERWRITTEN,
        STATUS_RESTORED,
        restore_transcripts,
    )

    if (conversation is None) == (not all_):
        raise _fail("--conversation と --all のどちらか一方を指定してください。")
    with _errors():
        cfg = _config(ctx)
        con = _open_db(cfg)
        outcomes = restore_transcripts(con, cfg, conversation, force=force, dry_run=dry_run)
        con.close()
        done = 0
        for o in outcomes:
            label = o.conversation_id or o.path.name
            if o.status in (STATUS_RESTORED, STATUS_OVERWRITTEN):
                done += 1
                verb = "復元します" if dry_run else "復元しました"
                if o.status == STATUS_OVERWRITTEN:
                    verb = "上書きします" if dry_run else "上書きしました"
                typer.echo(f"{label}: {verb}（発話 {o.utterance_count} 件）")
            else:
                typer.echo(f"{label}: {o.message}", err=o.status == STATUS_ERROR)
        suffix = "（--dry-run のため、DB には書いていません）" if dry_run else ""
        typer.echo(f"\n復元 {done} 件 / 対象 {len(outcomes)} 件{suffix}。")
        typer.echo("minutes（議事録）は復元されません。必要なら `kijun minutes --conversation <ID>` をやり直してください。")
        if any(o.status == STATUS_ERROR for o in outcomes):
            raise typer.Exit(code=1)


# --- minutes -----------------------------------------------------------------


@app.command()
def minutes(
    ctx: typer.Context,
    conversation: Annotated[
        str | None,
        typer.Option("--conversation", help="議事録を作る（作り直す）会議ID。文字起こし済みであること。"),
    ] = None,
    pending: Annotated[
        bool, typer.Option("--pending", help="議事録の無い、文字起こし済みの会議すべてについて作る。")
    ] = False,
) -> None:
    """文字起こしから議事録（Markdown）を作り、minutes テーブルに記録する。"""
    from kijun.llm.chat import OllamaJsonChat
    from kijun.minutes.generate import MinutesError, generate_minutes

    if (conversation is None) == (not pending):
        raise _fail("--conversation と --pending のどちらか一方を指定してください。")
    with _errors():
        cfg = _config(ctx)
        con = _open_db(cfg)
        try:
            ids = [conversation] if conversation else [
                c.conversation_id for c in repo.list_conversations_without_minutes(con)
            ]
            if not ids:
                typer.echo("議事録を作る対象の会議がありません（文字起こし済みで、議事録の無い会議）。")
                return
            chat = OllamaJsonChat(cfg.llm)
            failed = 0
            for cid in ids:
                started = time.monotonic()
                try:
                    run = generate_minutes(con, cfg, cid, chat)
                except MinutesError as e:
                    failed += 1
                    typer.echo(f"{cid}: 議事録を作れませんでした: {e}", err=True)
                    continue
                typer.echo(
                    f"{cid}: 議事録を保存しました: {run.path}"
                    f"（モデル {run.model_name}、チャンク {run.chunk_count} 個、{time.monotonic() - started:.0f} 秒）"
                )
        finally:
            con.close()
        if failed:
            raise typer.Exit(code=1)


# --- doctor ------------------------------------------------------------------


@app.command()
def doctor(
    ctx: typer.Context,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="結果の JSON の保存先。省略すると data/doctor/ollama_<日時>.json。"),
    ] = None,
) -> None:
    """推論サーバー（Ollama）を実測し、GPU に収まっているかと config.toml に書くべき値を出す。"""
    from kijun.doctor import STATUS_ERROR, format_report, report_path, run_doctor
    from kijun.transport import HttpxTransport

    with _errors():
        cfg = _config(ctx)
        # DB は使わないので、`db init` の前でも実行できる。
        report = run_doctor(HttpxTransport(), cfg)
        typer.echo(format_report(report))
        path = out or report_path(cfg, report.started_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.to_json(), encoding="utf-8")
        typer.echo(f"\n結果を保存しました: {path}")
        if any(c.status == STATUS_ERROR for c in report.checks):
            raise typer.Exit(code=1)


# --- discord -----------------------------------------------------------------


@discord_app.command("post")
def discord_post(ctx: typer.Context) -> None:
    """未投稿の議事録を Discord に投稿する(夜間バッチの手順4)。"""
    from kijun.discord.run import post_command

    with _errors():
        cfg = _config(ctx)
        con = _open_db(cfg)
        try:
            code = post_command(con, cfg, echo=typer.echo)
        finally:
            con.close()
        if code:
            raise typer.Exit(code=code)


# --- batch -------------------------------------------------------------------


@batch_app.command("run")
def batch_run(
    ctx: typer.Context,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="実行する順序と各コマンドを表示するだけで、何も実行しない。")
    ] = False,
) -> None:
    """夜間バッチを実行する(ingest -> transcribe -> minutes -> discord post)。"""
    from kijun.batch.run import format_summary, plan_lines, run_batch, write_log
    from kijun.batch.runner import SubprocessRunner

    with _errors():
        cfg = _config(ctx)
        if dry_run:
            for line in plan_lines(cfg, ctx.obj):
                typer.echo(line)
            typer.echo("\n--dry-run のため、何も実行していません。")
            return
        transport = None
        if ollama_api.keeps_model_loaded(cfg.llm):
            from kijun.transport import HttpxTransport

            transport = HttpxTransport()  # keep_alive が "0" 以外のとき、モデルを下ろす要求に使う
        result = run_batch(cfg, ctx.obj, SubprocessRunner(), echo=typer.echo, transport=transport)
        log_path = write_log(cfg, result)
        typer.echo("\n" + format_summary(result))
        typer.echo(f"\n記録を保存しました: {log_path}")
        if not result.ok:
            raise typer.Exit(code=1)


@batch_app.command("install-task")
def batch_install_task(
    ctx: typer.Context,
    time: Annotated[
        str | None,
        typer.Option("--time", help="毎日実行する時刻(HH:MM)。省略すると設定の batch.task_time。"),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="XML の保存先。省略すると ./kijun-nightly-task.xml。")
    ] = None,
) -> None:
    """Windows のタスクスケジューラに登録するための XML を作り、登録コマンドを案内する(登録はしない)。"""
    import shutil

    from kijun.batch.task_xml import TASK_NAME, build_task_xml, schtasks_command

    with _errors():
        cfg = _config(ctx)  # 設定ファイルの誤りを先に報告する
        time = time or cfg.batch.task_time
        config_args = f'--config "{Path(ctx.obj).resolve()}" ' if ctx.obj else ""
        uv = shutil.which("uv") or "uv"
        try:
            xml = build_task_xml(time, uv, f"run kijun {config_args}batch run", Path.cwd())
        except ValueError as e:
            raise _fail(str(e)) from e
        path = (out or Path("kijun-nightly-task.xml")).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(xml)
        typer.echo(f"タスクスケジューラ用の XML を保存しました: {path}")
        typer.echo(f"  毎日 {time} に、{Path.cwd()} で `{uv} run kijun {config_args}batch run` を実行します。")
        typer.echo("  スリープ中でも起動し(WakeToRun)、電源が切れていた場合は次の起動時に実行します(StartWhenAvailable)。")
        typer.echo("\n登録するには、PowerShell(管理者権限が必要なことがあります)で次を実行してください。")
        typer.echo(f"  {schtasks_command(path)}")
        typer.echo(f'同じ名前({TASK_NAME})のタスクを置き換えるときは、末尾に /f を付けます。')
        if uv == "uv":
            typer.echo("注意: uv のフルパスが見つからなかったので `uv` と書きました。タスクスケジューラの PATH に uv が無い場合は、XML の Command を uv.exe のフルパスに書き換えてください。")
        typer.echo("注意: ユーザーがログオンしている間(画面ロックを含む)に実行します。ログオフ中は実行されません。")


# --- purge-audio -------------------------------------------------------------


@app.command("purge-audio")
def purge_audio(
    ctx: typer.Context,
    execute: Annotated[
        bool, typer.Option("--execute", help="実際に削除する。付けなければ一覧を出すだけ。")
    ] = False,
) -> None:
    """文字起こし完了から保存期間が過ぎた音声を削除する（既定は一覧の表示のみ）。"""
    from kijun.retention.purge import find_purge_candidates, format_candidates, purge_candidates

    with _errors():
        cfg = _config(ctx)
        con = _open_db(cfg)
        candidates = find_purge_candidates(con, cfg, datetime.now())
        con.close()
        if not candidates:
            typer.echo(f"削除対象の音声はありません（保存期間 {cfg.retention.audio_days} 日）。")
            return
        typer.echo(format_candidates(candidates))
        if not execute:
            typer.echo(f"\n{len(candidates)} 件が対象です。削除するには --execute を付けて再実行してください。")
            return
        deleted, failed = purge_candidates(candidates)
        for c in deleted:
            typer.echo(f"削除しました: {c.path}")
        for c, reason in failed:
            typer.echo(f"削除できませんでした: {c.path}: {reason}", err=True)
        typer.echo(f"\n削除 {len(deleted)} 件、失敗 {len(failed)} 件。")
        if failed:
            raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
