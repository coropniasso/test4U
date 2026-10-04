"""kijun のサブコマンド（計画書 第12節）。

GPU・ネットワークに触る実装（faster-whisper、pyannote、Ollama、Claude API、sentence-transformers）は、
サブコマンドの実行時に初めて import する。`kijun --help` は、これらが入っていなくても動く。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from kijun.config import Config, ConfigError, load_config
from kijun.db import migrate, repo
from kijun.deps import MissingDependencyError
from kijun.table import format_table

app = typer.Typer(
    help="会話録音から判断基準を抽出し、チェックリスト化する仕組み。",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="DuckDB の操作。", no_args_is_help=True)
app.add_typer(db_app, name="db")


class Provider(str, Enum):
    ollama = "ollama"
    claude = "claude"


ConversationOpt = Annotated[
    str, typer.Option("--conversation", help="会議ID（例: 20260920_1930_f2f_teirei）")
]
ProviderOpt = Annotated[
    Provider | None,
    typer.Option("--provider", help="抽出に使うモデルの提供元。省略すると設定の extract.provider。"),
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
    if migrate.current_version(con) is None:
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


def _build_extractor(cfg: Config, provider: Provider):
    if provider == Provider.ollama:
        from kijun.extract.ollama_impl import OllamaExtractor

        return OllamaExtractor(cfg)
    from kijun.extract.claude_impl import ClaudeExtractor

    return ClaudeExtractor(cfg)


def _build_embedder(cfg: Config):
    if cfg.match.backend != "e5":
        raise _fail(f"未対応の match.backend です: {cfg.match.backend}")
    from kijun.match.e5_impl import E5Embedder

    return E5Embedder(cfg.match)


def _provider_model(cfg: Config, provider: Provider | None) -> tuple[Provider, str]:
    p = provider or Provider(cfg.extract.provider)
    model = cfg.extract.ollama.model if p == Provider.ollama else cfg.extract.claude.model
    return p, model


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
            utterances = transcribe_single_audio(
                audio, out_path, transcriber, diarizer, cfg.transcribe.min_utterance_sec
            )
            for u in utterances:
                typer.echo(
                    f"[{u.seq}] {u.start_sec:8.1f}-{u.end_sec:8.1f} {u.speaker_label or 'UNKNOWN'}: {u.text}"
                )
            typer.echo(f"\nJSON を保存しました: {out_path}")
        else:
            assert conversation is not None
            con = _open_db(cfg)
            transcriber = _build_transcriber(cfg)
            diarizer = _build_diarizer(cfg)
            utterances = transcribe_conversation(con, cfg, conversation, transcriber, diarizer)
            con.close()
            typer.echo(f"{conversation}: 発話 {len(utterances)} 件を保存しました。")
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
        typer.echo(
            "extracted_requests、knowledge_items、item_evidence は復元されません。"
            "必要なら `kijun extract` と `kijun match` をやり直してください。"
        )
        if any(o.status == STATUS_ERROR for o in outcomes):
            raise typer.Exit(code=1)


# --- extract -----------------------------------------------------------------


@app.command()
def extract(ctx: typer.Context, conversation: ConversationOpt, provider: ProviderOpt = None) -> None:
    """発話から要求・指摘を抽出し、extracted_requests に書く。"""
    from kijun.extract.run import run_extraction

    with _errors():
        cfg = _config(ctx)
        prov, _model = _provider_model(cfg, provider)
        con = _open_db(cfg)
        extractor = _build_extractor(cfg, prov)
        run = run_extraction(con, cfg, conversation, extractor)
        con.close()
        o = run.outcome
        typer.echo(
            f"{conversation}: モデル {run.model_name} で {run.saved_count} 件を保存しました"
            f"（チャンク {run.chunk_count} 個、失敗 {len(run.failures)} 個）。"
        )
        typer.echo(
            f"捨てた件数: 番号が範囲外 {o.invalid_count}、重複区間 {o.overlap_dropped_count}、"
            f"quote の重複 {o.duplicate_count}"
        )
        for f in run.failures:
            typer.echo(f"失敗したチャンク: 発話 {f.seq_from}〜{f.seq_to}: {f.reason}", err=True)


# --- compare -----------------------------------------------------------------


@app.command()
def compare(
    ctx: typer.Context,
    conversation: ConversationOpt,
    out: Annotated[Path | None, typer.Option("--out", help="Markdown の保存先。省略すると data/compare/<ID>.md。")] = None,
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help="比較するモデル名。2 回指定する。省略すると設定の Qwen と Claude。"),
    ] = None,
) -> None:
    """Qwen と Claude の抽出結果の対照表（Markdown）を出す。"""
    from kijun.compare.report import build_report

    with _errors():
        cfg = _config(ctx)
        con = _open_db(cfg)
        report = build_report(con, cfg, conversation, model)
        con.close()
        out_path = out or (cfg.paths.db.parent / "compare" / f"{conversation}.md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        typer.echo(f"対照表を保存しました: {out_path}")


# --- match -------------------------------------------------------------------


@app.command()
def match(
    ctx: typer.Context,
    conversation: ConversationOpt,
    provider: ProviderOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="DB に書かず、最類似の既存項目と類似度を表で出す。")
    ] = False,
) -> None:
    """抽出結果を既存のチェック項目と照合する。"""
    from kijun.match.link import link_requests

    with _errors():
        cfg = _config(ctx)
        _prov, model_name = _provider_model(cfg, provider)
        con = _open_db(cfg)
        embedder = _build_embedder(cfg)
        results = link_requests(con, cfg, conversation, embedder, model_name=model_name, dry_run=dry_run)
        con.close()
        if not results:
            typer.echo("照合の対象がありません（reusable = true で、未紐付けの抽出結果が無い）。")
            return
        rows = [
            [
                r.request_id,
                "-" if r.similarity is None else f"{r.similarity:.3f}",
                r.best_item_id or "-",
                "既存項目に紐付け" if r.action == "linked" else "新規候補",
                r.request_summary,
            ]
            for r in results
        ]
        typer.echo(
            format_table(["抽出結果", "類似度", "最類似の既存項目", "結果", "要約"], rows)
        )
        typer.echo(f"\n閾値: {cfg.match.similarity_threshold}")
        if dry_run:
            typer.echo("--dry-run のため、DB には書いていません。")


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
