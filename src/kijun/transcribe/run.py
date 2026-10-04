"""transcribe サブコマンドの本体（計画書 6-5）。

文字起こしと話者分離を実行し、utterances への書き込み、文字起こしの JSON の保存、
conversations の更新、音声ファイルの移動までを行う。
Transcriber と Diarizer は引数で受け取るので、テストでは偽の実装を渡せる。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.models import Utterance
from kijun.transcribe.base import Diarizer, Transcriber
from kijun.transcribe.merge import merge_segments


def transcribe_to_utterances(
    audio_paths: list[Path],
    transcriber: Transcriber,
    diarizer: Diarizer,
    min_utterance_sec: float,
) -> list[Utterance]:
    """音声から Utterance のリストを作る。DB にもファイルにも書かない。"""
    segments = transcriber.transcribe(audio_paths)
    turns = diarizer.diarize(audio_paths)
    return merge_segments(segments, turns, min_utterance_sec)


def utterances_to_json(
    conversation_id: str | None,
    source_files: list[str],
    duration_sec: float | None,
    transcribed_at: datetime,
    utterances: list[Utterance],
) -> str:
    """文字起こしの JSON 文字列を作る。DuckDB を壊したときに文字起こしをやり直さずに済むよう保存する。"""
    payload = {
        "conversation_id": conversation_id,
        "source_files": source_files,
        "duration_sec": duration_sec,
        "transcribed_at": transcribed_at.isoformat(timespec="seconds"),
        "utterances": [asdict(u) for u in utterances],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def find_audio_file(cfg: Config, name: str) -> Path:
    """inbox、無ければ processed から音声ファイルを探す。どちらにも無ければ FileNotFoundError。"""
    for base in (cfg.paths.inbox, cfg.paths.processed):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"音声ファイルが見つかりません: {name}（探した場所: {cfg.paths.inbox}, {cfg.paths.processed}）"
    )


def transcribe_conversation(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    conversation_id: str,
    transcriber: Transcriber,
    diarizer: Diarizer,
    now: datetime | None = None,
) -> list[Utterance]:
    """登録済みの会議を文字起こしする。

    手順: 音声を探す → 文字起こしと話者分離 → utterances に書く → JSON を保存 →
    conversations を更新 → 音声を processed に移す。
    音声の移動は最後に行う。途中で失敗しても、録音は inbox に残る。
    """
    conv = repo.get_conversation(con, conversation_id)
    if conv is None:
        raise LookupError(
            f"会議 {conversation_id} が conversations にありません。先に `kijun ingest` を実行してください。"
        )
    now = now or datetime.now()
    audio_paths = [find_audio_file(cfg, n) for n in conv.source_files]

    utterances = transcribe_to_utterances(
        audio_paths, transcriber, diarizer, cfg.transcribe.min_utterance_sec
    )
    repo.replace_utterances(con, conversation_id, utterances)

    cfg.paths.transcripts.mkdir(parents=True, exist_ok=True)
    json_path = cfg.paths.transcripts / f"{conversation_id}.json"
    json_path.write_text(
        utterances_to_json(conversation_id, conv.source_files, conv.duration_sec, now, utterances),
        encoding="utf-8",
    )
    repo.mark_transcribed(con, conversation_id, str(json_path), conv.duration_sec, now)

    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    for p in audio_paths:
        if p.parent.resolve() != cfg.paths.processed.resolve():
            shutil.move(str(p), str(cfg.paths.processed / p.name))
    return utterances


def transcribe_single_audio(
    audio_path: Path,
    out_json: Path,
    transcriber: Transcriber,
    diarizer: Diarizer,
    min_utterance_sec: float,
    now: datetime | None = None,
) -> list[Utterance]:
    """単発の試し実行（`transcribe --audio`）。DuckDB には書かず、JSON だけを保存する。音声は移動しない。"""
    now = now or datetime.now()
    utterances = transcribe_to_utterances(
        [audio_path], transcriber, diarizer, min_utterance_sec
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        utterances_to_json(None, [audio_path.name], None, now, utterances), encoding="utf-8"
    )
    return utterances
