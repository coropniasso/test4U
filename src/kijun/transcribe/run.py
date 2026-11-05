"""transcribe サブコマンドの本体（計画書 6-5）。

文字起こしと話者分離を実行し、utterances への書き込み、文字起こしの JSON の保存、
conversations の更新、音声ファイルの移動までを行う。
Transcriber と Diarizer は引数で受け取るので、テストでは偽の実装を渡せる。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.models import Conversation, Utterance
from kijun.transcribe.base import Diarizer, Transcriber
from kijun.transcribe.merge import DroppedSegment, filter_hallucinated_segments, merge_segments


@dataclass(frozen=True)
class TranscribeResult:
    """文字起こしの結果。utterances は DB に入れる発話、dropped は幻聴の疑いで除いた区間。"""

    utterances: list[Utterance]
    dropped: list[DroppedSegment] = field(default_factory=list)


def transcribe_to_utterances(
    audio_paths: list[Path],
    transcriber: Transcriber,
    diarizer: Diarizer,
    min_utterance_sec: float,
    max_utterance_sec: float = 0.0,
    drop_max_speaker_overlap: float = 0.0,
    drop_max_avg_logprob: float = -0.5,
) -> TranscribeResult:
    """音声から発話のリストを作る。DB にもファイルにも書かない。

    話者分離の後、話者の割り当てと結合の前に、幻聴の疑いがある区間を除く
    （drop_max_speaker_overlap が 0 なら除かない）。除いた区間は dropped に返す。
    """
    segments = transcriber.transcribe(audio_paths)
    turns = diarizer.diarize(audio_paths)
    kept, dropped = filter_hallucinated_segments(
        segments, turns, drop_max_speaker_overlap, drop_max_avg_logprob
    )
    utterances = merge_segments(kept, turns, min_utterance_sec, max_utterance_sec)
    return TranscribeResult(utterances, dropped)


def transcript_to_json(
    conversation: Conversation | None,
    source_files: list[str],
    transcribed_at: datetime,
    utterances: list[Utterance],
    dropped_segments: list[DroppedSegment] | None = None,
) -> str:
    """文字起こしの JSON 文字列を作る。

    DuckDB を壊したときに、`kijun restore-transcript` が conversations と utterances を復元できるよう、
    conversations の行を再構成するのに必要な項目をすべて入れる（構造は計画書 6-5）。
    日時は、DuckDB の TIMESTAMP と同じ精度（マイクロ秒）で往復できるよう、秒未満も含めて書く。
    conversation が None のとき（`transcribe --audio` の単発の試し実行）は、会議に属さないので
    conversation_id を null にする。この JSON は復元の対象にならない。
    dropped_segments は、幻聴の疑いで除いた区間（黙って消さず、利用者が確かめられるよう残す）。
    復元（restore.py）は読まないので、このキーが無い古い JSON も復元できる。
    """
    payload = {
        "conversation_id": conversation.conversation_id if conversation else None,
        "recorded_at": conversation.recorded_at.isoformat() if conversation else None,
        "setting_format": conversation.setting_format if conversation else None,
        "setting_kind": conversation.setting_kind if conversation else None,
        "source_files": source_files,
        "duration_sec": conversation.duration_sec if conversation else None,
        "created_at": conversation.created_at.isoformat() if conversation else None,
        "transcribed_at": transcribed_at.isoformat(),
        "utterances": [asdict(u) for u in utterances],
        "dropped_segments": [asdict(d) for d in (dropped_segments or [])],
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
) -> TranscribeResult:
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

    result = transcribe_to_utterances(
        audio_paths,
        transcriber,
        diarizer,
        cfg.transcribe.min_utterance_sec,
        cfg.transcribe.max_utterance_sec,
        cfg.transcribe.drop_max_speaker_overlap,
        cfg.transcribe.drop_max_avg_logprob,
    )
    utterances = result.utterances
    repo.replace_utterances(con, conversation_id, utterances)

    cfg.paths.transcripts.mkdir(parents=True, exist_ok=True)
    json_path = cfg.paths.transcripts / f"{conversation_id}.json"
    json_path.write_text(
        transcript_to_json(conv, conv.source_files, now, utterances, result.dropped),
        encoding="utf-8",
    )
    repo.mark_transcribed(con, conversation_id, str(json_path), conv.duration_sec, now)

    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    for p in audio_paths:
        if p.parent.resolve() != cfg.paths.processed.resolve():
            shutil.move(str(p), str(cfg.paths.processed / p.name))
    return result


def transcribe_single_audio(
    audio_path: Path,
    out_json: Path,
    transcriber: Transcriber,
    diarizer: Diarizer,
    min_utterance_sec: float,
    max_utterance_sec: float = 0.0,
    now: datetime | None = None,
    drop_max_speaker_overlap: float = 0.0,
    drop_max_avg_logprob: float = -0.5,
) -> TranscribeResult:
    """単発の試し実行（`transcribe --audio`）。DuckDB には書かず、JSON だけを保存する。音声は移動しない。"""
    now = now or datetime.now()
    result = transcribe_to_utterances(
        [audio_path],
        transcriber,
        diarizer,
        min_utterance_sec,
        max_utterance_sec,
        drop_max_speaker_overlap,
        drop_max_avg_logprob,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        transcript_to_json(None, [audio_path.name], now, result.utterances, result.dropped),
        encoding="utf-8",
    )
    return result
