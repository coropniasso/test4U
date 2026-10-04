"""文字起こしの JSON から、DuckDB の conversations と utterances を復元する（計画書 6-5）。

transcribe が [paths].transcripts に保存した JSON は、DuckDB を壊したときに文字起こしをやり直さずに
済むようにするためのもの。この処理が、その JSON を読み戻す側である。

復元の対象は conversations と utterances だけ。extracted_requests、knowledge_items、item_evidence は
JSON に入っていないので復元できない。復元後に `kijun extract` と `kijun match` をやり直す。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.models import Conversation, Utterance

# conversations の行を再構成するのに必要な、JSON の最上位の項目
# （conversations.transcript_path は、読み込んだ JSON ファイル自身のパスなので JSON には入れない）
REQUIRED_KEYS = (
    "conversation_id",
    "recorded_at",
    "setting_format",
    "setting_kind",
    "source_files",
    "duration_sec",
    "created_at",
    "transcribed_at",
    "utterances",
)
UTTERANCE_KEYS = ("seq", "speaker_label", "start_sec", "end_sec", "text")

STATUS_RESTORED = "restored"  # 新しく復元した
STATUS_OVERWRITTEN = "overwritten"  # --force で上書きした
STATUS_SKIPPED_EXISTS = "skipped_exists"  # conversations に同じ会議が既にあるのでスキップした
STATUS_SKIPPED_SINGLE = "skipped_single_audio"  # 会議に属さない単発の試し実行の JSON
STATUS_ERROR = "error"  # JSON の形が不正


class TranscriptFormatError(ValueError):
    """JSON の項目が欠けている、または値が不正なときの例外。ファイル名と項目名を含める。"""


@dataclass(frozen=True)
class RestoreOutcome:
    path: Path
    conversation_id: str | None
    status: str
    utterance_count: int = 0
    message: str = ""


def parse_transcript(path: Path) -> tuple[Conversation, list[Utterance]] | None:
    """JSON を読んで (conversations の行, 発話のリスト) にする。

    単発の試し実行の JSON（conversation_id が null）は None を返す。
    項目が欠けている、または値が不正なら TranscriptFormatError を出す。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TranscriptFormatError(f"{path.name}: JSON として読めません: {e}") from e
    if not isinstance(data, dict):
        raise TranscriptFormatError(f"{path.name}: 最上位が JSON のオブジェクトではありません")

    if "conversation_id" in data and data["conversation_id"] is None:
        return None
    missing = [k for k in REQUIRED_KEYS if k not in data]
    cid = data.get("conversation_id")
    if missing:
        raise TranscriptFormatError(
            f"{path.name}（会議ID: {cid}）: 必要な項目がありません: {', '.join(missing)}"
        )

    try:
        utterances = []
        for i, u in enumerate(data["utterances"]):
            lacking = [k for k in UTTERANCE_KEYS if k not in u]
            if lacking:
                raise TranscriptFormatError(
                    f"{path.name}（会議ID: {cid}）: utterances[{i}] に必要な項目がありません: {', '.join(lacking)}"
                )
            utterances.append(
                Utterance(
                    seq=int(u["seq"]),
                    speaker_label=u["speaker_label"],
                    start_sec=float(u["start_sec"]),
                    end_sec=float(u["end_sec"]),
                    text=str(u["text"]),
                )
            )
        conversation = Conversation(
            conversation_id=str(cid),
            recorded_at=datetime.fromisoformat(data["recorded_at"]),
            setting_format=str(data["setting_format"]),
            setting_kind=str(data["setting_kind"]),
            source_files=[str(f) for f in data["source_files"]],
            transcript_path=str(path),
            duration_sec=None if data["duration_sec"] is None else float(data["duration_sec"]),
            transcribed_at=datetime.fromisoformat(data["transcribed_at"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )
    except TranscriptFormatError:
        raise
    except (TypeError, ValueError) as e:
        raise TranscriptFormatError(f"{path.name}（会議ID: {cid}）: 値が不正です: {e}") from e
    return conversation, utterances


def restore_transcript(
    con: duckdb.DuckDBPyConnection, path: Path, force: bool = False, dry_run: bool = False
) -> RestoreOutcome:
    """JSON 1本から会議を復元する。

    - conversations に同じ conversation_id の行が既にあれば、上書きせずスキップする
      （既にある正しい行を JSON で踏み潰さないため）。スキップしたときは utterances にも触らない。
      force が True のときだけ上書きする。
    - utterances は、その会議の行を消してから JSON の内容を入れ直す。
    - dry_run が True のときは、DB に何も書かず、復元される場合の結果だけを返す。
    """
    try:
        parsed = parse_transcript(path)
    except TranscriptFormatError as e:
        return RestoreOutcome(path, None, STATUS_ERROR, message=str(e))
    if parsed is None:
        return RestoreOutcome(
            path, None, STATUS_SKIPPED_SINGLE,
            message="単発の試し実行（transcribe --audio）の JSON なので、会議には復元できません",
        )
    conversation, utterances = parsed
    exists = repo.get_conversation(con, conversation.conversation_id) is not None
    if exists and not force:
        return RestoreOutcome(
            path, conversation.conversation_id, STATUS_SKIPPED_EXISTS,
            message="conversations に同じ会議が既にあるのでスキップしました（上書きするには --force）",
        )
    status = STATUS_OVERWRITTEN if exists else STATUS_RESTORED
    if not dry_run:
        repo.replace_conversation(con, conversation)
        repo.replace_utterances(con, conversation.conversation_id, utterances)
    return RestoreOutcome(path, conversation.conversation_id, status, utterance_count=len(utterances))


def restore_transcripts(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    conversation_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[RestoreOutcome]:
    """conversation_id を指定すればその会議の JSON を、None なら [paths].transcripts の JSON をすべて復元する。"""
    directory = cfg.paths.transcripts
    if conversation_id is not None:
        path = directory / f"{conversation_id}.json"
        if not path.exists():
            raise LookupError(f"文字起こしの JSON が見つかりません: {path}")
        paths = [path]
    else:
        if not directory.exists():
            raise LookupError(f"文字起こしの JSON のディレクトリが見つかりません: {directory}")
        paths = sorted(directory.glob("*.json"))
    return [restore_transcript(con, p, force=force, dry_run=dry_run) for p in paths]
