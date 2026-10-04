"""extract サブコマンドの本体（計画書 7-7）。

会議の発話をチャンクに分け、Extractor で抽出し、重複を除き、extracted_requests に書く。
Extractor は引数で受け取るので、テストでは FakeExtractor を渡せる。
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.extract.base import ExtractionError, Extractor
from kijun.extract.chunk import MergeOutcome, merge_chunk_results, split_into_chunks
from kijun.extract.models import ExtractedRequest


@dataclass(frozen=True)
class ChunkFailure:
    """抽出に失敗したチャンク。seq は会議全体の番号。"""

    chunk_index: int
    seq_from: int
    seq_to: int
    reason: str


@dataclass
class ExtractionRun:
    model_name: str
    chunk_count: int
    failures: list[ChunkFailure] = field(default_factory=list)
    outcome: MergeOutcome = field(default_factory=lambda: MergeOutcome(items=[]))
    saved_count: int = 0


def extract_log_path(cfg: Config, conversation_id: str, model_name: str) -> Path:
    """失敗したチャンクの記録の保存先。

    失敗したチャンクは DuckDB のスキーマに列が無いので、DB と同じディレクトリの
    extract_logs/ に JSON で残す。比較表（compare）がここを読む。
    """
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model_name)
    return cfg.paths.db.parent / "extract_logs" / f"{conversation_id}__{safe_model}.json"


def write_extract_log(path: Path, conversation_id: str, run: ExtractionRun, at: datetime) -> None:
    payload = {
        "conversation_id": conversation_id,
        "model_name": run.model_name,
        "extracted_at": at.isoformat(timespec="seconds"),
        "chunk_count": run.chunk_count,
        "failed_chunks": [
            {"chunk_index": f.chunk_index, "seq_from": f.seq_from, "seq_to": f.seq_to, "reason": f.reason}
            for f in run.failures
        ],
        "invalid_count": run.outcome.invalid_count,
        "overlap_dropped_count": run.outcome.overlap_dropped_count,
        "duplicate_count": run.outcome.duplicate_count,
        "saved_count": run.saved_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_failed_chunk_count(path: Path) -> int | None:
    """記録から失敗したチャンク数を読む。記録が無ければ None。"""
    if not path.exists():
        return None
    return len(json.loads(path.read_text(encoding="utf-8")).get("failed_chunks", []))


def _to_row(
    item: ExtractedRequest, conversation_id: str, model_name: str, number: int, at: datetime
) -> repo.RequestRow:
    return repo.RequestRow(
        request_id=f"{conversation_id}#{model_name}#{number}",
        conversation_id=conversation_id,
        seq_from=item.seq_from,
        seq_to=item.seq_to,
        speaker_from_label=item.speaker_from_label,
        speaker_to_label=item.speaker_to_label,
        # speaker_map が埋まるまで方向は確定できないので NULL にする（計画書 4-1）
        direction=None,
        counterpart_role=item.counterpart_role,
        artifact_type=item.artifact_type,
        quote=item.quote,
        request_summary=item.request_summary,
        stated_reason=item.stated_reason,
        applies_when=item.applies_when,
        reusable=item.reusable,
        pass_criterion=item.pass_criterion,
        source_type=repo.SOURCE_LIVE_CONVERSATION,
        model_name=model_name,
        extracted_at=at,
    )


def run_extraction(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    conversation_id: str,
    extractor: Extractor,
    now: datetime | None = None,
    log: Callable[[str], None] | None = None,
) -> ExtractionRun:
    """会議を抽出して extracted_requests に保存する。

    - 失敗したチャンクは記録して次に進む。会議全体は落とさない。失敗したチャンクの
      発話の範囲は、標準エラーに出す。
    - 同じ会議・同じモデルの抽出結果が既にある場合は、置き換える。
      ただし、その結果が item_evidence に紐付いている場合は、根拠の参照先が無くなるので中止する。
    - 全チャンクが失敗した場合は、既存の結果を消さずに例外にする。
    """
    now = now or datetime.now()
    log = log or (lambda m: print(m, file=sys.stderr))

    utterances = repo.get_utterances(con, conversation_id)
    if not utterances:
        raise LookupError(
            f"会議 {conversation_id} の発話がありません。先に `kijun transcribe` を実行してください。"
        )

    chunks = split_into_chunks(
        utterances, cfg.extract.chunk_minutes, cfg.extract.overlap_minutes
    )
    run = ExtractionRun(model_name=extractor.model_name, chunk_count=len(chunks))
    results: list[tuple] = []
    for chunk in chunks:
        try:
            result = extractor.extract(chunk)
            results.append((chunk, result.items))
        except ExtractionError as e:
            run.failures.append(ChunkFailure(chunk.index, chunk.seq_from, chunk.seq_to, str(e)))
            log(
                f"失敗: チャンク {chunk.index + 1}/{len(chunks)}（発話 {chunk.seq_from}〜{chunk.seq_to}）: {e}"
            )
            results.append((chunk, None))

    if len(run.failures) == len(chunks):
        write_extract_log(extract_log_path(cfg, conversation_id, run.model_name), conversation_id, run, now)
        raise ExtractionError(f"すべてのチャンク（{len(chunks)} 個）の抽出に失敗しました")

    run.outcome = merge_chunk_results(results)

    linked = repo.count_linked_requests(con, conversation_id, run.model_name)
    if linked:
        raise RuntimeError(
            f"会議 {conversation_id}・モデル {run.model_name} の抽出結果のうち {linked} 件が "
            "item_evidence に紐付いているため、置き換えられません。"
        )
    repo.delete_requests(con, conversation_id, run.model_name)
    rows = [
        _to_row(item, conversation_id, run.model_name, i, now)
        for i, item in enumerate(run.outcome.items, start=1)
    ]
    repo.insert_requests(con, rows)
    run.saved_count = len(rows)
    write_extract_log(extract_log_path(cfg, conversation_id, run.model_name), conversation_id, run, now)
    return run
