"""夜間バッチが、どの会議に何の処理が要るかを DB から調べる(読み取りだけ)。

子プロセスが DB を書き換えるので、各段の直前に調べ直す。子プロセスが動いている間は、この接続を開かない
(DuckDB は同時に複数のプロセスから書き込めない)。
"""

from __future__ import annotations

import json
from contextlib import closing, contextmanager
from collections.abc import Iterator

import duckdb

from kijun.config import Config
from kijun.db import migrate, repo
from kijun.extract.run import extract_log_path


@contextmanager
def open_db_if_ready(cfg: Config) -> Iterator[duckdb.DuckDBPyConnection | None]:
    """スキーマ適用済みの DB に接続する。DB が無い・スキーマが未適用なら None を渡す。"""
    if not cfg.paths.db.exists():
        yield None
        return
    with closing(repo.connect(cfg.paths.db)) as con:
        yield con if migrate.current_version(con) is not None else None


def pending_transcription(con: duckdb.DuckDBPyConnection) -> list[str]:
    """登録済みで、まだ文字起こしが済んでいない会議(録音の古い順)。"""
    return [c.conversation_id for c in repo.list_conversations(con) if c.transcribed_at is None]


def _extraction_done(con: duckdb.DuckDBPyConnection, cfg: Config, conversation_id: str, model_name: str) -> bool:
    """この会議・このモデルの抽出が済んでいるか。

    抽出結果の行があれば済み。0件だった会議は行が無いので、extract_logs の記録で見分ける
    (毎晩 0件の会議を抽出し直して、GPU の時間を無駄にしないため)。記録があっても、
    すべてのチャンクが失敗していた場合は済みとみなさず、再試行する。一部のチャンクだけ失敗した会議は、
    自動では再試行しない(同じチャンクが毎晩失敗し続けるのを避けるため。失敗は抽出の出力と記録に残る)。
    """
    if repo.get_requests(con, conversation_id, model_name):
        return True
    path = extract_log_path(cfg, conversation_id, model_name)
    if not path.exists():
        return False
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
        return len(log.get("failed_chunks", [])) < int(log.get("chunk_count", 0))
    except (OSError, ValueError):
        return False


def pending_extraction(con: duckdb.DuckDBPyConnection, cfg: Config) -> list[str]:
    model = cfg.extract.model_name()
    return [
        c.conversation_id
        for c in repo.list_conversations(con)
        if c.transcribed_at is not None and not _extraction_done(con, cfg, c.conversation_id, model)
    ]


def pending_match(con: duckdb.DuckDBPyConnection, cfg: Config) -> list[str]:
    """reusable な抽出結果のうち、まだ item_evidence に紐付いていないものがある会議。

    照合は埋め込みモデルを読み込むので、対象が無い会議では起動しない。
    """
    rows = con.execute(
        "SELECT c.conversation_id FROM conversations c WHERE EXISTS ("
        "  SELECT 1 FROM extracted_requests r WHERE r.conversation_id = c.conversation_id "
        "  AND r.model_name = ? AND r.reusable "
        "  AND NOT EXISTS (SELECT 1 FROM item_evidence e WHERE e.request_id = r.request_id)"
        ") ORDER BY c.recorded_at",
        [cfg.extract.model_name()],
    ).fetchall()
    return [r[0] for r in rows]
