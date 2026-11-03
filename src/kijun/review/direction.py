"""speaker_map から extracted_requests.direction を導出する(仕様 2-8)。

抽出の時点では話者ラベル(SPEAKER_00 など)しか分からないので、direction は NULL で保存してある
(docs/implementation-plan.md 4-1)。話者と人物の対応(speaker_map)が埋まった会議について、ここで導出する。
direction が決まらないと、CLAUDE.md 5章の「direction = 相手→自分 の抽出結果は、相手別の要求一覧として扱う」ができず、
自分が言ったことと言われたことを分けられない。

規則(speaker_map の person_label が self の話者を「自分」とする):
- 要求した側(speaker_from_label)が自分で、言われた側が自分でない -> self_to_other
- 言われた側(speaker_to_label)が自分で、要求した側が自分でない  -> other_to_self
- どちらも自分でない(第三者間のやり取り)                       -> NULL
- 両方とも自分(独り言、または話者分離の誤り)                    -> NULL
speaker_map に無い話者ラベルと、話者ラベルが NULL の側は、自分でないものとして扱う。

self が1つも決まっていない会議は導出しない(既存の direction にも触らない)。
変更は review_log に記録しない。review_log はユーザーのレビューの履歴を残す表であり、機械的な導出は混ぜない。
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from kijun.db import repo


def compute_direction(
    speaker_from_label: str | None, speaker_to_label: str | None, speaker_map: dict[str, str]
) -> str | None:
    """1件の抽出結果の direction。決められなければ None。"""
    from_is_self = speaker_from_label is not None and speaker_map.get(speaker_from_label) == repo.PERSON_SELF
    to_is_self = speaker_to_label is not None and speaker_map.get(speaker_to_label) == repo.PERSON_SELF
    if from_is_self and not to_is_self:
        return repo.DIRECTION_SELF_TO_OTHER
    if to_is_self and not from_is_self:
        return repo.DIRECTION_OTHER_TO_SELF
    return None


@dataclass(frozen=True)
class DirectionResult:
    conversation_id: str
    derived: bool  # self が決まっていて導出した
    total: int = 0
    self_to_other: int = 0
    other_to_self: int = 0
    undetermined: int = 0  # NULL のままになった件数(第三者間、両方 self、言われた側が不明など)
    note: str = ""


def derive_directions(con: duckdb.DuckDBPyConnection, conversation_id: str) -> DirectionResult:
    """会議の全モデルの抽出結果について direction を導出して書き込む。

    導出の結果が NULL でも書き込む(speaker_map を直した後の再実行で、以前の値が残らないようにするため)。
    """
    speaker_map = repo.get_speaker_map(con, conversation_id)
    if repo.PERSON_SELF not in speaker_map.values():
        return DirectionResult(
            conversation_id, derived=False, note="speaker_map に self がありません。direction を導出しません"
        )
    counts = {repo.DIRECTION_SELF_TO_OTHER: 0, repo.DIRECTION_OTHER_TO_SELF: 0, None: 0}
    requests = repo.get_requests(con, conversation_id)
    for r in requests:
        direction = compute_direction(r.speaker_from_label, r.speaker_to_label, speaker_map)
        counts[direction] += 1
        if direction != r.direction:
            repo.update_request_direction(con, r.request_id, direction)
    return DirectionResult(
        conversation_id,
        derived=True,
        total=len(requests),
        self_to_other=counts[repo.DIRECTION_SELF_TO_OTHER],
        other_to_self=counts[repo.DIRECTION_OTHER_TO_SELF],
        undetermined=counts[None],
    )


def conversations_with_speaker_map(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute("SELECT DISTINCT conversation_id FROM speaker_map ORDER BY conversation_id").fetchall()
    return [r[0] for r in rows]
