"""重要度の3段階判定(仕様 2-5、CLAUDE.md 4-6)。

初期は LLM が 1〜3 で判定して根拠を1文添え、ユーザーがリアクションで上書きする。
2〜3か月後、上書き履歴を目的変数にして、ルールの重みを回帰で推定し直す。

上書きの履歴の形を今決めておく理由: 後から形を変えると、それまでの履歴が分析に使えなくなる。
- 判定したとき: knowledge_items.importance = LLM の値。review_log に
  importance_proposed(after = 根拠の1文)。
- ユーザーが上書きしたとき: review_log に importance_overridden(before = LLM の値、after = ユーザーの値)。
  1行から、目的変数(ユーザーの値)と LLM の判定値の両方が取れる。
回帰の説明変数の候補(直近30日の出現回数、成果物の種類、同じ指摘をした相手の人数)は
extracted_requests と item_evidence から後から計算できるので、列は増やさない。

判定に失敗したら importance は NULL のままにする。処理は止めない。
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Protocol

import duckdb
from pydantic import BaseModel, Field

from kijun.db import repo
from kijun.db.repo import KnowledgeItemRow
from kijun.review.llm import JsonChat, ReviewLlmError

# 根拠として LLM に見せる発言の最大件数(仕様 2-5)
MAX_EVIDENCE_QUOTES = 3


class ImportanceVerdict(BaseModel):
    """重要度の判定1件。"""

    importance: int = Field(ge=1, le=3, description="重要度。1(低)から3(高)の整数。")
    rationale: str = Field(min_length=1, description="その重要度にした根拠を述べた1文。")


class ImportanceJudge(Protocol):
    def judge(self, check_text: str, quotes: list[str]) -> ImportanceVerdict:
        """チェック項目の文言と根拠の発言から重要度を判定する。失敗したら ReviewLlmError を出す。"""
        ...


SYSTEM_PROMPT = """\
あなたは、会話から抜き出したチェック項目の重要度を3段階で判定する係です。
チェック項目は、話し手本人が他の人の成果物をレビューしたときの指摘から作られています。
判定結果は話し手本人が見て、リアクションで書き換えます。あなたの判定は初期案です。

# 重要度の基準
- 3: 守られていないと、成果物の目的が果たせない(読み手が必要な行動を取れない、誤解を招く、事実や数値を誤る、など)。
- 2: 守られている方が明らかに良いが、守られていなくても目的は果たせる。
- 1: 好みや細かい体裁の指摘で、守られていなくても影響が小さい。

# 規則
- 判定の材料は、渡されたチェック項目と発言だけです。発言に無い理由や状況を推測して加えないでください。
- 根拠は1文で、上の基準のどれに当たるかが分かるように書いてください。

# 出力形式
JSON だけを返してください。前置き、説明、コードフェンス(```)は付けません。
形は {"importance": 整数(1から3), "rationale": "根拠の1文"} のオブジェクトです。
"""


def build_user_message(check_text: str, quotes: list[str]) -> str:
    lines = [f"チェック項目: {check_text}", "", "根拠になった発言:"]
    lines += [f"- {q}" for q in quotes] or ["- (記録なし)"]
    return "\n".join(lines)


class LlmImportanceJudge:
    def __init__(self, chat: JsonChat) -> None:
        self._chat = chat

    def judge(self, check_text: str, quotes: list[str]) -> ImportanceVerdict:
        return self._chat.ask(
            SYSTEM_PROMPT, build_user_message(check_text, quotes), ImportanceVerdict, "importance"
        )


def propose_importance(
    con: duckdb.DuckDBPyConnection,
    judge: ImportanceJudge | None,
    item: KnowledgeItemRow,
    at: datetime,
) -> int | None:
    """項目の重要度を LLM に判定させて保存する。判定した値を返す。

    judge が None、または判定に失敗したときは、importance を NULL のまま None を返す(処理は止めない)。
    """
    if judge is None:
        return None
    evidence = repo.get_item_evidence_requests(con, item.item_id, MAX_EVIDENCE_QUOTES)
    try:
        verdict = judge.judge(item.check_text, [r.quote for r in evidence])
    except ReviewLlmError as e:
        print(f"警告: 項目 {item.item_id} の重要度を判定できませんでした(NULL のままにします): {e}", file=sys.stderr)
        return None
    repo.update_item_importance(con, item.item_id, verdict.importance, at)
    repo.insert_review_log(
        con, item.item_id, repo.ACTION_IMPORTANCE_PROPOSED, None, verdict.rationale, at
    )
    return verdict.importance
