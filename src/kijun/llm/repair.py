"""壊れた JSON を修復する純粋な関数群。

Ollama は format にスキーマを渡せば出力をスキーマに適合させるが、適合の保証は版とモデルに依存する。
保証が効かない場合、モデルの出力には次のような壊れ方が混じる。

- 前置きや後書きの散文（「はい、まとめました。」）が JSON の前後に付く
- コードフェンス（```json ... ```）で囲まれる

確認できない背面の挙動を、確認できる場所（外部に接続しない純粋な関数）に寄せるのがこの
モジュールの目的である。

出力が num_predict に達して途中で切れた場合は、修復しない。議事録の JSON は節（topics、decisions、
todos、open_issues）が並ぶ1つのオブジェクトで、途中で閉じると後ろの節が黙って空になるからである
（「失うより止める」。呼び出し側が done_reason = length を見てエラーにする）。

行った修復の内容は RepairOutcome.actions に英数字の識別子で入れ、画面表示のときに日本語にする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ACTION_STRIPPED_WHITESPACE = "stripped_whitespace"
ACTION_REMOVED_CODE_FENCE = "removed_code_fence"
ACTION_REMOVED_PROSE_BEFORE = "removed_prose_before"
ACTION_REMOVED_PROSE_AFTER = "removed_prose_after"
ACTION_NO_JSON_FOUND = "no_json_found"

ACTION_DESCRIPTIONS = {
    ACTION_STRIPPED_WHITESPACE: "前後の空白を除去した",
    ACTION_REMOVED_CODE_FENCE: "コードフェンスの中身だけを取り出した",
    ACTION_REMOVED_PROSE_BEFORE: "JSON より前の散文を捨てた",
    ACTION_REMOVED_PROSE_AFTER: "JSON より後の散文を捨てた",
    ACTION_NO_JSON_FOUND: "JSON の始まり（{）が見つからなかった",
}


class UnrepairableJsonError(ValueError):
    """修復しても指定のモデルとして読めなかったときの例外。"""


@dataclass(frozen=True)
class RepairOutcome:
    """修復の結果。

    text:    修復後の文字列。修復できなかった場合は、途中までの最善の結果が入る。
    actions: 行った修復の識別子（ACTION_*）。何もしなければ空。
    """

    text: str
    actions: list[str] = field(default_factory=list)

    @property
    def descriptions(self) -> list[str]:
        """actions を日本語の説明にして返す（画面表示用）。"""
        return [ACTION_DESCRIPTIONS.get(a, a) for a in self.actions]


def _strip_code_fence(text: str) -> str:
    """コードフェンス（```json ... ``` または ``` ... ```）があれば中身だけを返す。

    閉じるフェンスが無い場合（出力が途中で切れた場合）も、開きフェンスより後を返す。
    """
    if not text.startswith("```"):
        return text
    after_open = text.find("\n")
    if after_open == -1:
        return text[3:]
    body = text[after_open + 1 :]
    close = body.rfind("```")
    return body[:close] if close != -1 else body


def _scan_object_end(text: str, start: int) -> int | None:
    """text[start]（"{"）から始まるオブジェクトが閉じる位置の次を返す。閉じていなければ None。

    文字列リテラルの中の括弧は数えない。
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            if in_string:
                escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def repair_json_text(text: str) -> RepairOutcome:
    """モデルの出力を、JSON のオブジェクトとして読める形に近づける。

    例外は出さない。修復できなかった場合も、途中までの最善の結果を返す
    （読めるかどうかの判断は parse_model_with_repair が行う）。

    次の順に行う。
    1. 前後の空白を除去する。
    2. コードフェンスがあれば中身だけを取り出す。
    3. 最初の { より前の散文を捨てる。
    4. オブジェクトが閉じていれば、その後の散文を捨てる。
    """
    actions: list[str] = []
    stripped = text.strip()
    if stripped != text:
        actions.append(ACTION_STRIPPED_WHITESPACE)

    unfenced = _strip_code_fence(stripped)
    if unfenced != stripped:
        actions.append(ACTION_REMOVED_CODE_FENCE)
        stripped = unfenced.strip()

    start = stripped.find("{")
    if start == -1:
        return RepairOutcome(stripped, [*actions, ACTION_NO_JSON_FOUND])
    if start > 0:
        actions.append(ACTION_REMOVED_PROSE_BEFORE)
        stripped = stripped[start:]

    end = _scan_object_end(stripped, 0)
    if end is None:
        return RepairOutcome(stripped, actions)  # 閉じていない。検証が失敗する
    if end < len(stripped.rstrip()):
        actions.append(ACTION_REMOVED_PROSE_AFTER)
    return RepairOutcome(stripped[:end], actions)


ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_model_with_repair(text: str, model_cls: type[ModelT]) -> tuple[ModelT, RepairOutcome]:
    """モデルの出力を修復してから model_cls として検証する。読めなければ UnrepairableJsonError を出す。

    LLM の出力から pydantic のモデルを作る経路は、この関数だけにする。修復を通さずに検証してしまう
    経路が残らないようにするためである（背面がスキーマへの適合を保証するかどうかに関わらず）。
    """
    outcome = repair_json_text(text)
    try:
        result = model_cls.model_validate_json(outcome.text)
    except ValidationError as e:
        head = outcome.text[:200].replace("\n", " ")
        raise UnrepairableJsonError(
            f"修復しても {model_cls.__name__} として読めませんでした"
            f"（行った修復: {', '.join(outcome.descriptions) or 'なし'}）: {head}: {e}"
        ) from e
    return result, outcome
