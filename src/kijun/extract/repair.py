"""壊れた JSON を修復する純粋な関数群（計画書 7-3）。

この仕組みが使う推論サーバー（Ollama）が、JSON スキーマへの適合を保証するかどうかは
確認できていない。保証しない場合、モデルの出力には次のような壊れ方が混じる。

- 前置きの散文（「はい、抽出しました。」）が JSON の前に付く
- コードフェンス（```json ... ```）で囲まれる
- トップレベルが配列になる（プロンプトが「JSON 配列のみ」と指示しているため、素直に従うとこうなる）
- 出力の上限（num_predict）に達して末尾が途中で切れる

確認できない背面の挙動を、確認できる場所（外部に接続しない純粋な関数）に寄せるのがこの
モジュールの目的である。ここは外部に一切接続しないので、クラウド環境で完全にテストできる。

行った修復の内容は RepairOutcome.actions に英数字の識別子で入れ、画面表示のときに日本語にする
（CLAUDE.md 4-2 の「英数字で持ち、表示のときに日本語にする」方針を、記録の値にも適用する）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from kijun.extract.models import ExtractionResult

# RepairOutcome.actions に入る識別子と、その日本語の説明
ACTION_STRIPPED_WHITESPACE = "stripped_whitespace"
ACTION_REMOVED_CODE_FENCE = "removed_code_fence"
ACTION_REMOVED_PROSE_BEFORE = "removed_prose_before"
ACTION_REMOVED_PROSE_AFTER = "removed_prose_after"
ACTION_WRAPPED_ARRAY_IN_ITEMS = "wrapped_array_in_items"
ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM = "truncated_to_last_complete_item"
ACTION_NO_JSON_FOUND = "no_json_found"

ACTION_DESCRIPTIONS = {
    ACTION_STRIPPED_WHITESPACE: "前後の空白を除去した",
    ACTION_REMOVED_CODE_FENCE: "コードフェンスの中身だけを取り出した",
    ACTION_REMOVED_PROSE_BEFORE: "JSON より前の散文を捨てた",
    ACTION_REMOVED_PROSE_AFTER: "JSON より後の散文を捨てた",
    ACTION_WRAPPED_ARRAY_IN_ITEMS: 'トップレベルの配列を {"items": [...]} に包んだ',
    ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM: "末尾が切れていたので、最後の完全な要素までで閉じた",
    ACTION_NO_JSON_FOUND: "JSON の始まり（{ または [）が見つからなかった",
}


class UnrepairableJsonError(ValueError):
    """修復しても ExtractionResult として読めなかったときの例外。"""


@dataclass(frozen=True)
class RepairOutcome:
    """修復の結果。

    text:            修復後の文字列。修復できなかった場合は、途中までの最善の結果が入る。
    actions:         行った修復の識別子（ACTION_* ）。何もしなければ空。
    truncated_items: 末尾が切れていたために捨てた要素の数。
    """

    text: str
    actions: list[str] = field(default_factory=list)
    truncated_items: int = 0

    @property
    def descriptions(self) -> list[str]:
        """actions を日本語の説明にして返す（画面表示用）。"""
        return [ACTION_DESCRIPTIONS.get(a, a) for a in self.actions]


@dataclass(frozen=True)
class ParsedExtraction:
    """修復を通して検証した結果と、その過程の記録。"""

    result: ExtractionResult
    repair: RepairOutcome


def _strip_code_fence(text: str) -> str:
    """コードフェンス（```json ... ``` または ``` ... ```）があれば中身だけを返す。

    閉じるフェンスが無い場合（出力が途中で切れた場合）も、開きフェンスより後を返す。
    """
    if not text.startswith("```"):
        return text
    after_open = text.find("\n")
    if after_open == -1:
        # ``` だけで改行が無い。言語名の直後に JSON が続く場合に備えて ``` を外す
        return text[3:]
    body = text[after_open + 1 :]
    close = body.rfind("```")
    return body[:close] if close != -1 else body


def _first_bracket(text: str) -> int | None:
    """最初の { または [ の位置。どちらも無ければ None。"""
    candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    return min(candidates) if candidates else None


def _scan_value_end(text: str, start: int) -> int | None:
    """text[start] から始まるオブジェクトまたは配列が閉じる位置の次を返す。

    閉じていなければ（出力が途中で切れていれば）None。
    文字列リテラルの中の括弧は数えない。
    """
    if start >= len(text) or text[start] not in "{[":
        return None
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
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _complete_element_ends(text: str, array_open: int) -> list[int]:
    """配列（text[array_open] が "["）の中で、深さ1の要素が閉じる位置の次を並べて返す。

    要素がオブジェクトまたは文字列の場合に数える。この仕組みの items の要素は
    オブジェクトなので、それで足りる。配列自体が閉じたらそこで打ち切る。
    """
    ends: list[int] = []
    depth = 0
    in_string = False
    escape = False
    for i in range(array_open, len(text)):
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
            if not in_string and depth == 1:
                ends.append(i + 1)
            continue
        if in_string:
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 1:
                ends.append(i + 1)
            elif depth == 0:
                return ends
    return ends


def _find_items_array(text: str) -> int | None:
    """オブジェクトの中の "items" の値である配列の、"[" の位置を返す。見つからなければ None。"""
    key = text.find('"items"')
    if key == -1:
        return None
    open_bracket = text.find("[", key)
    return open_bracket if open_bracket != -1 else None


def _repair_truncated(text: str, actions: list[str]) -> RepairOutcome:
    """末尾が切れている文字列を、items の最後の完全な要素までで切り詰めて閉じる。

    完全な要素が1つも無い場合は、切り詰めずにそのまま返す。理由は、空の items として
    閉じてしまうと「抽出0件」と区別が付かず、出力が切れてデータを失ったことが
    呼び出し側から見えなくなるため。この場合は検証が失敗し、チャンクは失敗として記録される。
    """
    is_array = text.lstrip().startswith("[")
    array_open = 0 if is_array else _find_items_array(text)
    if array_open is None:
        return RepairOutcome(text, actions)
    if is_array:
        array_open = text.index("[")

    ends = _complete_element_ends(text, array_open)
    if not ends:
        return RepairOutcome(text, actions)

    last_end = ends[-1]
    # 最後の完全な要素より後に、カンマと空白以外の文字があれば、要素が途中で切れている
    rest = text[last_end:].strip().lstrip(",").strip()
    truncated = 1 if rest else 0

    body = text[:last_end] + "]"
    if not is_array:
        body = body + "}"
    else:
        body = '{"items": ' + body + "}"
        actions = [*actions, ACTION_WRAPPED_ARRAY_IN_ITEMS]
    return RepairOutcome(body, [*actions, ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM], truncated)


def repair_json_text(text: str) -> RepairOutcome:
    """モデルの出力を、ExtractionResult として読める形に近づける。

    例外は出さない。修復できなかった場合も、途中までの最善の結果を返す
    （読めるかどうかの判断は parse_with_repair が行う）。

    次の順に行う。
    1. 前後の空白を除去する。
    2. コードフェンスがあれば中身だけを取り出す。
    3. 最初の { または [ より前の散文を捨てる。
    4. 値が閉じていれば、その後の散文を捨てる。閉じていなければ 6 に進む。
    5. トップレベルが配列なら {"items": [...]} に包む。
    6. 末尾が切れていれば、items の最後の完全な要素までで切り詰めて閉じる。
    """
    actions: list[str] = []
    stripped = text.strip()
    if stripped != text:
        actions.append(ACTION_STRIPPED_WHITESPACE)

    unfenced = _strip_code_fence(stripped)
    if unfenced != stripped:
        actions.append(ACTION_REMOVED_CODE_FENCE)
        stripped = unfenced.strip()

    start = _first_bracket(stripped)
    if start is None:
        return RepairOutcome(stripped, [*actions, ACTION_NO_JSON_FOUND])
    if start > 0:
        actions.append(ACTION_REMOVED_PROSE_BEFORE)
        stripped = stripped[start:]

    end = _scan_value_end(stripped, 0)
    if end is None:
        return _repair_truncated(stripped, actions)

    if end < len(stripped.rstrip()):
        actions.append(ACTION_REMOVED_PROSE_AFTER)
    stripped = stripped[:end]

    if stripped.startswith("["):
        actions.append(ACTION_WRAPPED_ARRAY_IN_ITEMS)
        stripped = '{"items": ' + stripped + "}"
    return RepairOutcome(stripped, actions)


ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_model_with_repair(text: str, model_cls: type[ModelT]) -> tuple[ModelT, RepairOutcome]:
    """モデルの出力を修復してから model_cls として検証する。読めなければ UnrepairableJsonError を出す。

    抽出以外の LLM の用途(重要度の判定、返信の振り分け、話者の対応付け)も、この関数を通す
    (原則2-3: 背面がスキーマへの適合を保証するかどうかに関わらず、修復を通してから検証する)。
    配列を包む・末尾の切れを閉じる修復は `items` というキーを前提にしているので、
    配列を返させるスキーマは、抽出と同じ `{"items": [...]}` の形にすること。
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


def parse_with_repair(text: str) -> ParsedExtraction:
    """モデルの出力を修復してから ExtractionResult として検証する。

    読めなければ UnrepairableJsonError を出す。背面がスキーマへの適合を保証するかどうかに
    関わらず、テキストから ExtractionResult を作る経路はこの関数だけにする。
    こうしておくと、修復を通さずに検証してしまう経路が残らない。
    """
    result, outcome = parse_model_with_repair(text, ExtractionResult)
    return ParsedExtraction(result, outcome)
