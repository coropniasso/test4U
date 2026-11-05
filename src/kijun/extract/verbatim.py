"""stated_reason と applies_when の原文照合（CLAUDE.md 4-4、2026-11-05 のユーザーの決定）。

LLM は、発言で語られていない理由・適用条件を、few-shot の例からの写しや言い換え、推測で埋めることがある
（1章が禁じていること）。そこで、この2項目は「文字起こしの原文の連続した一部を、そのまま写した文字列」
に限り、プログラムで照合する。原文に見つからない値は、列には NULL を入れ、モデルが返した元の値は
`*_unverified` の列に残す（黙って捨てない）。言い換えは外れる。それは承知の上での取捨である。
文字起こしの誤り（「過剰書き」など）は、誤りのまま写してあれば照合を通る。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from kijun.extract.models import ExtractedRequest
from kijun.models import Utterance

# 照合で見る項目。ExtractedRequest の項目名と、_unverified の列名の前半に使う。
VERBATIM_FIELDS = ("stated_reason", "applies_when")


class CheckedRequest(ExtractedRequest):
    """照合を通した後の抽出結果。モデルの出力のスキーマ（ExtractedRequest）には含めない。

    項目を ExtractedRequest 自体に足さないのは、JSON スキーマとして Ollama に渡る出力の形が変わり、
    モデルがこの2項目を自分で埋めようとするためである。
    """

    stated_reason_unverified: str | None = None
    applies_when_unverified: str | None = None


def normalize(text: str) -> str:
    """照合用に文字列を整える。

    Unicode NFKC で全角・半角などをそろえ、句読点・記号（カテゴリ P, S）、空白・区切り（Z）、
    空白文字と制御文字を取り除き、英字を小文字にする。
    発話をつなぐときの改行や、文字起こしの句読点の揺れで照合が外れないようにするためである。
    """
    out = []
    for c in unicodedata.normalize("NFKC", text):
        if c.isspace() or unicodedata.category(c)[0] in ("P", "Z", "S", "C"):
            continue
        out.append(c)
    return "".join(out).lower()


def is_verbatim(value: str, source: str) -> bool:
    """value を normalize したものが空でなく、source を normalize したものの連続した一部なら True。"""
    needle = normalize(value)
    return bool(needle) and needle in normalize(source)


def source_text(utterances: list[Utterance]) -> str:
    """照合の原文。モデルに渡したチャンクの全発話の本文を、区切りなしでつなぐ。

    seq_from〜seq_to の範囲に絞らないのは、モデルが返す番号が正確でないためである。
    normalize が空白を取り除くので、区切りは要らない。
    """
    return "".join(u.text for u in utterances)


@dataclass(frozen=True)
class Rejection:
    """照合で外した値1件。"""

    field: str  # stated_reason / applies_when
    value: str
    quote: str  # どの項目の値か分かるように


def check_item(item: ExtractedRequest, source: str) -> CheckedRequest:
    """stated_reason と applies_when を原文と照合する。

    原文にない値は None にし、元の値を *_unverified に入れる。通った値と、元から None の値はそのまま。
    """
    update: dict[str, str | None] = {}
    for name in VERBATIM_FIELDS:
        value = getattr(item, name)
        if value is not None and not is_verbatim(value, source):
            update[name] = None
            update[f"{name}_unverified"] = value
    return CheckedRequest.model_validate({**item.model_dump(), **update})


def check_items(items: list[ExtractedRequest], source: str) -> list[CheckedRequest]:
    return [check_item(i, source) for i in items]


def rejections(items: list[ExtractedRequest]) -> list[Rejection]:
    """check_items を通した結果から、外した値の一覧を取り出す。"""
    out = []
    for i in items:
        for name in VERBATIM_FIELDS:
            value = getattr(i, f"{name}_unverified", None)
            if value is not None:
                out.append(Rejection(name, value, i.quote))
    return out
