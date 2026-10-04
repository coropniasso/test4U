"""抽出結果の pydantic モデルと JSON スキーマ（計画書 7-1）。

トップレベルを `items` を持つオブジェクトにする理由: Ollama の `format` パラメータに渡す
JSON スキーマは、トップレベルが配列だと従わないことがあるため。
CLAUDE.md 4-4 の「出力は JSON 配列のみ」は、プロンプトでは「前置きを付けず JSON だけを返す」
という意味で守り、受け取る形を {"items": [...]} にして Python 側で配列に展開する。

各フィールドの description は JSON スキーマに入り、`format` で出力を制限するときにモデルへの
ヒントにもなる。プロンプト（prompt.py）の説明と食い違わせないこと。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedRequest(BaseModel):
    """ある人物が別の人物に対して行った要求・指摘・指示の1件。"""

    seq_from: int = Field(
        description="この要求が始まる発話の番号。入力の行頭の [番号] をそのまま書く。"
    )
    seq_to: int = Field(
        description="この要求が終わる発話の番号。1つの発話で完結していれば seq_from と同じ値。"
    )
    speaker_from_label: str = Field(
        description="要求・指摘・指示を言った側の話者ラベル（例: SPEAKER_00）。"
    )
    speaker_to_label: str | None = Field(
        description="言われた側の話者ラベル。発言から特定できなければ null。"
    )
    counterpart_role: str | None = Field(
        description="言われた側の役割（例: 後輩）。発言の中で分かる場合だけ書く。分からなければ null。"
    )
    artifact_type: str | None = Field(
        description="対象の成果物の種類を表す英小文字の短い識別子（例: announcement, report）。分からなければ null。"
    )
    quote: str = Field(
        description="根拠になる発言の引用。入力の文面をそのまま写す。要約や言い換えをしない。"
    )
    request_summary: str = Field(
        description="何を求めたかを1文で書いた要約。"
    )
    stated_reason: str | None = Field(
        description="発言の中で実際に語られた理由だけ。推測で補わない。語られていなければ null。"
    )
    applies_when: str | None = Field(
        description="発言の中で実際に語られた適用条件だけ。推測で補わない。語られていなければ null。"
    )
    reusable: bool = Field(
        description="同じ種類の成果物を別の機会に作るときにも使えるチェック項目になる場合は true。この場限りの修正指示なら false。"
    )
    pass_criterion: str | None = Field(
        description="Yes か No で判定できる形の基準（例: 冒頭2行以内に、読み手にしてほしい行動が書かれているか）。作れなければ null。"
    )


class ExtractionResult(BaseModel):
    """1チャンクの抽出結果。該当する発言が無ければ items は空の配列。"""

    items: list[ExtractedRequest]
