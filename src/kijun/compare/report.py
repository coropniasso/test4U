"""Qwen と Claude の抽出結果の比較表（作業表 #5、計画書 第8節）。

承認件数はユーザーがレビューして初めて決まる数字なので、このレポートでは出せない。
出せるのは、レビューのための対照表と、承認を待たずに計算できる数字である。
承認の記録は第2弾の Discord で行う。第1弾では、出力した Markdown に手で印を付ける。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import duckdb

from kijun.config import Config
from kijun.db import repo
from kijun.db.repo import RequestRow
from kijun.extract.chunk import normalize_quote
from kijun.extract.run import extract_log_path, read_failed_chunk_count

# 両モデルの抽出結果を同じ指摘とみなす条件（計画書 8-2）
PAIR_MAX_SEQ_DISTANCE = 2
PAIR_MIN_QUOTE_RATIO = 0.6


@dataclass(frozen=True)
class ModelStats:
    model_name: str
    total: int
    reusable: int
    reason_filled: int  # stated_reason が非 NULL の件数
    applies_when_filled: int
    pass_criterion_filled: int
    failed_chunks: int | None  # 記録が無ければ None

    @property
    def reusable_ratio(self) -> float | None:
        return self.reusable / self.total if self.total else None


@dataclass
class Pairing:
    """2つのモデルの抽出結果の突き合わせ結果。"""

    matched: list[tuple[RequestRow, RequestRow]]
    only_a: list[RequestRow]
    only_b: list[RequestRow]


def compute_stats(
    model_name: str, rows: list[RequestRow], failed_chunks: int | None
) -> ModelStats:
    return ModelStats(
        model_name=model_name,
        total=len(rows),
        reusable=sum(1 for r in rows if r.reusable),
        reason_filled=sum(1 for r in rows if r.stated_reason is not None),
        applies_when_filled=sum(1 for r in rows if r.applies_when is not None),
        pass_criterion_filled=sum(1 for r in rows if r.pass_criterion is not None),
        failed_chunks=failed_chunks,
    )


def quote_ratio(a: str, b: str) -> float:
    """quote の文字列類似度（difflib の比率）。空白と句読点を除いて比べる。"""
    return SequenceMatcher(None, normalize_quote(a), normalize_quote(b)).ratio()


def pair_requests(a_rows: list[RequestRow], b_rows: list[RequestRow]) -> Pairing:
    """両モデルの抽出結果を、seq_from の近さと quote の類似度で1対1に突き合わせる。

    seq_from の差が PAIR_MAX_SEQ_DISTANCE 以内で、quote の類似度が PAIR_MIN_QUOTE_RATIO 以上の
    組を候補にする。候補を類似度の高い順に確定し、同じ行を2回使わない。
    """
    candidates: list[tuple[float, int, int, int]] = []
    for i, a in enumerate(a_rows):
        for j, b in enumerate(b_rows):
            dist = abs(a.seq_from - b.seq_from)
            if dist > PAIR_MAX_SEQ_DISTANCE:
                continue
            ratio = quote_ratio(a.quote, b.quote)
            if ratio >= PAIR_MIN_QUOTE_RATIO:
                candidates.append((-ratio, dist, i, j))
    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[RequestRow, RequestRow]] = []
    for _neg_ratio, _dist, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((a_rows[i], b_rows[j]))
    matched.sort(key=lambda p: min(p[0].seq_from, p[1].seq_from))
    return Pairing(
        matched=matched,
        only_a=[r for i, r in enumerate(a_rows) if i not in used_a],
        only_b=[r for j, r in enumerate(b_rows) if j not in used_b],
    )


def _cell(text: str | None) -> str:
    """Markdown の表のセルに入れられる形にする（改行と縦線を置き換える）。"""
    if text is None:
        return "null"
    return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _describe(r: RequestRow | None) -> str:
    """対照表の1セル。モデルの出力を項目ごとに並べる。"""
    if r is None:
        return "（抽出なし）"
    parts = [
        f"発話 {r.seq_from}〜{r.seq_to} / {_cell(r.speaker_from_label)}→{_cell(r.speaker_to_label)}",
        f"引用: {_cell(r.quote)}",
        f"要約: {_cell(r.request_summary)}",
        f"理由: {_cell(r.stated_reason)}",
        f"適用条件: {_cell(r.applies_when)}",
        f"reusable: {str(r.reusable).lower()}",
        f"基準: {_cell(r.pass_criterion)}",
    ]
    return "<br>".join(parts)


def _ratio_text(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def select_models(
    available: list[str], cfg: Config, requested: list[str] | None = None
) -> list[str]:
    """比較するモデルを決める。

    requested があればそれを使う。無ければ、設定の ollama と claude のモデルのうち
    抽出結果があるものを（Qwen、Claude の順で）使い、1つも無ければ抽出結果のある全モデルを使う。
    """
    if requested:
        return requested
    preferred = [m for m in (cfg.extract.ollama.model, cfg.extract.claude.model) if m in available]
    return preferred or available


def build_report(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    conversation_id: str,
    models: list[str] | None = None,
    log_reader: Callable[[Path], int | None] = read_failed_chunk_count,
) -> str:
    """比較表の Markdown を作る。

    models が 2 つなら、先頭を左（A）、次を右（B）にした対照表を付ける。
    それ以外の数のときは、数字の表だけを出す。
    """
    available = repo.list_request_models(con, conversation_id)
    if not available:
        raise LookupError(
            f"会議 {conversation_id} の抽出結果がありません。先に `kijun extract` を実行してください。"
        )
    chosen = select_models(available, cfg, models)
    for m in chosen:
        if m not in available:
            raise LookupError(f"モデル {m} の抽出結果が会議 {conversation_id} にありません（あるモデル: {available}）")

    rows_by_model = {m: repo.get_requests(con, conversation_id, m) for m in chosen}
    stats = [
        compute_stats(m, rows_by_model[m], log_reader(extract_log_path(cfg, conversation_id, m)))
        for m in chosen
    ]

    lines: list[str] = [f"# 抽出結果の比較: {conversation_id}", ""]
    lines += [
        "承認件数は、レビューして初めて決まる数字なので、この表には出ていません。",
        "第1弾では、3 の対照表を見て、承認する方に手で印を付けてください（Discord での記録は第2弾）。",
        "",
        "## 1. 承認を待たずに計算できる数字",
        "",
        "| 項目 | " + " | ".join(_cell(s.model_name) for s in stats) + " |",
        "|---|" + "---|" * len(stats),
        "| 抽出件数 | " + " | ".join(str(s.total) for s in stats) + " |",
        "| reusable = true の件数 | " + " | ".join(str(s.reusable) for s in stats) + " |",
        "| reusable = true の割合 | " + " | ".join(_ratio_text(s.reusable_ratio) for s in stats) + " |",
        "| stated_reason が非 NULL の件数 | " + " | ".join(str(s.reason_filled) for s in stats) + " |",
        "| applies_when が非 NULL の件数 | " + " | ".join(str(s.applies_when_filled) for s in stats) + " |",
        "| pass_criterion が非 NULL の件数 | " + " | ".join(str(s.pass_criterion_filled) for s in stats) + " |",
        "| 失敗したチャンク数 | "
        + " | ".join("記録なし" if s.failed_chunks is None else str(s.failed_chunks) for s in stats)
        + " |",
        "",
        "stated_reason と applies_when が非 NULL の件数が、他のモデルに比べて極端に多い場合は、"
        "発言で語られていない理由・条件をモデルが推測で埋めている疑いがあります。"
        "対照表で、引用に理由・条件が実際に含まれているかを確認してください。",
        "",
    ]

    if len(chosen) != 2:
        lines += [
            "## 2. 対応付けと対照表",
            "",
            f"比較するモデルが {len(chosen)} 個なので、対照表は出しません（2 個のときだけ出します）。"
            "`--model` を 2 回指定して、比較するモデルを選んでください。",
            "",
        ]
        return "\n".join(lines)

    a_name, b_name = chosen
    pairing = pair_requests(rows_by_model[a_name], rows_by_model[b_name])
    lines += [
        "## 2. 対応付け",
        "",
        f"seq_from の差が {PAIR_MAX_SEQ_DISTANCE} 以内で、quote の類似度（difflib の比率）が "
        f"{PAIR_MIN_QUOTE_RATIO} 以上のものを、同じ指摘とみなしています。",
        "",
        "| 区分 | 件数 |",
        "|---|---|",
        f"| 両方が抽出 | {len(pairing.matched)} |",
        f"| {_cell(a_name)} だけ | {len(pairing.only_a)} |",
        f"| {_cell(b_name)} だけ | {len(pairing.only_b)} |",
        "",
        "## 3. レビュー用の対照表",
        "",
        "採用する方に印を付けてください（例: A、B、両方、どちらも不要）。",
        "",
        f"A = {_cell(a_name)} / B = {_cell(b_name)}",
        "",
        "| 発話 | A の出力 | B の出力 | 採用 |",
        "|---|---|---|---|",
    ]
    table_rows: list[tuple[int, RequestRow | None, RequestRow | None]] = []
    for a, b in pairing.matched:
        table_rows.append((min(a.seq_from, b.seq_from), a, b))
    table_rows += [(a.seq_from, a, None) for a in pairing.only_a]
    table_rows += [(b.seq_from, None, b) for b in pairing.only_b]
    table_rows.sort(key=lambda t: (t[0], t[1] is None))
    for seq, a, b in table_rows:
        lines.append(f"| {seq} | {_describe(a)} | {_describe(b)} |  |")
    lines += [
        "",
        "## 4. 承認件数（レビュー後に手で記入）",
        "",
        "| モデル | 承認した件数 |",
        "|---|---|",
        f"| {_cell(a_name)} |  |",
        f"| {_cell(b_name)} |  |",
        "",
        "CLAUDE.md 4-4 の判断基準: Qwen の承認件数が Claude の 8 割以上なら Qwen に一本化し、未満なら Claude API を使う。",
        "",
    ]
    return "\n".join(lines)
