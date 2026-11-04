"""抽出の設定を変えて同じ会議を測り、結果を1つの JSON と Markdown にまとめる。

目的は、`reasoning_effort` やプロンプトを変えたときに出力の品質がどう変わるかを、
人の目視に頼らず同じ条件で並べることである。`kijun extract` との違いは3点。

1. **DB に書かない。** 測定が `extracted_requests` を書き換えないので、`run_label` の付け替えや
   `delete_requests` による取り違えが起きない。読むのは `utterances` だけである。
2. **1回の実行で複数の設定を順に試す。** 設定の違い以外（プロンプト、チャンク分割、入力の発話）が
   必ず揃う。実行を別々に行うと、どちらかに別の変更が混ざる余地が残る。
3. **文字起こしに無い語を機械的に検出する。** 目視では見落とす。

リクエストの本文は `kijun extract` と同じ `build_payload` で作る。測定したものと本番が送るものが
食い違わないようにするためである。

使い方:

    uv run python scripts/measure_extraction.py \
      --config <設定ファイル> --conversation <会議ID> \
      --variants scripts/variants.example.toml --out <出力先のディレクトリ>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kijun.config import Config, load_config
from kijun.db import repo
from kijun.extract.chunk import split_into_chunks
from kijun.extract.repair import UnrepairableJsonError, parse_with_repair
from kijun.extract.strata_impl import build_payload
from kijun.transport import HttpxTransport, TransportError

# 内容語の目安。カタカナは3文字以上、漢字は2文字以上の連なりを拾う。
# 形態素解析を入れないのは、この検出の用途が「人が読む候補の一覧を出す」ことであり、
# 取りこぼしより見落としを防ぐことを優先するためである。1文字の漢字（求、書など）は
# 要約の言い回しで必ず出るので拾わない。
KATAKANA_RUN = re.compile(r"[ァ-ヴー]{3,}")
KANJI_RUN = re.compile(r"[一-龥]{2,}")

# 値の由来を発言に求める項目（CLAUDE.md 4-4 と抽出プロンプトの「最も重要な規則」）。
GROUNDED_FIELDS = ("request_summary", "stated_reason", "applies_when", "pass_criterion")


def content_terms(text: str) -> list[str]:
    """文字列から内容語の候補を取り出す。出た順で、重複は取り除く。"""
    found = KATAKANA_RUN.findall(text) + KANJI_RUN.findall(text)
    seen: dict[str, None] = {}
    for t in found:
        seen.setdefault(t, None)
    return list(seen)


def novel_terms(text: str | None, reference: str) -> list[str]:
    """text の内容語のうち、reference に出てこないものを返す。

    reference は会議の文字起こし全文。ここに無い語が出力に入っていれば、
    モデルが語を補ったか、言い換えで新しい語を作ったかのどちらかである。
    どちらであるかは人が見て判断する。この関数は候補を挙げるだけである。
    """
    if not text:
        return []
    return [t for t in content_terms(text) if t not in reference]


@dataclass
class Variant:
    label: str
    reasoning_effort: str = ""
    reasoning_budget_tokens: int = 0
    max_tokens: int = 20000
    # 0 なら config.toml の timeout_sec を使う。正の値のときだけ上書きする。
    # 推論の予算を大きくした設定は、設定ファイルの打ち切り時間に掛かることがある。
    # 実効の予算（下記）を観測された最も遅い生成速度で割った秒数より大きくしておく。
    timeout_sec: int = 0

    def effective_reasoning_budget(self) -> int:
        """この設定で実際に推論に使える上限。0 は「Strata の逆算に任せる」。

        明示した reasoning_budget_tokens は max_tokens からの逆算より優先されるが、そのあとに
        答え用の取り置きで切り下げられる（2026-11-04 実機で Strata のソースとログで確認）。
        取り置き = min(max(512, max_tokens // 4), max_tokens // 2)。額は指定できない。
        """
        if self.max_tokens <= 0:
            return self.reasoning_budget_tokens
        reserve = min(max(512, self.max_tokens // 4), self.max_tokens // 2)
        ceiling = self.max_tokens - reserve
        if self.reasoning_budget_tokens <= 0:
            return ceiling
        return min(self.reasoning_budget_tokens, ceiling)

    def applied_to(self, cfg: Config) -> Config:
        out = cfg.model_copy(deep=True)
        out.extract.strata.reasoning_effort = self.reasoning_effort
        out.extract.strata.reasoning_budget_tokens = self.reasoning_budget_tokens
        out.extract.strata.max_tokens = self.max_tokens
        if self.timeout_sec > 0:
            out.extract.strata.timeout_sec = self.timeout_sec
        return out


@dataclass
class ChunkOutcome:
    chunk_index: int
    seq_from: int
    seq_to: int
    elapsed_sec: float = 0.0
    status_code: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    items: list[dict] = field(default_factory=list)
    repair_actions: list[str] = field(default_factory=list)
    error: str | None = None


def load_variants(path: Path | None) -> list[Variant]:
    """変える設定の一覧を読む。省略時は、比較に必要な2つを既定にする。"""
    if path is None:
        return [
            Variant("medium-mt20k", reasoning_effort="medium", max_tokens=20000),
            Variant(
                "xhigh-rb40k",
                reasoning_effort="xhigh",
                reasoning_budget_tokens=40000,
                max_tokens=48000,
                timeout_sec=3600,
            ),
        ]
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    variants = [Variant(**v) for v in data.get("variant", [])]
    if not variants:
        raise SystemExit(f"{path} に [[variant]] が1つもありません")
    labels = [v.label for v in variants]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"label が重複しています: {labels}")
    return variants


def run_variant(cfg: Config, variant: Variant, chunks: list, transport) -> list[ChunkOutcome]:
    """1つの設定で全チャンクを投げる。失敗したチャンクも記録して次へ進む。"""
    vcfg = variant.applied_to(cfg)
    url = f"{vcfg.extract.strata.base_url}{vcfg.extract.strata.chat_path}"
    outcomes = []
    for chunk in chunks:
        payload = build_payload(vcfg, chunk, vcfg.extract.strata.enforces_json_schema)
        out = ChunkOutcome(chunk.index, chunk.seq_from, chunk.seq_to)
        print(
            f"[{variant.label}] チャンク {chunk.index + 1}/{len(chunks)}"
            f"（発話 {chunk.seq_from}〜{chunk.seq_to}）を送ります",
            file=sys.stderr,
        )
        started = time.monotonic()
        try:
            response = transport.post(url, payload, vcfg.extract.strata.timeout_sec)
        except TransportError as e:
            out.elapsed_sec = time.monotonic() - started
            out.error = f"接続できませんでした: {e}"
            outcomes.append(out)
            continue
        out.elapsed_sec = time.monotonic() - started
        out.status_code = response.status_code
        if not response.ok:
            out.error = f"HTTP {response.status_code}: {response.text[:300]}"
            outcomes.append(out)
            continue
        body = response.json()
        usage = body.get("usage") or {}
        out.prompt_tokens = usage.get("prompt_tokens")
        out.completion_tokens = usage.get("completion_tokens")
        choice = (body.get("choices") or [{}])[0]
        out.finish_reason = choice.get("finish_reason")
        text = (choice.get("message") or {}).get("content") or ""
        try:
            parsed = parse_with_repair(text)
        except UnrepairableJsonError as e:
            out.error = f"出力を JSON として読めませんでした: {e}"
            outcomes.append(out)
            continue
        out.repair_actions = list(parsed.repair.actions)
        out.items = [i.model_dump() for i in parsed.result.items]
        print(
            f"[{variant.label}] {out.elapsed_sec:.1f} 秒、出力 {out.completion_tokens} トークン、"
            f"{len(out.items)} 件、finish_reason={out.finish_reason}",
            file=sys.stderr,
        )
        outcomes.append(out)
    return outcomes


def summarize(outcomes: list[ChunkOutcome], transcript: str) -> dict:
    """1つの設定の結果を1行分の数値にまとめる。"""
    items = [i for o in outcomes for i in o.items]
    novel: dict[str, list[str]] = {}
    for f in GROUNDED_FIELDS:
        terms: list[str] = []
        for i in items:
            terms += novel_terms(i.get(f), transcript)
        novel[f] = sorted(set(terms))
    return {
        "elapsed_sec": round(sum(o.elapsed_sec for o in outcomes), 1),
        "completion_tokens": sum(o.completion_tokens or 0 for o in outcomes),
        "prompt_tokens": sum(o.prompt_tokens or 0 for o in outcomes),
        "finish_reasons": sorted({o.finish_reason for o in outcomes if o.finish_reason}),
        "failed_chunks": sum(1 for o in outcomes if o.error),
        "item_count": len(items),
        "non_null": {f: sum(1 for i in items if i.get(f) is not None) for f in GROUNDED_FIELDS},
        "reusable_true": sum(1 for i in items if i.get("reusable")),
        "novel_terms": novel,
        "novel_term_count": sum(len(v) for v in novel.values()),
    }


def build_markdown(conversation_id: str, variants: list[Variant], results: dict) -> str:
    """人が読む対照表。数値と、文字起こしに無い語の一覧を並べる。"""
    labels = [v.label for v in variants]
    lines = [
        f"# 抽出の設定ごとの測定: {conversation_id}",
        "",
        f"測定日時: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 設定",
        "",
        "| label | reasoning_effort | reasoning_budget_tokens | max_tokens | 実効の推論の予算 |",
        "|---|---|---|---|---|",
    ]
    for v in variants:
        effort = v.reasoning_effort or "（送らない＝テンプレートの既定 xhigh）"
        budget = v.reasoning_budget_tokens or "（送らない）"
        lines.append(
            f"| {v.label} | {effort} | {budget} | {v.max_tokens} | {v.effective_reasoning_budget()} |"
        )
    lines += [
        "",
        "実効の推論の予算 = min(reasoning_budget_tokens, max_tokens - 答え用の取り置き)。",
        "取り置き = min(max(512, max_tokens // 4), max_tokens // 2) で、額は指定できない。",
        "推論がこの値に達すると、Strata は推論を閉じて答えを書かせる。達したかどうかは API の応答には",
        "出ないので、サーバープロセスの標準出力の `[strata] thinking:` の行で確認する。",
    ]
    lines += ["", "## 結果", "", "| 項目 | " + " | ".join(labels) + " |"]
    lines.append("|---|" + "---|" * len(labels))

    def row(name: str, pick) -> None:
        lines.append(f"| {name} | " + " | ".join(str(pick(results[l])) for l in labels) + " |")

    row("所要（秒）", lambda s: s["elapsed_sec"])
    row("出力トークン", lambda s: s["completion_tokens"])
    row("finish_reason", lambda s: "/".join(s["finish_reasons"]) or "—")
    row("失敗したチャンク", lambda s: s["failed_chunks"])
    row("抽出件数", lambda s: s["item_count"])
    row("reusable = true", lambda s: s["reusable_true"])
    for f in GROUNDED_FIELDS:
        row(f"{f} が非 null", lambda s, f=f: s["non_null"][f])
    row("文字起こしに無い語の数", lambda s: s["novel_term_count"])

    lines += [
        "",
        "## 文字起こしに無い語",
        "",
        "下の語は、会議の文字起こし全文に出てこない内容語である。**すべてが誤りではない。**",
        "要約の言い換えで新しい語ができることはある。モデルが聞き取れない語に別の語を当てはめた",
        "ものか、正当な言い換えかは、人が見て判断する。",
        "",
    ]
    for label in labels:
        lines.append(f"### {label}")
        lines.append("")
        found = False
        for f in GROUNDED_FIELDS:
            terms = results[label]["novel_terms"][f]
            if terms:
                found = True
                lines.append(f"- `{f}`: " + "、".join(terms))
        if not found:
            lines.append("文字起こしに無い語はなかった。")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None, help="設定ファイル。省略すると ./config.toml")
    ap.add_argument("--conversation", required=True, help="測る会議ID")
    ap.add_argument("--variants", type=Path, default=None, help="設定の一覧（TOML）。省略すると既定の2つ")
    ap.add_argument("--out", type=Path, default=None, help="出力先のディレクトリ。省略すると data/measure")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    variants = load_variants(args.variants)

    # 読み取り専用で開く。この測定が extracted_requests を書き換えないことを、接続の側で保証する。
    con = repo.connect(cfg.paths.db, read_only=True)
    try:
        utterances = repo.get_utterances(con, args.conversation)
    finally:
        con.close()
    if not utterances:
        raise SystemExit(
            f"会議 {args.conversation} の発話がありません。先に kijun transcribe を実行してください。"
        )

    transcript = "\n".join(u.text for u in utterances)
    chunks = split_into_chunks(
        utterances, cfg.extract.chunk_minutes, cfg.extract.overlap_minutes
    )
    print(
        f"会議 {args.conversation}: 発話 {len(utterances)} 件、チャンク {len(chunks)} 件、"
        f"設定 {len(variants)} 通りを順に測ります",
        file=sys.stderr,
    )

    transport = HttpxTransport()
    raw: dict[str, list[ChunkOutcome]] = {}
    results: dict[str, dict] = {}
    for v in variants:
        raw[v.label] = run_variant(cfg, v, chunks, transport)
        results[v.label] = summarize(raw[v.label], transcript)

    out_dir = args.out or (cfg.paths.db.parent / "measure")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.conversation}_{stamp}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    json_path.write_text(
        json.dumps(
            {
                "conversation_id": args.conversation,
                "measured_at": datetime.now().isoformat(timespec="seconds"),
                "chunk_minutes": cfg.extract.chunk_minutes,
                "overlap_minutes": cfg.extract.overlap_minutes,
                "model": cfg.extract.strata.model,
                "enforces_json_schema": cfg.extract.strata.enforces_json_schema,
                "utterance_count": len(utterances),
                "chunk_count": len(chunks),
                "variants": [vars(v) for v in variants],
                "summary": results,
                "outcomes": {k: [vars(o) for o in v] for k, v in raw.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(build_markdown(args.conversation, variants, results), encoding="utf-8")
    print(f"\n保存しました:\n  {json_path}\n  {md_path}", file=sys.stderr)

    failed = sum(s["failed_chunks"] for s in results.values())
    if failed:
        print(f"注意: 失敗したチャンクが合計 {failed} 件あります", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
