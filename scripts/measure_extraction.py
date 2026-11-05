"""抽出の設定を変えて同じ会議を測り、結果を1つの JSON と Markdown にまとめる。

目的は、モデル・think・num_ctx・num_predict を変えたときに出力の品質がどう変わるかを、
人の目視に頼らず同じ条件で並べることである。`kijun extract` との違いは3点。

1. **DB に書かない。** 測定が `extracted_requests` を書き換えないので、`run_label` の付け替えや
   `delete_requests` による取り違えが起きない。読むのは `utterances` だけである。
2. **1回の実行で複数の設定を順に試す。** 設定の違い以外（プロンプト、チャンク分割、入力の発話）が
   必ず揃う。実行を別々に行うと、どちらかに別の変更が混ざる余地が残る。
3. **文字起こしに無い語を機械的に検出する。** 目視では見落とす。

リクエストの本文は `kijun extract` と同じ `build_payload`（Ollama の /api/chat）で作る。
測定したものと本番が送るものが食い違わないようにするためである。

結果の表には、Strata（2026-11-04、同じ会議・同じプロンプト）の基準を並べる。基準が使えるのは
会議 20261103_1930_f2f_teirei だけである。モデルの採否は人が決める。この表は判断の材料を並べるだけで、
どれがよいとは書かない。

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

from pydantic import ValidationError

from kijun import ollama_api
from kijun.config import Config, load_config
from kijun.db import repo
from kijun.extract.chunk import split_into_chunks
from kijun.extract.ollama_impl import build_payload
from kijun.extract.repair import UnrepairableJsonError, parse_with_repair
from kijun.transport import HttpxTransport, TransportError

# 内容語の目安。カタカナは3文字以上、漢字は2文字以上の連なりを拾う。
# 形態素解析を入れないのは、この検出の用途が「人が読む候補の一覧を出す」ことであり、
# 取りこぼしより見落としを防ぐことを優先するためである。1文字の漢字（求、書など）は
# 要約の言い回しで必ず出るので拾わない。
KATAKANA_RUN = re.compile(r"[ァ-ヴー]{3,}")
KANJI_RUN = re.compile(r"[一-龥]{2,}")

# Strata の基準（CLAUDE.md 7章、data/measure/20261103_1930_f2f_teirei_20261104_143728.md）。
# 同じ会議（20261103_1930_f2f_teirei）、同じプロンプトで測った値。会議が違えば比べられない。
BASELINE_CONVERSATION = "20261103_1930_f2f_teirei"
STRATA_BASELINE = {
    "Strata xhigh": {"item_count": 6, "stated_reason": 4, "applies_when": 2, "pass_criterion": 4, "novel": 5},
    "Strata medium": {"item_count": 6, "stated_reason": 4, "applies_when": 1, "pass_criterion": 4, "novel": 17},
}

# 人が確認する2点（docs/implementation-plan.md 15-2）。
# 1. 文字起こしが壊れた発話（「感嘆符」が「簡単フォーイス」に化けた箇所）が transcription_unclear = true、
#    pass_criterion が null、reusable が false になっているか。
# 2. applies_when に「参加者が募りたい場合」が取れているか。
UNCLEAR_QUOTE_MARK = "フォーイス"
APPLIES_WHEN_MARK = "募り"

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
    """変える設定。省略した項目（空文字・0・未指定）は config.toml の値を使う。"""

    label: str
    model: str = ""
    # TOML には null が無いので、キーを書かなければ None（variant としては config.toml の値のまま）。
    think: bool | None = None
    num_ctx: int = 0
    num_predict: int = 0
    # 0 なら config.toml の timeout_sec を使う。正の値のときだけ上書きする。
    timeout_sec: int = 0

    def applied_to(self, cfg: Config) -> Config:
        """設定を上書きした Config を返す。値の検査（num_ctx と入力の整合など）をやり直す。

        上書きの結果が設定の検査に通らない組み合わせ（例: num_ctx を下げて入力が収まらない）は
        ValidationError になる。測定の前に拒否するのは、入力が黙って切り捨てられた結果を並べないためである。
        """
        data = cfg.model_dump()
        o = data["extract"]["ollama"]
        if self.model:
            o["model"] = self.model
        if self.think is not None:
            o["think"] = self.think
        if self.num_ctx > 0:
            o["num_ctx"] = self.num_ctx
        if self.num_predict > 0:
            o["num_predict"] = self.num_predict
        if self.timeout_sec > 0:
            o["timeout_sec"] = self.timeout_sec
        return Config.model_validate(data)


@dataclass
class ChunkOutcome:
    chunk_index: int
    seq_from: int
    seq_to: int
    elapsed_sec: float = 0.0
    status_code: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    done_reason: str | None = None
    thinking_chars: int = 0
    items: list[dict] = field(default_factory=list)
    repair_actions: list[str] = field(default_factory=list)
    error: str | None = None


def load_variants(path: Path | None) -> list[Variant]:
    """変える設定の一覧を読む。省略時は、config.toml のままの1通りだけを測る。"""
    if path is None:
        return [Variant("config")]
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
    ollama = vcfg.extract.ollama
    url = ollama_api.url_of(ollama, ollama_api.CHAT_PATH)
    outcomes = []
    for chunk in chunks:
        payload = build_payload(vcfg, chunk)
        out = ChunkOutcome(chunk.index, chunk.seq_from, chunk.seq_to)
        print(
            f"[{variant.label}] チャンク {chunk.index + 1}/{len(chunks)}"
            f"（発話 {chunk.seq_from}〜{chunk.seq_to}）を送ります",
            file=sys.stderr,
        )
        started = time.monotonic()
        try:
            response = transport.post(url, payload, ollama.timeout_sec)
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
        try:
            reply = ollama_api.parse_chat_reply(response)
        except ollama_api.OllamaError as e:
            out.error = str(e)
            outcomes.append(out)
            continue
        out.prompt_tokens = reply.prompt_eval_count
        out.completion_tokens = reply.eval_count
        out.done_reason = reply.done_reason
        out.thinking_chars = reply.thinking_chars
        if ollama_api.input_was_truncated(ollama, reply):
            # 切り捨てられた入力から出た結果を並べると、発話が静かに欠けた結果を品質として比べてしまう。
            out.error = ollama_api.truncation_message(ollama, reply)
            outcomes.append(out)
            continue
        try:
            parsed = parse_with_repair(reply.content)
        except UnrepairableJsonError as e:
            out.error = f"出力を JSON として読めませんでした: {e}"
            outcomes.append(out)
            continue
        out.repair_actions = list(parsed.repair.actions)
        out.items = [i.model_dump() for i in parsed.result.items]
        print(
            f"[{variant.label}] {out.elapsed_sec:.1f} 秒、出力 {out.completion_tokens} トークン、"
            f"{len(out.items)} 件、done_reason={out.done_reason}",
            file=sys.stderr,
        )
        outcomes.append(out)
    return outcomes


def human_checks(items: list[dict]) -> dict:
    """人が確認する2点を、該当する項目の値として取り出す。判定はしない（人が見る）。

    - unclear_items: quote に「フォーイス」を含む項目の transcription_unclear / pass_criterion / reusable。
      壊れた発話が transcription_unclear = true、pass_criterion が null、reusable が false なら期待どおり。
    - applies_when_hits: applies_when に「募り」を含む項目（「参加者が募りたい場合」が取れたか）。
    """
    unclear = [
        {
            "quote": i.get("quote"),
            "transcription_unclear": i.get("transcription_unclear"),
            "pass_criterion": i.get("pass_criterion"),
            "reusable": i.get("reusable"),
        }
        for i in items
        if UNCLEAR_QUOTE_MARK in (i.get("quote") or "")
    ]
    hits = [i["applies_when"] for i in items if APPLIES_WHEN_MARK in (i.get("applies_when") or "")]
    return {"unclear_items": unclear, "applies_when_hits": hits}


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
        "done_reasons": sorted({o.done_reason for o in outcomes if o.done_reason}),
        "failed_chunks": sum(1 for o in outcomes if o.error),
        "item_count": len(items),
        "non_null": {f: sum(1 for i in items if i.get(f) is not None) for f in GROUNDED_FIELDS},
        "reusable_true": sum(1 for i in items if i.get("reusable")),
        "novel_terms": novel,
        "novel_term_count": sum(len(v) for v in novel.values()),
        "human_checks": human_checks(items),
    }


def build_markdown(conversation_id: str, variants: list[Variant], results: dict, cfg: Config) -> str:
    """人が読む対照表。数値と、Strata の基準、人が確認する2点、文字起こしに無い語を並べる。"""
    labels = [v.label for v in variants]
    o = cfg.extract.ollama
    lines = [
        f"# 抽出の設定ごとの測定: {conversation_id}",
        "",
        f"測定日時: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 設定",
        "",
        "| label | model | think | num_ctx | num_predict |",
        "|---|---|---|---|---|",
    ]
    for v in variants:
        model = v.model or f"{o.model}（config）"
        think = {True: "true", False: "false", None: f"（variant では指定せず、config は {o.think}）"}[v.think]
        ctx = v.num_ctx or f"{o.num_ctx}（config）"
        predict = v.num_predict or f"{o.num_predict}（config）"
        lines.append(f"| {v.label} | {model} | {think} | {ctx} | {predict} |")
    lines += [
        "",
        "think = true のモデルは、答えの前に推論を書く。推論も num_predict に数えられる。",
        "モデルが think に対応しない場合は HTTP 400 になり、その設定のチャンクは失敗として記録される。",
    ]
    lines += ["", "## 結果", "", "| 項目 | " + " | ".join(labels) + " |"]
    lines.append("|---|" + "---|" * len(labels))

    def row(name: str, pick) -> None:
        lines.append(f"| {name} | " + " | ".join(str(pick(results[l])) for l in labels) + " |")

    row("所要（秒）", lambda s: s["elapsed_sec"])
    row("出力トークン", lambda s: s["completion_tokens"])
    row("done_reason", lambda s: "/".join(s["done_reasons"]) or "—")
    row("失敗したチャンク", lambda s: s["failed_chunks"])
    row("抽出件数", lambda s: s["item_count"])
    row("reusable = true", lambda s: s["reusable_true"])
    for f in GROUNDED_FIELDS:
        row(f"{f} が非 null", lambda s, f=f: s["non_null"][f])
    row("文字起こしに無い語の数", lambda s: s["novel_term_count"])

    lines += ["", "## Strata の基準との対照", ""]
    if conversation_id == BASELINE_CONVERSATION:
        base = list(STRATA_BASELINE)
        lines += [
            "同じ会議・同じプロンプトで測った Strata の結果（2026-11-04）を右に並べる。"
            "これは基準であり、どれを採るかは人が決める。",
            "",
            "| 項目 | " + " | ".join(labels + base) + " |",
            "|---|" + "---|" * (len(labels) + len(base)),
        ]
        spec = [
            ("抽出件数", "item_count", lambda s: s["item_count"]),
            ("stated_reason が非 null", "stated_reason", lambda s: s["non_null"]["stated_reason"]),
            ("applies_when が非 null", "applies_when", lambda s: s["non_null"]["applies_when"]),
            ("pass_criterion が非 null", "pass_criterion", lambda s: s["non_null"]["pass_criterion"]),
            ("文字起こしに無い語の数", "novel", lambda s: s["novel_term_count"]),
        ]
        for name, key, pick in spec:
            cells = [str(pick(results[l])) for l in labels] + [str(STRATA_BASELINE[b][key]) for b in base]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines += [
            f"基準は会議 {BASELINE_CONVERSATION} のものなので、この会議（{conversation_id}）とは比べられない。",
            "",
        ]

    lines += ["## 人が確認する2点", ""]
    for label in labels:
        checks = results[label]["human_checks"]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"1. 「簡単フォーイス」の発話（quote に「{UNCLEAR_QUOTE_MARK}」を含む項目）。"
            "期待: transcription_unclear = true、pass_criterion が null、reusable が false。"
        )
        if checks["unclear_items"]:
            for u in checks["unclear_items"]:
                lines.append(
                    f"   - transcription_unclear = {u['transcription_unclear']}、"
                    f"pass_criterion = {u['pass_criterion']!r}、reusable = {u['reusable']}"
                    f"（quote: {u['quote']}）"
                )
        else:
            lines.append("   - 該当する項目がない（抽出されなかった。quote が書き換えられた可能性もあるので、JSON を見る）")
        lines.append(f"2. applies_when に「{APPLIES_WHEN_MARK}」を含む項目（期待: 「参加者が募りたい場合」）。")
        if checks["applies_when_hits"]:
            for h in checks["applies_when_hits"]:
                lines.append(f"   - {h}")
        else:
            lines.append("   - 該当する項目がない")
        lines.append("")

    lines += [
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
    ap.add_argument(
        "--variants", type=Path, default=None, help="設定の一覧（TOML）。省略すると config.toml のままの1通り"
    )
    ap.add_argument("--out", type=Path, default=None, help="出力先のディレクトリ。省略すると data/measure")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    variants = load_variants(args.variants)
    # 測る前に、すべての設定が検査を通ることを確かめる（途中で止まって時間を無駄にしない）。
    for v in variants:
        try:
            v.applied_to(cfg)
        except ValidationError as e:
            raise SystemExit(f"設定 {v.label} は設定の検査に通りません:\n{e}") from e

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
                "model": cfg.extract.ollama.model,
                "enforces_json_schema": cfg.extract.ollama.enforces_json_schema,
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
    md_path.write_text(build_markdown(args.conversation, variants, results, cfg), encoding="utf-8")
    print(f"\n保存しました:\n  {json_path}\n  {md_path}", file=sys.stderr)

    failed = sum(s["failed_chunks"] for s in results.values())
    if failed:
        print(f"注意: 失敗したチャンクが合計 {failed} 件あります", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
