# 実装計画（第1弾：作業表 #4・#5・#6・#9）

作成日：2026-10-04
対象：`CLAUDE.md` 8章の作業表のうち #4（文字起こしと話者分離）、#5（Qwen と Claude の抽出比較）、#6（DuckDB のスキーマ）、#9（音声の30日削除）と、これらが共有する基盤。
対象外：#7（Discord bot）、#8（夜間バッチの統合とタスクスケジューラ登録）。第2弾に回す。

この文書は、実装を担当する者が設定値とファイル名のレベルまで迷わずに書ける水準で書いてある。

---

## 0. 実行環境の前提と、検証できる範囲・できない範囲

この計画を書いている Claude Code のセッションは、Anthropic のクラウド上の Linux コンテナで動いている。GPU はなく、CUDA もなく、iPhone も iCloud Drive も Discord のトークンもなく、録音された音声ファイルも1本もない。したがって、次の区別が実装計画そのものに影響する。

**クラウド側（Claude Code）で書いて、かつ自動テストで動作を確認できるもの**

| 対象 | 確認方法 |
|---|---|
| 設定ファイルの読み込みと既定値 | pytest |
| ファイル名 `yyyyMMdd_HHmm_<形式>_<種類>.m4a` の解析 | pytest |
| 複数ファイルを会議単位に束ねる処理（CLAUDE.md 4-7） | pytest |
| DuckDB のスキーマ適用と、各テーブルへの書き込み・読み出し | pytest（DuckDB は Linux でも同じファイル形式で動く） |
| 文字起こし結果を20〜30分のチャンクに分割し、抽出結果の発話番号を元に戻す処理 | pytest |
| チャンクの重複区間から同じ指摘が2回出たときの重複排除 | pytest |
| 抽出結果と既存項目のコサイン類似度の計算、閾値による分岐（CLAUDE.md 4-5） | pytest（埋め込みモデルの代わりに、文字列から決定的にベクトルを作る偽の実装を使う） |
| 音声の30日削除の対象選定と、削除前の一覧表示 | pytest |
| LLM に渡す抽出プロンプトの文面と、出力を受け取る JSON スキーマの定義 | pytest（スキーマの検証のみ。LLM は呼ばない） |

**クラウド側では書けるが、動作確認はユーザーの Windows PC でしかできないもの**

| 対象 | 理由 | ユーザーが実行する確認コマンド |
|---|---|---|
| faster-whisper による文字起こし | GPU と large-v3 のモデルファイルが必要 | `uv run kijun transcribe <音声ファイル>` |
| pyannote.audio による話者分離 | GPU、Hugging Face のトークン、モデルページでの利用条件への同意が必要 | 同上 |
| Ollama（Qwen）による抽出 | Ollama の常駐プロセスとモデルファイルが必要 | `uv run kijun extract --provider ollama --conversation <ID>` |
| Claude API による抽出 | API キーと課金が必要 | `uv run kijun extract --provider claude --conversation <ID>` |
| sentence-transformers による埋め込み | モデルファイルのダウンロードが必要 | `uv run kijun match --conversation <ID>` |
| torch の CUDA 版のインストール | Windows + CUDA の組み合わせ | `uv run python -c "import torch; print(torch.cuda.is_available())"` |

この区別があるため、GPU・ネットワーク・外部プロセスに触る処理はすべて、呼び出し側から差し替えられる形（Python の `typing.Protocol`）にする。テストでは偽の実装を差し込む。偽の実装は `tests/fakes.py` に置く。

pytest のマーカーを2つ用意する。`requires_gpu` と `requires_network` を付けたテストは既定で実行しない。`pyproject.toml` に `addopts = "-m 'not requires_gpu and not requires_network'"` を書く。クラウド側では `uv run pytest` がすべて通ることを完了条件とする。

---

## 1. VRAM の制約への対応方法：プロセスの境界を VRAM の境界にする

CLAUDE.md 4-4 に「faster-whisper・pyannote と Qwen を同時に VRAM に載せない。文字起こし・話者分離の後に GPU メモリを解放してから Ollama のモデルを読み込む」とある。RTX 4070 Ti の VRAM は12GBで、large-v3（約3GB）＋ pyannote（約1GB）＋ Qwen3 14B の4bit量子化（約9GB）は同時には載らない。

Python の中で `del model` と `torch.cuda.empty_cache()` を呼ぶ方法は、PyTorch のキャッシュアロケータや CTranslate2 の確保分が残ることがあり、残ったかどうかを確認しにくい。そこで次の設計にする。

- 文字起こしと話者分離を行う `transcribe` と、抽出を行う `extract` を、別の CLI サブコマンドにする。
- 夜間バッチ（第2弾の #8）は、この2つを**別々のプロセスとして順番に起動する**。プロセスが終了すれば OS が GPU メモリを確実に解放するので、解放されたかどうかを疑う必要がなくなる。
- Ollama はもともと別プロセスなので、`/api/chat` のリクエストに `keep_alive: 0` を付け、1回の抽出が終わったらモデルをアンロードさせる。これで次の `transcribe` が VRAM を使える。

つまり「プロセスの境界 = VRAM の境界」とする。この方針により、`transcribe` と `extract` は Python のモジュールとしても互いを import しない。

---

## 2. ディレクトリ構成

パッケージ名は `kijun` とする（判断基準の「基準」）。

```
test4U/
├── CLAUDE.md                        # 引き継ぎ文書。決定の記録を追記済み
├── README.md                        # 導入手順と、ユーザーが実行するコマンド一覧
├── docs/
│   └── implementation-plan.md       # この文書
├── pyproject.toml                   # uv のプロジェクト定義。依存は用途別の extra に分ける
├── config.example.toml              # 設定ファイルの雛形。実ファイル config.toml は .gitignore
├── .gitignore
├── src/kijun/
│   ├── __init__.py
│   ├── config.py                    # config.toml の読み込みと既定値
│   ├── cli.py                       # サブコマンドの定義（typer）
│   ├── db/
│   │   ├── __init__.py
│   │   ├── schema.sql               # 5章の8テーブル＋schema_version＋embeddings
│   │   ├── migrate.py               # スキーマの適用とバージョン管理
│   │   └── repo.py                  # 各テーブルへの書き込み・読み出し関数
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── filename.py              # ファイル名の解析
│   │   ├── audio.py                 # 音声の長さの取得（mutagen）
│   │   └── bundle.py                # 会議単位への束ね（4-7）
│   ├── transcribe/
│   │   ├── __init__.py
│   │   ├── base.py                  # Protocol: Transcriber, Diarizer
│   │   ├── faster_whisper_impl.py   # faster-whisper large-v3
│   │   ├── pyannote_impl.py         # pyannote speaker-diarization-3.1
│   │   ├── merge.py                 # 文字起こしの区間と話者区間の突き合わせ
│   │   └── restore.py               # 文字起こしの JSON から conversations と utterances を復元（6-5）
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── base.py                  # Protocol: Extractor
│   │   ├── models.py                # 抽出結果の pydantic モデルと JSON スキーマ
│   │   ├── prompt.py                # 抽出プロンプトの文面
│   │   ├── chunk.py                 # チャンク分割と、発話番号の復元、重複排除
│   │   ├── ollama_impl.py           # Ollama の /api/chat
│   │   └── claude_impl.py           # Claude API（anthropic SDK）
│   ├── match/
│   │   ├── __init__.py
│   │   ├── base.py                  # Protocol: Embedder
│   │   ├── e5_impl.py               # sentence-transformers
│   │   └── link.py                  # コサイン類似度と閾値による分岐
│   ├── compare/
│   │   ├── __init__.py
│   │   └── report.py                # Qwen と Claude の抽出結果の比較表（#5）
│   └── retention/
│       ├── __init__.py
│       └── purge.py                 # 音声の30日削除（#9）
└── tests/
    ├── conftest.py                  # 一時的な DuckDB ファイルを作る fixture
    ├── fakes.py                     # FakeTranscriber / FakeDiarizer / FakeExtractor / FakeEmbedder
    ├── test_config.py
    ├── test_filename.py
    ├── test_bundle.py
    ├── test_db_schema.py
    ├── test_chunk.py
    ├── test_extract_models.py
    ├── test_link.py
    ├── test_compare_report.py
    └── test_purge.py
```

---

## 3. 設定ファイル（`config.example.toml`）

設定の読み込み順は、(1) `config.toml` の値、(2) 無ければ `config.py` に書いた既定値。環境変数は秘密情報（API キーとトークン）にだけ使い、それ以外は設定ファイルに書く。秘密情報を設定ファイルに書かせない理由は、設定ファイルをうっかりコミットしたときの被害を避けるため。

```toml
[paths]
# Windows の実機では "C:/Users/<ユーザー名>/iCloudDrive/Recordings/inbox" を書く。
# 区切りは / で書く（Python では / でも動き、TOML のエスケープを避けられる）。
inbox      = "./data/inbox"
processed  = "./data/processed"
transcripts = "./data/transcripts"
db         = "./data/kijun.duckdb"

[settings]
# ファイル名の <形式>。この2つ以外は解析を失敗させる。
formats = ["f2f", "call"]
# ファイル名の <種類>。運用しながら追加する。ここに無い種類は、警告を出したうえでそのまま記録する
# （拒否しない理由は、CLAUDE.md 4-2 に「運用しながら追加」と書いてあるため）。
kinds = ["teirei", "sagyou", "zatsudan"]

[bundle]
# CLAUDE.md 4-7。前のファイルの終了時刻から次のファイルの開始時刻までが
# この分数以内で、かつ同じ日であれば、同じ conversation_id にする。
gap_minutes = 30
# 形式または種類が違うファイルは束ねない。false にすると形式・種類を無視して束ねる。
require_same_setting = true

[transcribe]
backend = "faster_whisper"
model = "large-v3"
# faster-whisper の compute_type。RTX 4070 Ti では float16 を使う。
# VRAM が足りない場合は int8_float16 にする。
compute_type = "float16"
device = "cuda"
language = "ja"
# faster-whisper の VAD（無音区間の除去）。会議の録音では有効にする。
vad_filter = true
# 同一話者の連続する区間で、前の区間の終了から次の区間の開始までの間隔がこの秒数未満なら、1つの発話に結合する。
# 話者が不明（None）の区間どうしは、同じ話者と判断できないので結合しない。
min_utterance_sec = 0.4

[diarize]
backend = "pyannote"
model = "pyannote/speaker-diarization-3.1"
device = "cuda"
# 話者数が分かっている場合はその人数を書く。0 なら pyannote が推定する（TOML には null が無いため、0 を「指定なし」として扱う）。
num_speakers = 0

[extract]
# "ollama" または "claude"
provider = "ollama"
# 1チャンクの長さ（分）。CLAUDE.md 4-4 の「20〜30分ごとに区切り」。
chunk_minutes = 25
# 前後の重複（分）。CLAUDE.md 4-4 の「前後を2分程度重ねて」。
overlap_minutes = 2

[extract.ollama]
base_url = "http://localhost:11434"
# 実機で `ollama list` を実行し、実際に入っているタグに合わせて書き換える。
# CLAUDE.md 4-4 の候補は qwen3:14b（4bit量子化で約9GB）と qwen3:30b-a3b（約18GB）。
# このクラウド環境から ollama.com は egress proxy でブロックされているため、
# 最新のタグ名は確認できていない。ユーザーが実機で確認して書き換える。
model = "qwen3:14b"
# 1回の抽出が終わったらモデルを VRAM からアンロードさせる（第1節の理由）。
keep_alive = 0
# 1リクエストの上限（秒）。25分のチャンクで余裕を持たせる。
timeout_sec = 900

[extract.claude]
model = "claude-sonnet-5-5"
# CLAUDE.md 4-4 の「Sonnet クラス」に対応するモデルID。
max_tokens = 16000
# 抽出は分類寄りの作業なので low から始め、承認件数を見て上げる。
effort = "low"
timeout_sec = 600

[match]
backend = "e5"
model = "intfloat/multilingual-e5-large"
# e5 系のモデルは入力に接頭辞が必要。付け忘れると類似度の精度が落ちる。
# ruri に差し替える場合は、ruri の規約に合わせてこの2つを書き換える。
query_prefix = "query: "
passage_prefix = "passage: "
# CLAUDE.md 4-5 の閾値。これ以上なら既存項目の根拠として紐付け、未満なら新規候補。
similarity_threshold = 0.85
device = "cuda"

[retention]
# 文字起こしが完了してからこの日数が経った音声を削除する（CLAUDE.md 2章）。
audio_days = 30
```

秘密情報は環境変数で渡す。

| 環境変数 | 用途 |
|---|---|
| `HF_TOKEN` | pyannote のモデル取得。Hugging Face のトークン |
| `ANTHROPIC_API_KEY` | Claude API |

`DISCORD_BOT_TOKEN` は第2弾で追加する。

---

## 4. DuckDB のスキーマ（作業表 #6）

CLAUDE.md 5章の8テーブルをそのまま作る。そのうえで、CLAUDE.md の表には無いが実装上必要になるものを3つ追加する。追加の理由を下に書く。

### 4-1. CLAUDE.md の表からの変更点と、その理由

| 変更 | 内容 | 理由 |
|---|---|---|
| `conversations.setting` を2列に分ける | `setting_format`（`f2f` / `call`）と `setting_kind`（`teirei` など） | CLAUDE.md 6章の評価指標に「対面と通話それぞれの抽出精度」がある。1列に結合して入れると、集計のたびに文字列を分解することになる |
| `extracted_requests` に `speaker_from_label` と `speaker_to_label` を追加し、`direction` を NULL 許容にする | 抽出の時点では話者ラベル（`SPEAKER_00` など）しか分からない | CLAUDE.md 4-8 の処理順では、話者と人物の対応付けを Discord に依頼するのは抽出（手順6）より後の手順8である。つまり抽出の時点で「自分→相手」か「相手→自分」かは確定できない。話者ラベルを記録しておき、`speaker_map` が埋まった後に `direction` を導出する |
| `embeddings` テーブルを追加 | `(owner_type, owner_id, model_name, text_hash, dim, vector FLOAT[])` | CLAUDE.md 4-5 の照合にベクトルの保存先が必要。`knowledge_items` と `extracted_requests` の両方のベクトルを1つの表に入れ、`owner_type` で区別する。`text_hash` は、ベクトル化した元テキストの NFKC 正規化後の SHA-256（16進）。`check_text` はユーザーがレビューで文言を修正する前提の列（CLAUDE.md 4-6）なので、編集されたことを検出して古いベクトルを使い続けないために持つ |
| `schema_version` テーブルを追加 | `(version INTEGER, applied_at TIMESTAMP)` | スキーマを後から変更したときに、どのバージョンが適用済みか分かるようにする |

列挙型の値は英数字で持ち、画面表示のときに日本語にする。CLAUDE.md 4-2 でファイル名を英数字にする理由として「Windows と Python での文字化けを避けるため」と書かれているので、同じ方針をデータベースの値にも適用する。対応は次のとおり。

| 列 | 保存する値 | 表示 |
|---|---|---|
| `extracted_requests.direction` | `self_to_other` / `other_to_self` | 自分→相手 / 相手→自分 |
| `extracted_requests.source_type` | `live_conversation` / `dictated_review` | 実会話 / 口述振り返り |
| `knowledge_items.status` | `candidate` / `approved` / `rejected` / `held` | 候補 / 承認 / 却下 / 保留 |

### 4-2. `schema.sql` の内容

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id VARCHAR PRIMARY KEY,   -- 例 20260920_1930_f2f_teirei
    recorded_at     TIMESTAMP NOT NULL,    -- 束の先頭ファイルの開始時刻
    setting_format  VARCHAR NOT NULL,      -- f2f / call
    setting_kind    VARCHAR NOT NULL,      -- teirei / sagyou / zatsudan / ...
    source_files    VARCHAR[] NOT NULL,    -- 束ねた音声ファイル名（processed 配下の相対パス）
    transcript_path VARCHAR,               -- 文字起こしの JSON の相対パス
    duration_sec    DOUBLE,                -- 束ねた音声の合計の長さ
    transcribed_at  TIMESTAMP,             -- 文字起こしが完了した時刻。#9 の削除判定に使う
    created_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS utterances (
    conversation_id VARCHAR NOT NULL,
    seq             INTEGER NOT NULL,      -- 会議内で 0 から始まる通し番号
    speaker_label   VARCHAR,               -- SPEAKER_00 など。話者分離が失敗した場合は NULL
    start_sec       DOUBLE NOT NULL,
    end_sec         DOUBLE NOT NULL,
    text            VARCHAR NOT NULL,
    PRIMARY KEY (conversation_id, seq)
);

CREATE TABLE IF NOT EXISTS extracted_requests (
    request_id          VARCHAR PRIMARY KEY,  -- {conversation_id}#{model_name}#{連番}
    conversation_id     VARCHAR NOT NULL,
    seq_from            INTEGER NOT NULL,
    seq_to              INTEGER NOT NULL,
    speaker_from_label  VARCHAR,              -- 要求した側の話者ラベル
    speaker_to_label    VARCHAR,              -- 要求された側の話者ラベル
    direction           VARCHAR,              -- self_to_other / other_to_self。speaker_map 確定後に埋める
    counterpart_role    VARCHAR,
    artifact_type       VARCHAR,              -- 成果物の種類。例 announcement / report / analysis
    quote               VARCHAR NOT NULL,     -- 発言の引用。文字起こしの文面をそのまま入れる
    request_summary     VARCHAR NOT NULL,
    stated_reason       VARCHAR,              -- 発言中で語られた理由のみ。語られていなければ NULL
    applies_when        VARCHAR,              -- 発言中で語られた適用条件のみ。語られていなければ NULL
    reusable            BOOLEAN NOT NULL,
    pass_criterion      VARCHAR,              -- Yes/No で判定できる基準。作れなければ NULL
    source_type         VARCHAR NOT NULL,     -- live_conversation / dictated_review
    model_name          VARCHAR NOT NULL,     -- 抽出に使ったモデル。#5 の比較で使う
    extracted_at        TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_items (
    item_id         VARCHAR PRIMARY KEY,
    artifact_type   VARCHAR,
    check_text      VARCHAR NOT NULL,
    pass_criterion  VARCHAR,
    reason          VARCHAR,               -- ユーザーが後から書き足す。抽出時は NULL
    applies_when    VARCHAR,               -- ユーザーが後から書き足す。抽出時は NULL
    not_applies_when VARCHAR,
    importance      INTEGER CHECK (importance BETWEEN 1 AND 3),
    status          VARCHAR NOT NULL,      -- candidate / approved / rejected / held
    version         INTEGER NOT NULL,
    updated_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS item_evidence (
    item_id    VARCHAR NOT NULL,
    request_id VARCHAR NOT NULL,
    similarity DOUBLE,                     -- 新規候補として作られた場合は NULL
    linked_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (item_id, request_id)
);

CREATE TABLE IF NOT EXISTS review_log (
    log_id      BIGINT PRIMARY KEY,        -- シーケンスから採番
    item_id     VARCHAR NOT NULL,
    action      VARCHAR NOT NULL,          -- approve / reject / hold / edit_reason / ...
    before      VARCHAR,
    after       VARCHAR,
    reviewed_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS speaker_map (
    conversation_id VARCHAR NOT NULL,
    speaker_label   VARCHAR NOT NULL,
    person_label    VARCHAR NOT NULL,      -- self / kouhai_a のように英数字で持つ
    PRIMARY KEY (conversation_id, speaker_label)
);

CREATE TABLE IF NOT EXISTS discord_posts (
    message_id      VARCHAR PRIMARY KEY,
    post_type       VARCHAR NOT NULL,      -- new_candidate / contradiction / speaker_sample / bundle_result
    item_id         VARCHAR,
    conversation_id VARCHAR,
    posted_at       TIMESTAMP NOT NULL,
    processed_at    TIMESTAMP              -- 翌晩のバッチで読み取った時刻
);

CREATE TABLE IF NOT EXISTS embeddings (
    owner_type VARCHAR NOT NULL,           -- knowledge_item / extracted_request
    owner_id   VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    text_hash  VARCHAR NOT NULL,           -- ベクトル化した元テキストの NFKC 正規化後の SHA-256（16進）。テキストの編集を検出する
    dim        INTEGER NOT NULL,
    vector     FLOAT[] NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (owner_type, owner_id, model_name)
);

CREATE SEQUENCE IF NOT EXISTS review_log_seq START 1;
```

コサイン類似度の計算は、DuckDB の `array_cosine_similarity` ではなく Python の numpy で行う。理由は、`array_cosine_similarity` が固定長の `FLOAT[N]` 型を要求し、N をスキーマに直接書く必要があるため。埋め込みモデルを差し替えると次元数が変わる（multilingual-e5-large は1024次元、ruri は別）ので、スキーマを書き換えずに済む可変長の `FLOAT[]` で保存し、計算は Python 側で行う。パイロットの規模（`knowledge_items` が数百件）では numpy で十分に速い。項目数が1万件を超えたら、固定長の列を別に持たせるか DuckDB の vss 拡張に切り替える。

### 4-3. 6章の評価指標のためのビュー

CLAUDE.md 6章の「毎週記録する指標」を SQL で出せるように、`schema.sql` の末尾にビューを作る。

ビューは `model_name` ごとに行を分ける。パイロットの最初の3週は同じ会議を Qwen と Claude の両方で抽出する（CLAUDE.md 4-4）ので、`model_name` で分けないと、その会議の件数を二重に数えてしまい、指標が必要な時期に壊れるため。

```sql
-- 会議ごと・抽出モデルごとの抽出件数と reusable の割合、形式別の比較に使う。
-- model_name で分けるのは、Qwen と Claude の両方で抽出した会議が二重に数えられないようにするため。
-- 1つの会議を2モデルで抽出すると2行になり、各行がその会議を1回だけ数える。
-- 抽出が1件も無い会議は、model_name が NULL、requests が 0 の行になる。
CREATE OR REPLACE VIEW v_weekly_metrics AS
WITH per_model AS (
    SELECT
        date_trunc('week', c.recorded_at) AS week,
        c.setting_format,
        r.model_name,
        count(DISTINCT c.conversation_id)              AS conversations,
        count(r.request_id)                            AS requests,
        sum(CASE WHEN r.reusable THEN 1 ELSE 0 END)    AS reusable_requests
    FROM conversations c
    LEFT JOIN extracted_requests r USING (conversation_id)
    GROUP BY 1, 2, 3
)
SELECT
    week,
    setting_format,
    model_name,                                        -- 抽出していない会議では NULL
    conversations,
    requests,
    requests / nullif(conversations, 0) AS requests_per_conversation,
    reusable_requests / nullif(requests, 0) AS reusable_ratio
FROM per_model;
```

「LLM の提案文言を修正せず承認した割合」と「1日のレビュー所要時間」は `review_log` と `discord_posts` に記録が入ってからでないと計算できない。第2弾で Discord を実装したときにビューを追加する。

---

## 5. ファイル名の解析と会議単位への束ね

### 5-1. `ingest/filename.py`

CLAUDE.md 4-2 の書式 `yyyyMMdd_HHmm_<形式>_<種類>.m4a` を解析する。

```
^(?P<date>\d{8})_(?P<time>\d{4})_(?P<fmt>[a-z0-9]+)_(?P<kind>[a-z0-9]+)(?:_(?P<part>\d+))?\.(?P<ext>m4a|wav|mp3)$
```

- `fmt` が `[settings].formats` に無ければ解析を失敗させる。形式は `f2f` と `call` の2つしかなく、ここを間違えると形式別の集計が壊れるため。
- `kind` が `[settings].kinds` に無い場合は、警告を標準エラーに出したうえで、その値のまま記録する。CLAUDE.md 4-2 に「運用しながら追加」と書いてあるため、未知の種類を拒否すると運用が止まる。
- `part` は、1回の会議が複数ファイルに分かれたときにショートカット側で付く可能性のある連番（`_2` など）を許容するための任意部分。
- 解析に失敗したファイルは**削除も移動もしない**。`inbox` に残したまま、処理対象から外し、ファイル名の一覧を標準出力に出す。録音は録り直せないため、解析できないという理由で動かしてはいけない。

戻り値は `ParsedName` という dataclass（`date`, `time`, `recorded_at`, `setting_format`, `setting_kind`, `part`, `ext`, `path`）。

### 5-2. `ingest/audio.py`

音声の長さを秒で返す。実装は `mutagen` を使う。`mutagen` は pure Python で、ffmpeg のような外部コマンドを必要としない。Windows への導入を1つ減らすため、ffprobe は使わない。長さが取れなかった場合は `None` を返し、呼び出し側で警告を出す。

### 5-3. `ingest/bundle.py`

CLAUDE.md 4-7 の規則を実装する。

入力：`ParsedName` と長さのリスト。
処理：

1. `recorded_at` の昇順に並べる。
2. 隣接する2ファイルについて、次の3条件をすべて満たすなら同じ束にする。
   - `recorded_at` の日付（ローカル日付）が同じ
   - `[bundle].require_same_setting` が true の場合、`setting_format` と `setting_kind` が同じ
   - 前のファイルの終了時刻（`recorded_at` + 長さ）から次のファイルの `recorded_at` までが `[bundle].gap_minutes` 分以内
3. 束の `conversation_id` は、束の先頭ファイルの `yyyyMMdd_HHmm_<形式>_<種類>` とする。
4. 長さが取れなかったファイルは、終了時刻を `recorded_at` と同じとみなして判定し、その会議に警告の印を付ける。

条件に「形式と種類が同じ」を加えたのは CLAUDE.md 4-7 への追加である。理由は、同じ日の30分以内に種類の違う会議（定例の直後に雑談）が録られた場合、それは別の会議として扱うのが自然だから。ただし運用で邪魔になる可能性があるので `require_same_setting` で無効にできるようにする。

束ねた結果は、第2弾で Discord に投稿して確認する（CLAUDE.md 4-7）。第1弾では `uv run kijun ingest --dry-run` で標準出力に表で出す。

---

## 6. 文字起こしと話者分離（作業表 #4）

### 6-1. `transcribe/base.py`

```python
class Transcriber(Protocol):
    def transcribe(self, audio_paths: list[Path]) -> list[TranscribedSegment]: ...

class Diarizer(Protocol):
    def diarize(self, audio_paths: list[Path]) -> list[SpeakerTurn]: ...
```

`TranscribedSegment` は `(start_sec, end_sec, text)`、`SpeakerTurn` は `(start_sec, end_sec, speaker_label)`。
複数ファイルを渡せるようにするのは、4-7 で束ねた会議が複数ファイルから成るため。2本目以降の時刻は、1本目の長さを足してつなげる。

### 6-2. `transcribe/faster_whisper_impl.py`

```python
from faster_whisper import WhisperModel
model = WhisperModel(cfg.transcribe.model, device=cfg.transcribe.device,
                     compute_type=cfg.transcribe.compute_type)
segments, info = model.transcribe(str(path), language=cfg.transcribe.language,
                                  vad_filter=cfg.transcribe.vad_filter,
                                  word_timestamps=False)
```

`segments` は generator なので、使う前に list にする（generator のまま次の処理に渡すと、モデルを解放した後に評価されて落ちる）。

### 6-3. `transcribe/pyannote_impl.py`

```python
from pyannote.audio import Pipeline
import torch
pipeline = Pipeline.from_pretrained(cfg.diarize.model, use_auth_token=os.environ["HF_TOKEN"])
pipeline.to(torch.device(cfg.diarize.device))
annotation = pipeline(str(path), num_speakers=n or None)
```

`HF_TOKEN` が未設定の場合は、`pyannote/speaker-diarization-3.1` と `pyannote/segmentation-3.0` の両方のモデルページで利用条件に同意する必要があることを含めたエラーメッセージを出して終了する。CLAUDE.md 7章に「pyannote.audio が Windows + CUDA 環境で問題なく動くか」が未検証として挙がっているので、エラーメッセージは原因が分かる文面にする。

### 6-4. `transcribe/merge.py`

文字起こしの区間と話者の区間を突き合わせ、`utterances` の行を作る。

- 各 `TranscribedSegment` について、時間の重なりが最も長い `SpeakerTurn` の `speaker_label` を割り当てる。重なりが全く無い場合は `None`。
- 割り当て後、同じ `speaker_label` が連続し、かつ間隔が `min_utterance_sec` 未満の区間を1つの発話に結合する。
- `seq` は 0 から順に振る。

この処理は純粋な計算なのでクラウド側で pytest で検証できる。偽の入力（固定の区間リスト）で、重なりの判定と結合の境界を確認する。

### 6-5. 出力

- `utterances` テーブルに書き込む。
- 同じ内容を `[paths].transcripts/<conversation_id>.json` に保存する。DuckDB を壊したときに文字起こしをやり直さずに済むようにするため。`conversations.transcript_path` にこのパスを入れる。
- JSON は、`conversations` の行を再構成するのに必要な項目をすべて持つ。構造は次のとおり。日時は ISO 8601 で、DuckDB の TIMESTAMP と同じ精度（マイクロ秒）で往復できるよう、秒未満も書く。

  ```json
  {
    "conversation_id": "20260920_1930_f2f_teirei",
    "recorded_at": "2026-09-20T19:30:00",
    "setting_format": "f2f",
    "setting_kind": "teirei",
    "source_files": ["20260920_1930_f2f_teirei.m4a"],
    "duration_sec": 3600.5,
    "created_at": "2026-09-20T23:30:15.123456",
    "transcribed_at": "2026-09-21T00:10:05.987654",
    "utterances": [
      {"seq": 0, "speaker_label": "SPEAKER_00", "start_sec": 0.0, "end_sec": 4.0, "text": "..."}
    ]
  }
  ```

  `conversations.transcript_path` は JSON ファイル自身のパスなので、JSON には入れず、復元時に読み込んだファイルのパスを入れる。`duration_sec` と `speaker_label` は null になり得る。`transcribe --audio`（単発の試し実行）が書く JSON は会議に属さないので、`conversation_id` が null で、他の会議の項目も null になる。この JSON は復元の対象外。
- `uv run kijun restore-transcript` が、この JSON から `conversations` と `utterances` を復元する。
  - `conversations` に同じ `conversation_id` の行が既にあれば、上書きせずスキップし、`utterances` にも触らない（既にある正しい行を JSON で踏み潰さないため）。`--force` を付けたときだけ上書きする。
  - `utterances` は、その会議の行を削除してから JSON の内容を入れ直す。
  - `extracted_requests`、`knowledge_items`、`item_evidence` は JSON に入っていないので復元できない。復元後に `extract` と `match` をやり直す。このことをコマンドの出力にも書く。
  - データを消さずに足す操作なので、既定で実行する。`--dry-run` は、何件復元されるかを先に見るための任意の指定。
  - JSON の項目が欠けている場合は、ファイル名・会議ID・欠けた項目名を出してエラーにする（`--all` では、そのファイルだけをエラーにして他を続け、終了コードを 1 にする）。
- `conversations.transcribed_at` に完了時刻を入れる。#9 の削除判定がこの列を見る。
- 音声ファイルを `[paths].processed` に移す。

---

## 7. 抽出（作業表 #5 の前半）

### 7-1. `extract/models.py`

抽出結果1件の pydantic モデル。

```python
class ExtractedRequest(BaseModel):
    seq_from: int
    seq_to: int
    speaker_from_label: str
    speaker_to_label: str | None
    counterpart_role: str | None
    artifact_type: str | None
    quote: str
    request_summary: str
    stated_reason: str | None
    applies_when: str | None
    reusable: bool
    pass_criterion: str | None

class ExtractionResult(BaseModel):
    items: list[ExtractedRequest]
```

トップレベルを `items` を持つオブジェクトにする理由：Ollama の `format` パラメータに渡す JSON スキーマは、トップレベルが配列だと従わないことがあるため、オブジェクトで包む。CLAUDE.md 4-4 の「出力は JSON 配列のみとし、前置きの文章を付けない」という要件はプロンプト側の指示として守り、受け取る形は `{"items": [...]}` にして、Python 側で配列に展開する。この差はプロンプトの文面にも書く。

### 7-2. `extract/prompt.py`

CLAUDE.md 4-4 の「抽出プロンプトの要件（決定）」をそのまま満たす。プロンプトは日本語で書く（文字起こしが日本語のため）。要件のうち守らせる必要が特に高いのは次の2つで、プロンプトの中で繰り返し書く。

- `stated_reason` と `applies_when` は、発言中で実際に語られた内容だけを書く。推測で補ってはならない。語られていなければ `null`。
  理由をプロンプトに明示する：この仕組みの目的は、ユーザー自身が理由と適用条件を言語化することであり、LLM が推測で埋めるとユーザーが言語化する機会が失われる（CLAUDE.md 1章）。
- `pass_criterion` は Yes/No で判定できる形にする。例として CLAUDE.md の「冒頭2行以内に、読み手にしてほしい行動が書かれているか」を入れる。作れない場合は `null`。

入力は、話者ラベル付きの発話を `[seq] SPEAKER_00: 本文` の形で並べたもの。発話番号を行頭に入れるのは、抽出結果の `seq_from` / `seq_to` をモデルに書かせるため。

プロンプトは1つの関数 `build_messages(chunk, cfg) -> tuple[str, str]`（system と user）にして、Ollama と Claude の両方で同じ文面を使う。これが #5 の比較の前提になる。文面が違えば比較が成立しない。

### 7-3. `extract/chunk.py`

CLAUDE.md 4-4 の「20〜30分ごとに区切り、前後を2分程度重ねて分割処理する」を実装する。

- 入力：`utterances` のリスト、`chunk_minutes`、`overlap_minutes`。
- チャンクの境界は発話の境界に丸める。発話を途中で切らない。
- 各チャンクは `(seq_from, seq_to, utterances)` を持つ。
- 重複排除：チャンク N とチャンク N+1 の重複区間（`overlap_minutes` 分）に `seq_from` が収まる抽出結果のうち、後続チャンク（N+1）から出たものを捨てる。さらに保険として、`quote` の正規化（空白と句読点を除去）が一致し、かつ `seq_from` の差が2以内のものを同一と見なして1件にまとめる。
- 1つの発話が `chunk_minutes` を超える長さの場合（長い独話）は、そのまま1チャンクにする。分割しない。

この処理はすべて純粋な計算なのでクラウド側で pytest で検証する。テストでは、境界をまたぐ抽出結果が1件に落ちることと、`seq` がグローバルな番号として正しく復元されることを確認する。

### 7-4. `extract/ollama_impl.py`

`httpx` で `POST {base_url}/api/chat` を呼ぶ。`ollama` パッケージは使わない（依存を1つ減らすため）。

```python
payload = {
    "model": cfg.extract.ollama.model,
    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    "format": ExtractionResult.model_json_schema(),   # CLAUDE.md 4-4 の「format パラメータに JSON スキーマを渡し」
    "stream": False,
    "keep_alive": cfg.extract.ollama.keep_alive,      # 0 で処理後にアンロード（第1節）
    "options": {"temperature": 0},
}
```

`format` にスキーマを渡しても出力が壊れることはあるので、`model_validate_json` が失敗したら、同じチャンクをもう1回だけ投げ直す。2回失敗したら、そのチャンクを失敗として記録して次に進む（会議全体を落とさない）。失敗したチャンクの `seq` の範囲は標準エラーに出す。

### 7-5. `extract/claude_impl.py`

`anthropic` SDK を使う。モデルは `claude-sonnet-5-5`（CLAUDE.md 4-4 の「Sonnet クラス」に対応）。

実装時に守る点：

- 構造化出力は `client.messages.parse()` を使い、`ExtractionResult` を渡してスキーマ検証を SDK に任せる。
- `output_config={"effort": cfg.extract.claude.effort}` を渡す。Claude Sonnet 5.5 の effort の既定は `high` で、抽出は分類に近い作業なので設定の既定値は `low` にしてある。承認件数が足りなければ `medium` に上げる。
- `thinking={"type": "disabled"}` は Claude Sonnet 5.5 では 400 エラーになるので渡さない。渡さなければ adaptive で動く。
- 安全性の分類器によって応答が拒否される場合がある（HTTP 200 で `stop_reason == "refusal"`）。会議の録音には感情的なやり取りが含まれ得るので、`betas=["server-side-fallback-2026-07-01"]` と `fallbacks="default"` を付けてサーバー側のフォールバックを有効にし、`stop_reason` を読む前に `content` を触らないようにする。拒否された場合はそのチャンクを失敗として記録する。
- システムプロンプト（抽出プロンプト）はチャンク間で変わらないので `cache_control={"type": "ephemeral"}` を付ける。ただしキャッシュの最小長（モデルによって512〜4096トークン）に届かないとキャッシュされないので、`response.usage.cache_read_input_tokens` を標準エラーに出し、0 のままなら効いていないことが分かるようにする。

### 7-6. Claude API の費用

CLAUDE.md 7章の「Claude API の最新単価」に対する回答。2026-10-04 時点の Claude Sonnet 5.5（`claude-sonnet-5-5`）の単価は、入力 $2.00 / 100万トークン、出力 $10.00 / 100万トークン。

2時間の会議1回（日本語3〜4万字）の概算：

| 項目 | 見積り | 根拠 |
|---|---|---|
| 入力トークン | 約6万 | 日本語4万字は概ね4万〜6万トークン。チャンクの重複でさらに約1割増える。正確な値は `client.messages.count_tokens` で実測する |
| 入力の費用 | $0.12 | 6万 × $2.00 / 100万 |
| 出力トークン | 約1.5万 | 抽出30〜60件 × 1件あたり約250トークン |
| 出力の費用 | $0.15 | 1.5万 × $10.00 / 100万 |
| 合計 | 約 $0.27（1ドル150円換算で約40円） | |

CLAUDE.md 4-4 の「1回数十円程度」という概算は妥当だった。Batch API を使うと半額になるが、第1弾の比較では即時に結果が要るので使わない。第2弾の夜間バッチは待てる処理なので、Batch API への切り替えを費用削減の手段として検討する。

Console で月額の上限金額を設定できるので、比較を始める前に設定することをユーザーに勧める。3回分の比較で Claude 側に発生する費用は、上の概算の3倍で約120円。

### 7-7. 抽出結果の保存

`extracted_requests` に書き込む。`request_id` は `{conversation_id}#{model_name}#{連番}` とする。Qwen と Claude の両方で抽出すると同じ会議に2組の行が入るので、`model_name` を ID に含めて衝突を避ける。CLAUDE.md 5章の `model_name` 列はこのために使う。

`direction` はこの時点では NULL のままにする（第4節の理由）。`source_type` は `live_conversation` を入れる。

---

## 8. Qwen と Claude の比較（作業表 #5 の後半）

CLAUDE.md 4-4 の「最初の3回分の会議は、同じ文字起こしを Qwen と Claude API の両方で抽出し、承認件数を比較する。Qwen の承認件数が Claude の8割以上なら Qwen に一本化し、未満なら Claude API を使う」を支える道具を作る。

### 8-1. コマンド

```
uv run kijun extract --conversation <ID> --provider ollama
uv run kijun extract --conversation <ID> --provider claude
uv run kijun compare --conversation <ID> --out ./data/compare/<ID>.md
```

### 8-2. `compare/report.py` が出す内容

承認件数はユーザーがレビューして初めて決まる数字なので、第1弾のこの段階では**承認件数そのものは出せない**。出せるのは、ユーザーがレビューするための対照表と、承認を待たずに計算できる数字である。両者を分けて出す。

機械的に計算できるもの：

| 項目 | 内容 |
|---|---|
| 抽出件数 | モデルごとの件数 |
| `reusable = true` の件数と割合 | モデルごと |
| `stated_reason` が非 NULL の件数 | 推測で補っていないかの手がかり。この値が大きすぎる場合、モデルが推測で埋めている疑いがある |
| `applies_when` が非 NULL の件数 | 同上 |
| `pass_criterion` が非 NULL の件数 | モデルごと |
| 対応付いた件数 | 両モデルの抽出結果を `seq_from` の近さ（差が2以内）と `quote` の文字列類似度（difflib の比率が0.6以上）で突き合わせ、片方にしか無い件数を出す |
| 失敗したチャンク数 | モデルごと |

レビュー用の対照表：`seq_from` で並べ、同じ発話に対する Qwen の出力と Claude の出力を左右に並べた Markdown の表を出す。ユーザーはこれを見て、どちらを承認するかを決める。承認の記録は第2弾の Discord で行うので、第1弾ではこの Markdown に手で印を付ける運用になる。この制約は README に書く。

---

## 9. 既存項目との照合（CLAUDE.md 4-5）

### 9-1. `match/base.py`

```python
class Embedder(Protocol):
    dim: int
    model_name: str
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
```

`queries` と `passages` を分けるのは、e5 系のモデルが入力に `query: ` / `passage: ` の接頭辞を要求するため。接頭辞を付け忘れると類似度の精度が落ちる。インターフェースの段階で分けておくことで、呼び出し側が付け忘れられないようにする。ruri に差し替える場合は、ruri の規約に合わせて `e5_impl.py` と同じ形の実装を足し、設定で接頭辞を変える。

### 9-2. `match/link.py`

1. 対象の `extracted_requests` のうち `reusable = true` のものについて、`request_summary` を `encode_queries` でベクトル化する（`reusable = false` は項目にならないので照合しない）。
2. `knowledge_items` のうち `status` が `candidate` / `approved` / `held` のものについて、`check_text` を `encode_passages` でベクトル化する。`rejected` は照合対象から外す。
3. ベクトルは `embeddings` テーブルに保存し、2回目以降は再計算しない。保存済みとみなすのは、`model_name` が設定と一致し、かつ `text_hash` が現在のテキストから計算したハッシュと一致する行だけである。`check_text` はユーザーがレビューで修正する前提の列なので、編集されていれば（`text_hash` が一致しなければ）再計算して上書きする。`text_hash` は、元テキストを NFKC で正規化したうえでの SHA-256 の16進文字列で、計算は `repo.text_hash` の1か所で行う（全角と半角の違いだけではベクトルを作り直さないため）。
4. 各抽出結果について、最も類似度が高い既存項目を求める。
   - 類似度が `[match].similarity_threshold`（既定 0.85）以上なら、`item_evidence` に `(item_id, request_id, similarity)` を入れる。
   - 未満なら、`knowledge_items` に `status = 'candidate'`、`version = 1` の行を新しく作り、`check_text` には `pass_criterion`（無ければ `request_summary`）を入れ、`item_evidence` に `similarity = NULL` で紐付ける。`reason` と `applies_when` は、抽出結果の `stated_reason` / `applies_when` をそのまま入れる（LLM が推測で埋めていないので、NULL のままになることが多い。それが正しい状態である）。
   - `importance` は、この段階では NULL にする。CLAUDE.md 4-6 の重要度の3段階判定は Discord への投稿時に行う処理なので、第2弾で実装する。
5. `knowledge_items` が0件の初回は、全件が新規候補になる。これは正常な動作である。

閾値 0.85 は CLAUDE.md 4-5 の初期値。閾値の調整を助けるために、`uv run kijun match --conversation <ID> --dry-run` で、各抽出結果の最類似の既存項目と類似度の一覧を表で出す。閾値を動かすとどう変わるかが見えるようにするため、類似度の降順で並べる。

---

## 10. 音声の30日削除（作業表 #9）

CLAUDE.md 2章の参加者への説明「音声は文字起こし完了後30日で削除」を守るための実装。

`retention/purge.py`：

- `conversations.transcribed_at` が `[retention].audio_days` 日より前の会議を対象にする。
- その会議の `source_files` に挙がっているファイルのうち、`[paths].processed` に実在するものを削除対象とする。
- `uv run kijun purge-audio` は既定で**削除せず**、対象のファイル名・会議ID・文字起こし完了日・経過日数の一覧を表で出す。
- 実際に削除するには `--execute` を付ける。削除したファイルは標準出力に1行ずつ出す。
- 文字起こしの JSON（`[paths].transcripts`）は削除しない。削除するのは音声だけである。参加者への説明も音声についてのものである。

既定を「表示のみ」にするのは、録音が録り直せないため。`--execute` を明示的に付けさせる。

---

## 11. `pyproject.toml` と依存関係

```toml
[project]
name = "kijun"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=1.1",
    "pydantic>=2.9",
    "typer>=0.12",
    "mutagen>=1.47",
    "numpy>=2.0",
]

[project.optional-dependencies]
transcribe = ["faster-whisper>=1.0", "pyannote.audio>=3.3", "torch>=2.4", "torchaudio>=2.4"]
extract-ollama = ["httpx>=0.27"]
extract-claude = ["anthropic>=0.40"]
match = ["sentence-transformers>=3.0"]

[dependency-groups]
dev = ["pytest>=8.0"]

[project.scripts]
kijun = "kijun.cli:app"

[tool.pytest.ini_options]
addopts = "-m 'not requires_gpu and not requires_network'"
markers = [
    "requires_gpu: GPU と大きなモデルファイルが必要。クラウド環境では実行しない",
    "requires_network: 外部サービス（Ollama、Claude API、Hugging Face）への通信が必要",
]
```

`requires-python` を `>=3.11` にする理由：このクラウド環境の Python が 3.11 であり、基盤部分のテストを動かすのに 3.11 で足りる。実機の Windows には Python 3.12 を入れることを README に書く（pyannote と torch の対応が確実な版）。

依存を用途別の extra に分ける理由：クラウド側で基盤のテストを動かすときに、torch や sentence-transformers（合計で数GB）をダウンロードさせないため。

torch の CUDA 版について。Windows で `uv sync --extra transcribe` をそのまま実行すると CPU 版の torch が入り、`torch.cuda.is_available()` が False になる。CUDA 版を入れるには `pyproject.toml` に次を書く必要がある。

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]
torchaudio = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]
```

`cu126` というタグが現時点で正しいかは、ユーザーの PC で確認が必要である。この計画を書いている時点の知識が古い可能性があるため、README に「https://pytorch.org/get-started/locally/ で現在のタグを確認して書き換える」手順を書く。RTX 4070 Ti は CUDA 12 系で動く。

---

## 12. CLI のサブコマンド

```
uv run kijun db init                                   # スキーマを適用する
uv run kijun ingest [--dry-run]                        # inbox を走査し、会議単位に束ねて conversations に登録
uv run kijun transcribe --conversation <ID>            # 文字起こしと話者分離。utterances に書く
uv run kijun transcribe --audio <ファイル>              # 単発の試し実行（作業表 #4 の用途）
uv run kijun restore-transcript (--conversation <ID> | --all) [--force] [--dry-run]
                                                       # 文字起こしの JSON から conversations と utterances を復元
uv run kijun extract --conversation <ID> --provider {ollama,claude}
uv run kijun compare --conversation <ID> [--out <パス>]  # Qwen と Claude の対照表（作業表 #5）
uv run kijun match --conversation <ID> [--dry-run]     # 既存項目との照合
uv run kijun purge-audio [--execute]                   # 音声の30日削除
```

`transcribe` に `--audio` を用意するのは、CLAUDE.md 8章 #4 の「1ファイルで faster-whisper と pyannote を試すスクリプト」が、DuckDB への登録を経ずに1本の音声で試せる必要があるため。`--audio` を使った場合は DuckDB には書かず、結果を標準出力と JSON ファイルに出すだけにする。これがユーザーが最初に実行するコマンドになる。

---

## 13. ユーザーが実機で行う確認の順序

README にこの順序で書く。

| # | 作業 | 成功の判定 |
|---|---|---|
| 1 | Python 3.12 と uv を入れる | `uv --version` が出る |
| 2 | `uv sync` | `uv run kijun --help` がサブコマンド一覧を出す |
| 3 | `uv run pytest` | すべて通る（GPU とネットワークが要るテストはスキップされる） |
| 4 | `uv run kijun db init` | `data/kijun.duckdb` ができ、テーブルが10個（CLAUDE.md 5章の8テーブルに `schema_version` と `embeddings` を足したもの）と、ビュー `v_weekly_metrics` が1つある |
| 5 | `pyproject.toml` の torch のインデックスを確認して `uv sync --extra transcribe` | `uv run python -c "import torch; print(torch.cuda.is_available())"` が True |
| 6 | Hugging Face で pyannote の2つのモデルページの利用条件に同意し、`HF_TOKEN` を設定 | 次の手順が通る |
| 7 | 5分程度の音声1本で `uv run kijun transcribe --audio <ファイル>` | 日本語の文字起こしと `SPEAKER_00` 等のラベルが出る。処理時間と VRAM 使用量を記録する（CLAUDE.md 7章の未検証事項） |
| 8 | `ollama list` で Qwen のタグを確認し、`config.toml` に書く | — |
| 9 | 2時間の音声で `uv run kijun extract --provider ollama` | 抽出が JSON として取れる。VRAM が足りるかを記録する（CLAUDE.md 7章の未検証事項） |
| 10 | `ANTHROPIC_API_KEY` を設定し、Console で月額上限を設定して `--provider claude` | 抽出が取れる。実際の費用を記録し、7-6 の概算と比べる |
| 11 | `uv run kijun compare` | 対照表の Markdown が出る |

手順7と9で記録した処理時間と VRAM 使用量は、CLAUDE.md 7章の未検証事項「Qwen の最新版と、VRAM 12GB での実際のメモリ使用量・処理時間」と「pyannote.audio が Windows + CUDA 環境で問題なく動くか」への回答になる。結果を CLAUDE.md 7章に書き込む。

---

## 14. この計画で解決していない事項

| 事項 | 現状 | いつ決めるか |
|---|---|---|
| Qwen の具体的なタグ名 | `qwen3:14b` を設定の既定値にしてあるが、ollama.com がこのクラウド環境からブロックされているため最新版を確認できていない | ユーザーが実機で `ollama list` を実行したとき |
| torch の CUDA のタグ（`cu126` か別か） | 推測で書いてある | ユーザーが pytorch.org で確認したとき |
| `artifact_type` の値の一覧 | LLM が自由な文字列を入れる。候補を固定していない | 3回分の抽出結果を見て、よく出る値を `config.toml` に列挙する |
| `counterpart_role` の値の一覧 | 同上 | 同上 |
| 重要度の3段階判定 | 第2弾（Discord への投稿時に行う処理） | #7 の実装時 |
| `direction` の確定 | `speaker_map` が埋まらないと決まらない。第1弾では NULL のまま | #7 で話者の対応付けを実装したとき |
| 承認件数の記録 | 第1弾では Markdown に手で印を付ける運用 | #7 の実装時 |
| 1つの会議が複数ファイルに分かれたときの束ね結果の確認 | 第1弾では `ingest --dry-run` の標準出力 | #7 で Discord への投稿を実装したとき |
