# 設計書：会話録音から判断基準を抽出する仕組み

最終更新：2026-11-03

---

## 0. この文書の位置づけ

`CLAUDE.md` は、この仕組みの**目的と制約**を書いた文書である。2026年9月に claude.ai で行った構想段階の議論の結果であり、何のために作るのか、何を入れてはいけないのか、参加者に何を伝えるのかが書いてある。

この文書は、それをどう実装したかを書いた**設計書**である。ファイルごとの役割、データベースの列、設定値の意味、そして「なぜその形にしたか」を書く。実装済みの部分は現在形で書き、未実装の部分はそう明記する。

初版（2026-10-04）は実装前の計画として書いたが、その後の変更（抽出の背面を Ollama と Claude API から Strata に置き換えた、文字起こしの復元コマンドを足した、埋め込みの持ち方を変えた）を差分で重ねたため、旧設計の痕跡が残っていた。2026-11-03 に実装に合わせて全体を書き直した。変更の履歴は13節にある。

**実装の状況**

| 対象 | 状況 |
|---|---|
| 取り込み、文字起こし、話者分離、抽出、照合、音声の削除、文字起こしの復元、推論サーバーの実測 | 実装済み。自動テスト352件が通る |
| Discord bot（`CLAUDE.md` 8章 #7） | 未着手 |
| 夜間バッチの統合とタスクスケジューラ登録（同 #8） | 未着手 |
| 実機（Windows + RTX 4070 Ti）での動作確認 | 未実施。11節に確認手段を書く |

---

## 1. 全体の流れ

音声ファイル1本が `inbox` に届いてから、チェック項目の候補になるまでの流れと、どのコマンドがどの段を担うかを示す。

| 段 | 入力 | 処理 | 出力 | コマンド |
|---|---|---|---|---|
| 1 | `inbox` の音声ファイル | ファイル名から日時・形式・種類を読み、同じ会議に属するファイルを束ねる | `conversations` の行 | `kijun ingest` |
| 2 | 束ねた音声 | faster-whisper で文字起こし、pyannote で話者分離、両者を突き合わせる | `utterances` の行、文字起こしの JSON、音声を `processed` へ移動 | `kijun transcribe` |
| 3 | `utterances` | 20〜30分のチャンクに分け、Strata で要求・指摘を抽出する | `extracted_requests` の行 | `kijun extract` |
| 4 | `extracted_requests` | 埋め込みモデルでベクトル化し、既存のチェック項目と照合する | `item_evidence` の行、または `knowledge_items` の新規候補 | `kijun match` |
| 5 | `knowledge_items` の候補 | ユーザーが理由と適用条件を書き足す | `knowledge_items` の確定 | **未実装**（Discord、`CLAUDE.md` 4-6） |

段5が未実装であることは、この仕組みの中心が未完であることを意味する。`CLAUDE.md` 1章は「この仕組みの中心は、抽出された指摘に対してユーザー自身が理由と適用条件を書き足す工程である」と書いている。段1から段4は、その工程に渡す材料を作るところまでである。

段3と段4の間に、推論サーバーの性能を測る `kijun doctor` と、同じ会議を2回抽出した結果を比べる `kijun compare` がある。どちらも流れの中に入るものではなく、設定値を決めるための道具である。

---

## 2. 設計を貫く4つの原則

この4つが、個々のファイルの形を決めている。将来この仕組みを変更するときは、まずこの4つを読んでほしい。

### 2-1. 外部に依存する処理の手前に、差し替え可能な境界を置く

GPU、外部プロセス、ネットワークに触る処理は、すべて `typing.Protocol` で定義したインターフェースの背後に置く。テストでは `tests/fakes.py` の偽の実装を差し込む。

| インターフェース | 定義場所 | 実装 | 偽の実装 |
|---|---|---|---|
| `Transcriber` | `transcribe/base.py` | `faster_whisper_impl.py` | `FakeTranscriber` |
| `Diarizer` | `transcribe/base.py` | `pyannote_impl.py` | `FakeDiarizer` |
| `Extractor` | `extract/base.py` | `strata_impl.py` | `FakeExtractor` |
| `HttpTransport` | `transport.py` | `httpx` を使う実装 | 偽の応答を返す実装 |
| `Embedder` | `match/base.py` | `e5_impl.py` | `FakeEmbedder` |

この境界を置く理由は2つある。1つは、開発と自動テストを行う環境（Anthropic のクラウド上の Linux コンテナ）に GPU も CUDA も無く、音声ファイルも1本も無いため、実装を差し込まずに論理を検査する必要があるから。もう1つは、`CLAUDE.md` 4-3 が kotoba-whisper を、4-5 が ruri を、将来の選択肢として残しているから。

### 2-2. プロセスの境界を VRAM の境界にする

`CLAUDE.md` 4-4 に「faster-whisper・pyannote と抽出用のモデルを同時に VRAM に載せない」とある。RTX 4070 Ti の VRAM は12GBで、large-v3（約3GB）と pyannote（約1GB）と 125B の MoE を同時には載せられない。

Python の中で `del model` と `torch.cuda.empty_cache()` を呼ぶ方法は採らない。PyTorch のキャッシュアロケータや CTranslate2 が確保した分が残ることがあり、残ったかどうかを確認しにくいためである。代わりに次の形にする。

- 文字起こしと話者分離を行う `transcribe` と、抽出を行う `extract` を、別の CLI サブコマンドにする。
- 夜間バッチ（未実装）は、この2つを**別々のプロセスとして順番に起動する**。プロセスが終了すれば OS が GPU メモリを解放するので、解放されたかどうかを疑う必要がなくなる。
- `transcribe` と `extract` は、Python のモジュールとしても互いを import しない。

Strata は別プロセスのサーバーなので、`transcribe` が動く間も VRAM を保持し続ける可能性がある。Strata にモデルを降ろす手段があるかは未確認なので、設定に `start_command` と `stop_command` を置いてプロセスごと止められるようにした（読む処理は夜間バッチの実装時に書く。10節参照）。

### 2-3. 確認できない前提は、設定値と実測コマンドに委ねる

Strata の API 仕様を記したサイト（`stratallm.org`、`docs.stratallm.org`）は、開発環境のネットワーク設定から到達できない。そのため次の4点が未確認である。

- OpenAI 互換のエンドポイントのパスが `/v1/chat/completions` か
- `response_format` による JSON スキーマの強制に対応しているか
- `/v1/models` の応答がどの形式か
- 応答に `usage`（トークン数）が含まれるか
- 1リクエストで受け付ける入力トークンの上限

これらを推測で決め打ちしない。すべて `config.toml` の設定値にし、既定値は**動かない可能性が最も低い安全側**に置く（たとえば `enforces_json_schema = false`、すなわち `response_format` を送らない）。そして `kijun doctor` が実機で測り、`config.toml` に書くべき値を出す。

この原則の帰結として、**確認できない背面を吸収する処理は、確認できる場所に寄せる**。具体的には、サーバーが壊れた JSON を返す可能性に対して `extract/repair.py` を置いた。これは外部に一切接続しない純粋な文字列処理なので、壊れ方の見本を使ってクラウド環境で完全にテストできる（27件）。

### 2-4. 録音は録り直せないので、失うより止める

会議の録音は再現できない。取り返しのつかない操作には、次の歯止めを置く。

- ファイル名を解析できない音声は、削除も移動もしない。`inbox` に残したまま処理対象から外し、ファイル名の一覧を出す。
- `kijun purge-audio` は既定で**削除しない**。対象の一覧を表で出すだけで、実際に削除するには `--execute` を付ける。
- `kijun restore-transcript` は、`conversations` に同じ会議の行が既にあれば**上書きせずスキップ**する。壊れたデータベースを直す操作で、正しい行を踏み潰す危険を作らないため。上書きには `--force` を付ける。
- 抽出の出力が途中で切れていて、完全な要素が1件も無い場合、`repair.py` は**修復しない**。`{"items": []}` として閉じると「抽出0件」と区別が付かず、データを失ったことが見えなくなるためである。検証で落とし、失敗として記録する。

---

## 3. 実装の構成

```
test4U/
├── CLAUDE.md                          # 目的と制約。構想段階の議論の引き継ぎと決定の記録
├── README.md                          # 導入手順、実機での確認の順序、CLI の一覧
├── docs/implementation-plan.md         # この文書
├── pyproject.toml                     # uv のプロジェクト定義。重い依存は extra に分ける
├── config.example.toml                # 設定ファイルの雛形。全項目の一次情報
├── .gitignore                         # config.toml、data/、*.duckdb を除外
├── src/kijun/
│   ├── config.py                      # config.toml の読み込みと既定値（pydantic）
│   ├── cli.py                         # サブコマンドの定義（typer）
│   ├── models.py                      # Utterance。transcribe と extract が共有する型の置き場
│   ├── deps.py                        # 重い依存の遅延 import。不足時に入れるべき extra を示す
│   ├── table.py                       # 全角文字を含む表の整形
│   ├── transport.py                   # HTTP の送受信の境界（HttpTransport）と所要時間の計測
│   ├── doctor.py                      # 推論サーバーの実測と、config.toml への推奨値
│   ├── db/
│   │   ├── schema.sql                 # 10テーブルとビュー1つ
│   │   ├── migrate.py                 # スキーマの適用とバージョン管理
│   │   └── repo.py                    # 各テーブルの読み書き、text_hash の計算
│   ├── ingest/
│   │   ├── filename.py                # yyyyMMdd_HHmm_<形式>_<種類>.m4a の解析
│   │   ├── audio.py                   # 音声の長さの取得（mutagen）
│   │   ├── bundle.py                  # 会議単位への束ね
│   │   └── run.py                     # ingest の本体
│   ├── transcribe/
│   │   ├── base.py                    # Transcriber / Diarizer
│   │   ├── faster_whisper_impl.py     # faster-whisper large-v3
│   │   ├── pyannote_impl.py           # pyannote speaker-diarization-3.1
│   │   ├── merge.py                   # 文字起こしの区間と話者区間の突き合わせ
│   │   ├── restore.py                 # 文字起こしの JSON から DB を復元する
│   │   └── run.py                     # transcribe の本体
│   ├── extract/
│   │   ├── base.py                    # Extractor と ExtractionError
│   │   ├── models.py                  # 抽出結果の pydantic モデルと JSON スキーマ
│   │   ├── prompt.py                  # 抽出プロンプトの文面（build_messages）
│   │   ├── chunk.py                   # チャンク分割、発話番号の復元、重複排除
│   │   ├── tokens.py                  # 入力の長さのトークン数の概算（係数の置き場）
│   │   ├── repair.py                  # 壊れた JSON の修復
│   │   ├── strata_impl.py             # Strata の OpenAI 互換エンドポイント
│   │   └── run.py                     # extract の本体
│   ├── match/
│   │   ├── base.py                    # Embedder
│   │   ├── e5_impl.py                 # sentence-transformers
│   │   └── link.py                    # コサイン類似度と閾値による分岐
│   ├── compare/report.py              # 同じ会議の2回の抽出の対照表
│   └── retention/purge.py             # 音声の削除
└── tests/                             # test_*.py が19ファイル、352件
```

`models.py`、`deps.py`、`table.py`、`transport.py`、`tokens.py` と各パッケージの `run.py` は、どれも1つの理由で分けてある。

- `models.py`：`Utterance` を `transcribe` と `extract` の両方が使うが、原則2-2により互いを import させたくないため、共通の置き場を作った。
- `deps.py`：重い依存を関数の中で import し、無いときに `uv sync --extra transcribe` のように入れるべき extra を示す。モジュールのトップレベルで import しないのは、`kijun --help` が依存の有無に関わらず動く必要があるため。
- `table.py`：日本語を含む表を端末で崩さずに出すため。
- `transport.py`：原則2-3により、HTTP の応答の判定を偽の応答で検査できるようにするため。所要時間の計測もここ1か所に置き、`doctor` の速度の実測で使う。
- `tokens.py`：トークン数の概算の係数を1か所に置くため。`run.py` の入力長の警告と `doctor.py` の入力の引き伸ばしが同じ値を使う。
- `run.py`：CLI の本体を `cli.py` から切り出し、偽の実装を差し込んでテストできるようにするため。

---

## 4. データ設計（DuckDB）

### 4-1. `CLAUDE.md` 5章の表からの変更点と、その理由

`CLAUDE.md` 5章は8テーブルを挙げている。実装は10テーブルとビュー1つである。変更点と理由を記す。

| 変更 | 内容 | 理由 |
|---|---|---|
| `conversations.setting` を2列に分けた | `setting_format`（`f2f` / `call`）と `setting_kind`（`teirei` など） | `CLAUDE.md` 6章の評価指標に「対面と通話それぞれの抽出精度」がある。1列に結合して入れると、集計のたびに文字列を分解することになる |
| `extracted_requests` に `speaker_from_label` と `speaker_to_label` を足し、`direction` を NULL 許容にした | 抽出の時点では話者ラベル（`SPEAKER_00` など）しか分からない | **`CLAUDE.md` 4-8 の処理順では `direction` が確定できない。** 話者と人物の対応付けを Discord に依頼するのは抽出（手順6）より後の手順8である。つまり抽出の時点で「自分→相手」か「相手→自分」かは決まらない。話者ラベルを記録しておき、`speaker_map` が埋まった後に導出する |
| `embeddings` テーブルを足した | `(owner_type, owner_id, model_name, text_hash, dim, vector, created_at)` | `CLAUDE.md` 4-5 の照合にベクトルの保存先が必要。`knowledge_items` と `extracted_requests` の両方を1つの表に入れ、`owner_type` で区別する |
| `embeddings` に `text_hash` を持たせた | ベクトル化した元テキストを NFKC 正規化した後の SHA-256（16進） | `knowledge_items.check_text` は、レビューでユーザーが文言を直す前提の列である（`CLAUDE.md` 4-6）。ハッシュが無いと、文言を直しても古いベクトルが使われ続ける。全角と半角の違いだけでは再計算しないよう NFKC 正規化を挟む |
| `schema_version` テーブルを足した | `(version, applied_at)` | スキーマを後から変更したときに、どのバージョンが適用済みか分かるようにする |
| `v_weekly_metrics` ビューを足した | 週・形式・抽出モデルごとの指標 | `CLAUDE.md` 6章の「毎週記録する指標」を SQL で出せるようにする |

### 4-2. テーブル一覧

正確な列と型は `src/kijun/db/schema.sql` を見てほしい。ここでは役割と、設計上の判断が入っている点だけを書く。

| テーブル | 役割 | 設計上の判断 |
|---|---|---|
| `schema_version` | 適用済みのスキーマのバージョン | 現在のバージョンは1 |
| `conversations` | 会議単位の管理 | `conversation_id` は束の先頭ファイルの `yyyyMMdd_HHmm_<形式>_<種類>`。`transcribed_at` は音声の削除判定（9節）が見る |
| `utterances` | 発話単位の文字起こし | 主キーは `(conversation_id, seq)`。`seq` は会議内で0から始まる通し番号。`speaker_label` は話者分離が失敗した場合 NULL |
| `extracted_requests` | 抽出した指摘・要求1件ごと | `request_id` は `{conversation_id}#{model_name}#{連番}`。同じ会議を2つの設定で抽出しても衝突しない。`stated_reason` と `applies_when` は語られていなければ NULL（6-4参照） |
| `knowledge_items` | チェック項目本体 | `status` は `candidate` / `approved` / `rejected` / `held`。`importance` は抽出の段階では NULL（判定は Discord の実装時） |
| `item_evidence` | 項目と根拠発言の対応 | 新規候補として作られた場合 `similarity` は NULL |
| `review_log` | 修正履歴 | `log_id` はシーケンス `review_log_seq` から採番。Discord の実装まで行は入らない |
| `speaker_map` | 話者ラベルと人物の対応 | `person_label` は `self` / `kouhai_a` のように英数字で持つ。Discord の実装まで行は入らない |
| `discord_posts` | Discord のメッセージと DB レコードの対応 | 同上 |
| `embeddings` | 照合用のベクトル | 主キーは `(owner_type, owner_id, model_name)`。`text_hash` を主キーに入れないのは、テキストを変えたときに行が置き換わり、古い行が溜まらないようにするため |

コサイン類似度の計算は、DuckDB の `array_cosine_similarity` ではなく Python の numpy で行う。`array_cosine_similarity` は固定長の `FLOAT[N]` 型を要求し、N をスキーマに直接書く必要があるためである。埋め込みモデルを差し替えると次元数が変わる（multilingual-e5-large は1024次元、ruri は別）ので、可変長の `FLOAT[]` で保存して計算は Python 側で行う。`knowledge_items` が数百件の規模では numpy で十分に速い。1万件を超えたら、固定長の列を別に持たせるか DuckDB の vss 拡張に切り替える。

### 4-3. 評価指標のビュー

```sql
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
    week, setting_format, model_name, conversations, requests,
    requests / nullif(conversations, 0) AS requests_per_conversation,
    reusable_requests / nullif(requests, 0) AS reusable_ratio
FROM per_model;
```

`model_name` を `GROUP BY` に入れているのは、同じ会議を2つの設定で抽出したときに会議の件数を二重に数えないためである。1つの会議を2回抽出すると2行になり、各行がその会議を1回だけ数える。抽出が1件も無い会議は `model_name` が NULL、`requests` が0の行になる。

`CLAUDE.md` 6章の指標のうち「LLM の提案文言を修正せず承認した割合」と「1日のレビュー所要時間」は、`review_log` と `discord_posts` に行が入ってからでないと計算できない。Discord の実装時にビューを足す。

### 4-4. 列挙型の値の対応

列挙型の値は英数字で持ち、画面表示のときに日本語にする。`CLAUDE.md` 4-2 がファイル名を英数字にする理由として「Windows と Python での文字化けを避けるため」を挙げているので、同じ方針をデータベースの値にも適用した。

| 列 | 保存する値 | 表示 |
|---|---|---|
| `extracted_requests.direction` | `self_to_other` / `other_to_self` | 自分→相手 / 相手→自分 |
| `extracted_requests.source_type` | `live_conversation` / `dictated_review` | 実会話 / 口述振り返り |
| `knowledge_items.status` | `candidate` / `approved` / `rejected` / `held` | 候補 / 承認 / 却下 / 保留 |

---

## 5. 取り込み：ファイル名の解析と会議単位への束ね

### 5-1. ファイル名の解析（`ingest/filename.py`）

`CLAUDE.md` 4-2 の書式 `yyyyMMdd_HHmm_<形式>_<種類>.m4a` を解析する。

```
^(?P<date>\d{8})_(?P<time>\d{4})_(?P<fmt>[a-z0-9]+)_(?P<kind>[a-z0-9]+)(?:_(?P<part>\d+))?\.(?P<ext>m4a|wav|mp3)$
```

- `fmt` が `settings.formats` に無ければ**解析を失敗させる**。形式は `f2f` と `call` の2つしかなく、ここを間違えると形式別の集計が壊れる。
- `kind` が `settings.kinds` に無い場合は、警告を出したうえで**その値のまま記録する**。`CLAUDE.md` 4-2 に「運用しながら追加」と書いてあるため、未知の種類を拒否すると運用が止まる。
- `part` は、1回の会議が複数ファイルに分かれたときにショートカット側で付く可能性のある連番（`_2` など）を許容するための任意部分。
- 解析に失敗したファイルは削除も移動もしない（原則2-4）。

### 5-2. 音声の長さ（`ingest/audio.py`）

`mutagen` で読む。pure Python で、ffmpeg のような外部コマンドを必要としない。Windows への導入を1つ減らすため ffprobe は使わない。長さが取れなければ `None` を返し、呼び出し側が警告を出す。

### 5-3. 会議単位への束ね（`ingest/bundle.py`）

`CLAUDE.md` 4-7 の規則を実装する。`recorded_at` の昇順に並べ、隣接する2ファイルが次の3条件をすべて満たすなら同じ束にする。

1. `recorded_at` の日付（ローカル日付）が同じ
2. `bundle.require_same_setting` が true の場合、`setting_format` と `setting_kind` が同じ
3. 前のファイルの終了時刻（`recorded_at` + 長さ）から次のファイルの `recorded_at` までが `bundle.gap_minutes` 分以内

条件2は `CLAUDE.md` 4-7 への追加である。同じ日の30分以内に種類の違う会議（定例の直後に雑談）が録られた場合、別の会議として扱うのが自然だからである。運用で邪魔になる可能性があるので `require_same_setting = false` で無効にできる。

長さが取れなかったファイルは、終了時刻を `recorded_at` と同じとみなして判定し、その会議に警告の印を付ける。

束ねた結果は `kijun ingest --dry-run` で表に出す。`CLAUDE.md` 4-7 は Discord に投稿して確認するとしているが、それは未実装である。

---

## 6. 文字起こしから抽出まで

### 6-1. 文字起こしと話者分離（`transcribe/`）

`Transcriber` と `Diarizer` は複数ファイルを受け取る。4-7 で束ねた会議が複数ファイルから成るためで、2本目以降の時刻は1本目の長さを足してつなげる。

**複数ファイルは1本につないでから話者分離する。** ファイルごとに pyannote を走らせると `SPEAKER_00` が別人を指してしまうためである。

pyannote には、PyAV（faster-whisper と同じデコーダー）で読んだ波形を渡す。torchaudio が Windows で m4a を読めないことがあるためである。pyannote 4.x にも対応しており、`from_pretrained` の引数名は `token` と `use_auth_token` のうち署名を見て選ぶ。

`HF_TOKEN` が未設定の場合は、`pyannote/speaker-diarization-3.1` と `pyannote/segmentation-3.0` の両方のモデルページで利用条件に同意する必要があることを含めたエラーメッセージを出して終了する。

faster-whisper の `segments` は generator なので、使う前に list にする。generator のまま次の処理に渡すと、モデルを解放した後に評価されて落ちる。

### 6-2. 区間の突き合わせ（`transcribe/merge.py`）

文字起こしの区間と話者の区間から `utterances` の行を作る。

1. 各文字起こし区間について、時間の重なりが最も長い話者区間のラベルを割り当てる。重なりが全く無ければ `None`。
2. 同じ話者ラベルが連続し、かつ前の区間の終了から次の区間の開始までの間隔が `transcribe.min_utterance_sec` 未満なら、1つの発話に結合する。話者が不明（`None`）の区間どうしは、同じ話者と判断できないので結合しない。
3. `seq` を0から順に振る。

この処理は純粋な計算なのでクラウド環境でテストできる。

### 6-3. 文字起こしの JSON と復元（`transcribe/restore.py`）

文字起こしの結果を `utterances` に書くと同時に、`paths.transcripts/<conversation_id>.json` に保存する。DuckDB を失ったときに文字起こしをやり直さずに済むようにするためである。

JSON には、`conversations` の行を再構成するのに必要な項目をすべて入れる：`conversation_id`、`recorded_at`、`setting_format`、`setting_kind`、`source_files`、`duration_sec`、`created_at`、`transcribed_at`、`utterances`。日時は `isoformat()` でマイクロ秒まで書く（秒未満を落とすと DuckDB の値と一致せず、復元しても元に戻らない）。`transcript_path` は JSON 自身のパスなので入れず、復元時に読み込んだファイルのパスを入れる。`transcribe --audio` による単発の試し実行の JSON は、会議に属さないので `conversation_id` が null になる。

`kijun restore-transcript` は `conversations` と `utterances` を復元する。`extracted_requests`、`knowledge_items`、`item_evidence` は JSON に入っていないので復元できず、`kijun extract` と `kijun match` のやり直しが必要であることを出力の最後に出す。既にある会議は上書きせずスキップする（原則2-4）。

### 6-4. 抽出プロンプト（`extract/prompt.py`）

`CLAUDE.md` 4-4 の「抽出プロンプトの要件（決定）」を満たす。入力は、話者ラベル付きの発話を `[seq] SPEAKER_00: 本文` の形で並べたもの。発話番号を行頭に入れるのは、抽出結果の `seq_from` / `seq_to` をモデルに書かせるためである。

要件のうち、プロンプトの中で繰り返し書いて守らせているのは次の2つである。

- **`stated_reason` と `applies_when` は、発言中で実際に語られた内容だけを書く。推測で補ってはならない。語られていなければ `null`。** プロンプトにはこの理由も書く。この仕組みの目的はユーザー自身が理由と適用条件を言語化することであり、LLM が推測で埋めるとその機会が失われる（`CLAUDE.md` 1章）。
- **`pass_criterion` は Yes/No で判定できる形にする。** 例として `CLAUDE.md` の「冒頭2行以内に、読み手にしてほしい行動が書かれているか」を入れる。作れない場合は `null`。

`build_messages(chunk, cfg) -> tuple[str, str]`（system と user）という1つの関数にまとめてある。背面を増やしたときに文面が分かれないようにするためで、文面が違えば2回の抽出の比較（8節）が成立しない。

### 6-5. チャンク分割（`extract/chunk.py`）

`CLAUDE.md` 4-4 の「20〜30分ごとに区切り、前後を2分程度重ねて分割処理する」を実装する。

- チャンクの境界は発話の境界に丸める。発話を途中で切らない。
- 1つの発話が `chunk_minutes` を超える長さ（長い独話）の場合は、その発話だけで1チャンクにする。
- LLM に見せる発話番号はチャンク内の0始まりで、返ってきた番号を会議全体の `seq` に復元する。チャンク範囲外の番号や逆順の番号は捨て、件数を数える。
- 重複排除：チャンク N と N+1 の重複区間に `seq_from` が収まる抽出結果のうち、後続チャンクから出たものを捨てる。保険として、`quote` の正規化（空白と句読点を除去）が一致し `seq_from` の差が2以内のものを1件にまとめる。**ただし前のチャンクが失敗している場合は捨てない。** 捨てると、その区間の結果がどちらからも得られなくなる。

**`chunk_minutes` を大きくすれば1チャンクになる。** Qwen3.8-Flash-Next のネイティブの文脈長は262,144トークンで、2時間の会議（約6万トークンの概算）は1リクエストに収まる計算である。`chunk_minutes = 180` / `overlap_minutes = 0` にすれば1チャンクになり、重複排除が不要になって会議全体を見た抽出になる。これはコードを変えずに設定だけで切り替えられる。

ただし 12GB の VRAM と 64GB のメモリで6万トークンの文脈を保持できるかは未確認なので、既定値は `chunk_minutes = 25` / `overlap_minutes = 2` のままにしてある。動かない設定を既定にしないためである。`kijun doctor` が実測し、通るなら180を推奨する。

### 6-6. 抽出の背面（`extract/base.py`、`strata_impl.py`、`transport.py`）

`Extractor` は4つのメンバーを持つ。

```python
class Extractor(Protocol):
    model_name: str              # request_id と extracted_requests.model_name に使う
    enforces_json_schema: bool   # サーバーがスキーマ適合を保証するか
    max_input_tokens: int | None # 1リクエストの入力の上限。未確認なら None
    repair_events: list[RepairEvent]  # 出力を修復した記録。run.py が extract_logs に書く
    def extract(self, chunk: Chunk) -> ExtractionResult: ...
```

`enforces_json_schema` と `max_input_tokens` を宣言させるのは、原則2-3のためである。背面ごとの挙動をハードコードせず、呼び出し側が判断できるようにする。

`strata_impl.py` は OpenAI 互換のエンドポイントに `transport.py` 経由で投げる。Anthropic 互換側を `anthropic` SDK で叩く案は採らなかった。SDK は `betas`、`fallbacks`、`messages.parse` のサーバー側構造化出力に依存しており、ローカルサーバーが実装している保証がなく、失敗したときに原因が SDK 側かサーバー側か切り分けられなくなるためである。

- `enforces_json_schema` が true のときだけ `response_format` に JSON スキーマを付ける。false のときは付けない（対応していないサーバーが 400 を返す可能性があるため）。
- 応答の本文は必ず `repair.py` を通してから検証する。失敗したら同じチャンクをもう1回だけ投げ直し、2回失敗したら `ExtractionError` を出す。会議全体は落とさず、失敗したチャンクを記録して次に進む。
- 失敗の記録先は `<db と同じディレクトリ>/extract_logs/<会議ID>__<モデル>.json`。スキーマに保存先の列が無いためファイルに置く。`compare` がこれを読む。

`transport.py` の `TransportError` は、応答そのものが得られなかった場合（接続できない、時間切れ）にだけ出す。HTTP のエラー応答（400 など）は `HttpResponse.status_code` で表す。**サーバーが機能に対応していないこと（400）と、サーバーに届かないこと（接続不可）を呼び出し側が区別できるようにするため**で、`doctor` の判定がこの区別に依存している。

### 6-7. 壊れた JSON の修復（`extract/repair.py`）

`repair_json_text(text) -> RepairOutcome` が次を順に行う。行った内容は `RepairOutcome.actions` に積み、`extract_logs` に記録する。

| 行う内容 | 定数 |
|---|---|
| 前後の空白を除去する | `ACTION_STRIPPED_WHITESPACE` |
| コードフェンス（```json）があれば中身だけを取り出す | `ACTION_REMOVED_CODE_FENCE` |
| 最初の `{` または `[` より前の散文を捨てる | `ACTION_REMOVED_PROSE_BEFORE` |
| 最後の `}` または `]` より後の散文を捨てる | `ACTION_REMOVED_PROSE_AFTER` |
| トップレベルが配列なら `{"items": [...]}` に包む | `ACTION_WRAPPED_ARRAY_IN_ITEMS` |
| 末尾が切れていれば最後の完全な要素までで閉じる | `ACTION_TRUNCATED_TO_LAST_COMPLETE_ITEM` |
| JSON が見つからない | `ACTION_NO_JSON_FOUND` |

トップレベルを配列から包み直すのは、`CLAUDE.md` 4-4 のプロンプト要件が「出力は JSON 配列のみ」であり、モデルが素直に従うと配列が返るためである。受け取る形は `{"items": [...]}` にしてある（Ollama の構造化出力がトップレベルの配列を扱いにくかったことに由来する形で、Strata でも同じ形を使っている）。

`parse_with_repair(text) -> ParsedExtraction` が、テキストから `ExtractionResult` を作る**唯一の経路**である。修復を通さずに検証する経路を残さないためである。読めなければ `UnrepairableJsonError` を出す。

要素の終わりを数えるとき、文字列の中の `}` や `]` は数えない。引用文に括弧が含まれると切り詰めの位置を誤るためで、テストで確認している。

完全な要素が1件も無いまま切れている場合は修復しない（原則2-4）。

### 6-8. 抽出結果の保存

`extracted_requests` に書く。`request_id` は `{conversation_id}#{model_name}#{連番}`。`direction` はこの時点では NULL（4-1の理由）。`source_type` は `live_conversation`。

`max_input_tokens` が `None` でないとき、チャンクの推定トークン数がそれを超えていたら警告を出す。処理は止めない。推定には `tokens.py` の係数を使う。

---

## 7. 推論サーバーの実測（`doctor.py`）

```
uv run kijun doctor [--out <パス>]
```

原則2-3の実体である。推測を測定に置き換える。固定の日本語の文を繰り返して入力を作るので、**音声が1本も無くても実行できる**。途中で失敗しても残りを続け、最後にまとめを出す。

| # | 検査 | 出す結果 |
|---|---|---|
| 1 | 接続確認 | `GET {base_url}/v1/models` の応答。モデルIDの一覧と、設定の `model` が含まれるか。応答は `data`/`id`、`models`/`name`、素の配列の3形式に対応 |
| 2 | `response_format` なしの出力 | `direct_json`（そのまま読めた）/ `repaired`（修復すれば読めた。行った修復も出す）/ `unreadable` |
| 3 | `response_format` ありの出力 | `unsupported`（400 等。対応していない）/ `enforced`（200でスキーマ適合。`enforces_json_schema = true` を推奨）/ `accepted_not_enforced`（200だが不適合。受け付けるが保証しない） |
| 4 | 速度の実測 | 約1,000トークンのものと `chunk_minutes` 相当の2回。入力・出力のトークン数と所要秒数から tokens/s。`usage` が無ければ「サーバーが usage を返さない」と出す |
| 5 | 最大入力長 | `chunk_minutes` 相当の1倍・2倍・4倍の3段階。エラーになる手前の長さを報告 |

最後に `config.toml` に書くべき推奨値（`enforces_json_schema`、`max_input_tokens`、`timeout_sec`、`chunk_minutes`、`overlap_minutes`）を出し、同じ内容を `<db と同じディレクトリ>/doctor/strata_<日時>.json` に保存する。

`chunk_minutes` の推奨値は、**実際に通った長さから算出する**。既定の `chunk_minutes = 25` では4倍でも100分相当（約5万トークン）にしかならず、2時間に届かない。120分相当以上が通れば `180` / `0` を推奨し、届かなければ通った長さの8割を5分単位に切り下げる。4倍まで通った場合は「2時間相当はまだ確かめていない。確かめるには `chunk_minutes` を30以上にして `doctor` をもう一度実行する」と出す。

段階を3つに留めているのは、実測に時間をかけすぎないためである（125B の MoE をオフロードして動かすため、1回の応答に数分から十数分かかる見込み）。

判定ロジックと推奨値の算出は、偽の `HttpTransport` を差し込んでクラウド環境でテストしている（42件）。実サーバーへの接続は `requires_network` マーカーを付けてある。

---

## 8. 同じ会議の2回の抽出の比較（`compare/report.py`）

```
uv run kijun compare --conversation <ID> [--model <名前> --model <名前>] [--out <パス>]
```

初版では `CLAUDE.md` 4-4 の「Qwen と Claude API の両方で抽出して承認件数を比較する」を支える道具だったが、2026-11-03 の決定で背面が Strata 1つになったため、**同じ Strata で設定を変えた2回の実行を比べる道具**になった。比べる対象は、量子化の違い、`chunk_minutes` の違い、`enforces_json_schema` の違いなどである。`--model` を2回指定すると比べる2つを明示でき、省略すると設定の `model` を左に置く。

**承認件数そのものは出せない。** ユーザーがレビューして初めて決まる数字であり、レビューの受け皿が Discord（未実装）だからである。出せるのは次の2つである。

機械的に計算できるもの：モデルごとの抽出件数、`reusable = true` の件数と割合、`stated_reason` / `applies_when` / `pass_criterion` が非 NULL の件数、失敗したチャンク数、両者の対応付いた件数と片方にしか無い件数（`seq_from` の差が2以内かつ `quote` の difflib の類似比0.6以上で突き合わせる）。

`stated_reason` と `applies_when` が非 NULL の件数は、**推測で補っていないかの手がかり**として見る。この値が大きすぎる場合、モデルが語られていない理由を推測で埋めている疑いがある（6-4の要件違反）。

レビュー用の対照表：`seq_from` で並べ、同じ発話に対する2つの出力を左右に並べた Markdown の表。

---

## 9. 既存項目との照合と音声の削除

### 9-1. 照合（`match/link.py`）

`CLAUDE.md` 4-5 を実装する。

1. 対象の `extracted_requests` のうち `reusable = true` のものについて、`request_summary` を `encode_queries` でベクトル化する（`reusable = false` は項目にならないので照合しない）。
2. `knowledge_items` のうち `status` が `candidate` / `approved` / `held` のものについて、`check_text` を `encode_passages` でベクトル化する。`rejected` は対象外。
3. ベクトルは `embeddings` に保存し、`model_name` と `text_hash` が一致する行があれば再計算しない。テキストを編集すると `text_hash` が変わるので再計算される。
4. 各抽出結果について最も類似度が高い既存項目を求め、`match.similarity_threshold`（既定0.85）以上なら `item_evidence` に紐付け、未満なら `knowledge_items` に `status = 'candidate'`、`version = 1` の行を作って `similarity = NULL` で紐付ける。`check_text` には `pass_criterion`（無ければ `request_summary`）を入れ、`reason` と `applies_when` には抽出結果の `stated_reason` / `applies_when` をそのまま入れる。**NULL のままになることが多く、それが正しい状態である。**
5. 照合は、関数を呼んだ時点で存在した既存項目とだけ行い、同じ呼び出しで作った新規候補とは照合しない。`knowledge_items` が0件の初回に全件が新規候補になる動作を保つためである。
6. `importance` はこの段階では NULL。3段階の判定は Discord への投稿時の処理である。

`Embedder` が `encode_queries` と `encode_passages` を分けているのは、e5 系のモデルが入力に `query: ` / `passage: ` の接頭辞を要求するためである。付け忘れると類似度の精度が落ちるので、インターフェースの段階で分けて呼び出し側が忘れられないようにした。ruri に差し替える場合は、ruri の規約に合わせて設定の接頭辞を変える。

`kijun match --dry-run` で、各抽出結果の最類似の既存項目と類似度を類似度の降順で出す。閾値を動かすとどう変わるかを見るためである。

### 9-2. 音声の削除（`retention/purge.py`）

`CLAUDE.md` 2章の参加者への説明「音声は文字起こし完了後30日で削除」を守るための実装である。

- `conversations.transcribed_at` からの経過日数が `retention.audio_days` 日以上（ちょうどを含む）の会議を対象にする。
- その会議の `source_files` のうち `paths.processed` に実在するものを削除対象とする。
- `kijun purge-audio` は既定で削除せず、対象のファイル名・会議ID・文字起こし完了日・経過日数を表で出す。削除には `--execute` を付ける（原則2-4）。
- **文字起こしの JSON は削除しない。** 削除するのは音声だけで、参加者への説明も音声についてのものである。

---

## 10. 設定

全項目とその既定値は `config.example.toml` にある。この文書に写しは置かない。初版は105行の写しを3節に置いていたが、それを保証するテストが無く、5回の変更で何度もずれたためである（`tests/test_config.py` が比べているのは `config.py` の既定値と `config.example.toml` であり、この文書ではない）。

ここには、**設計上の判断が入っている値**だけを書く。

| 設定 | 既定値 | 判断の内容 |
|---|---|---|
| `settings.formats` | `["f2f", "call"]` | ここに無い形式は解析を失敗させる（5-1） |
| `settings.kinds` | `["teirei", "sagyou", "zatsudan"]` | ここに無い種類は警告を出して受け入れる（5-1） |
| `bundle.require_same_setting` | `true` | `CLAUDE.md` 4-7 への追加条件。無効にできる（5-3） |
| `transcribe.min_utterance_sec` | `0.4` | 同一話者の連続区間を結合する間隔。話者が不明な区間どうしは結合しない（6-2） |
| `diarize.num_speakers` | `0` | TOML に null が無いため、`0` を「指定なし」として扱う |
| `extract.provider` | `"strata"` | 値は1つだが `Literal` のままにしてある。将来背面を足すときの型の受け皿 |
| `extract.chunk_minutes` | `25` | 動かない設定を既定にしないため25のまま。`doctor` の実測後に180へ変え得る（6-5） |
| `extract.strata.chat_path` | `"/v1/chat/completions"` | 未確認。`doctor` で確認してから書き換える |
| `extract.strata.enforces_json_schema` | `false` | 安全側の仮定。false のときは `response_format` を送らない（6-6） |
| `extract.strata.max_input_tokens` | `0` | `0` は未確認を意味し、入力長の検査をしない。`Extractor` の属性では `None` に変換する |
| `extract.strata.timeout_sec` | `3600` | 125B の MoE をオフロードして動かすため長く取ってある。`doctor` の実測で詰める |
| `extract.strata.start_command` / `stop_command` | `""` | 文字起こしに VRAM を明け渡すための手段。**設定に置いただけで、読む処理は未実装**（夜間バッチの実装時に書く） |
| `match.query_prefix` / `passage_prefix` | `"query: "` / `"passage: "` | e5 系が要求する接頭辞。ruri に差し替えるなら書き換える（9-1） |
| `match.similarity_threshold` | `0.85` | `CLAUDE.md` 4-5 の初期値 |
| `retention.audio_days` | `30` | `CLAUDE.md` 2章の参加者への説明と一致させる |

秘密情報は設定ファイルに書かず、環境変数で渡す。設定ファイルをうっかりコミットしたときの被害を避けるためである。現在必要なのは `HF_TOKEN`（pyannote のモデル取得）の1つだけで、`DISCORD_BOT_TOKEN` は Discord の実装時に追加する。

---

## 11. 依存関係、CLI、テスト、実機での確認

### 11-1. 依存関係と Python の版

`requires-python = ">=3.11"` としてある。開発と自動テストを行うクラウド環境の Python が3.11であり、基盤部分のテストを動かすのに足りるためである。実機の Windows には Python 3.12 を入れる（pyannote と torch の対応が確実な版）。

重い依存は用途別の extra に分けてある。クラウド環境で基盤のテストを動かすときに、torch や sentence-transformers（合計で数GB）をダウンロードさせないためである。

| extra | 入るもの |
|---|---|
| （既定） | duckdb、pydantic、typer、mutagen、numpy |
| `transcribe` | faster-whisper、pyannote.audio、torch、torchaudio |
| `extract` | httpx |
| `match` | sentence-transformers |

**torch の CUDA 版について。** `pyproject.toml` に CUDA 版のインデックス指定（`[[tool.uv.index]]` と `[tool.uv.sources]`）を書いてあるが、**コメントアウトしてある**。`uv lock` は extra を使わなくても `download.pytorch.org` に接続し、開発環境の proxy がそれを 403 で拒否して `uv sync` が失敗するためである。Windows では**コメントを外し、`cu126` というタグを現在の値に書き換えてから** `uv sync --extra transcribe` を実行する。現在のタグは https://pytorch.org/get-started/locally/ で確認する。RTX 4070 Ti は CUDA 12 系で動く。

### 11-2. CLI のサブコマンド

```
uv run kijun db init                                     # スキーマを適用する
uv run kijun ingest [--dry-run]                          # inbox を走査し、会議単位に束ねて登録
uv run kijun transcribe --conversation <ID>              # 文字起こしと話者分離
uv run kijun transcribe --audio <ファイル>                # 単発の試し実行。DB に書かない
uv run kijun restore-transcript (--conversation <ID> | --all) [--force] [--dry-run]
uv run kijun doctor [--out <パス>]                        # 推論サーバーの実測
uv run kijun extract --conversation <ID>                 # 抽出
uv run kijun compare --conversation <ID> [--model <名前>] [--out <パス>]
uv run kijun match --conversation <ID> [--provider <名前>] [--dry-run]
uv run kijun purge-audio [--execute]                     # 音声の削除
```

`transcribe --audio` は DuckDB に書かず、結果を標準出力と JSON に出すだけである。`CLAUDE.md` 8章 #4 の「1ファイルで faster-whisper と pyannote を試すスクリプト」が、登録を経ずに1本の音声で試せる必要があるためで、ユーザーが最初に実行するコマンドになる。

`match --provider` があるのは、2つの設定で抽出した場合に同じ指摘から項目が二重にできるのを防ぐためである。

### 11-3. テスト

自動テストは19ファイル・352件で、GPU とネットワークが要る6件は既定で実行しない（`pyproject.toml` の `addopts` に `-m 'not requires_gpu and not requires_network'` を書いてある）。実機では `uv run pytest -m requires_gpu` と `-m requires_network` で実行でき、試験用の音声のパスは環境変数 `KIJUN_TEST_AUDIO` で渡す。

クラウド環境で検証できるのは、設定の読み込み、ファイル名の解析、会議単位への束ね、DuckDB のスキーマと読み書き、チャンク分割と発話番号の復元、重複排除、区間の突き合わせ、壊れた JSON の修復、コサイン類似度と閾値の分岐、音声の削除の対象選定、文字起こしの書き出しと読み戻しの往復、`doctor` の判定ロジックと推奨値の算出である。

検証できないのは、faster-whisper、pyannote、Strata、sentence-transformers の実呼び出しと、torch の CUDA 版のインストールである。

### 11-4. 実機での確認の順序

| # | 作業 | 成功の判定 |
|---|---|---|
| 1 | Python 3.12 と uv を入れる | `uv --version` が出る |
| 2 | `uv sync` | `uv run kijun --help` がサブコマンド一覧を出す |
| 3 | `uv run pytest` | 352件通り、6件が除外される |
| 4 | `uv run kijun db init` | `data/kijun.duckdb` に10テーブルとビュー1つができる |
| 5 | Strata を起動し `uv run kijun doctor` | 実測の表が出る。**音声が無くても実行できる。** 出た推奨値を `config.toml` に書き写す |
| 6 | `pyproject.toml` の torch のインデックスを有効にして `uv sync --extra transcribe` | `uv run python -c "import torch; print(torch.cuda.is_available())"` が True |
| 7 | Hugging Face で pyannote の2つのモデルページの利用条件に同意し `HF_TOKEN` を設定 | 次の手順が通る |
| 8 | 5分程度の音声1本で `uv run kijun transcribe --audio <ファイル>` | 日本語の文字起こしと `SPEAKER_00` 等のラベルが出る。処理時間と VRAM 使用量を記録する（別の端末で `nvidia-smi`） |
| 9 | 2時間の音声で `uv run kijun extract` | 抽出が取れる。所要時間とメモリ使用量を記録する |
| 10 | `uv run kijun match` | 初回は全件が新規候補になる |

手順5、8、9で記録した値は、12節の未確認事項への回答になる。結果を `CLAUDE.md` 7章に書き込む。

手順1から4は、音声が1本も無くても、Strata が無くても実行できる。手順5は Strata があれば音声なしで実行できる。手順8以降は音声が必要で、`CLAUDE.md` 8章 #1〜#3（iCloud for Windows の導入、ショートカットの作成、通話の録音テスト）が終わっていることが前提である。

---

## 12. 未確認事項と、確認する手段

| 事項 | 現状 | 確認する手段 |
|---|---|---|
| Qwen3.8-Flash-Next が 12GB VRAM + 64GB メモリで動くか | **未確認。Unsloth は1bit量子化で最低75GB のメモリとしており、実機の64GB とは11GB の差がある** | `uv run kijun doctor` |
| 動く場合の tokens/s と最大入力長 | 未確認。第三者の報告は RTX 4070 で約21トークン/秒（公式の数字ではない） | 同上（手順4・5） |
| Strata が `response_format` による JSON スキーマの強制に対応するか | 未確認。既定は `false`（対応しない前提） | 同上（手順3） |
| Strata の OpenAI 互換のパス | 未確認。既定は `/v1/chat/completions` | 同上（手順1） |
| Strata の応答に `usage` が含まれるか | 未確認。3つの応答形式に対応済み | 同上（手順4） |
| Strata にモデルを VRAM から降ろす手段があるか | 未確認。無い場合は `stop_command` でプロセスごと止める | Strata のドキュメント（開発環境から到達できない） |
| トークン数の概算の係数 | 日本語1文字あたり1.5トークン、会議1分あたり500トークン。**実測ではなく概算** | `doctor` の実測で置き換える |
| pyannote.audio が Windows + CUDA で動くか | 未確認 | 実機手順8 |
| Qwen3.8-Flash-Next の量子化版の入手先とタグ | 未確認 | ユーザーが Strata のモデル一覧で確認 |
| torch の CUDA のタグ（`cu126` か別か） | 未確認 | https://pytorch.org/get-started/locally/ |
| 通話を PC スピーカー出力で録音したときに相手の声が十分に入るか | 未確認（`CLAUDE.md` 7章） | 30分の実通話テスト |
| `artifact_type` と `counterpart_role` の値の一覧 | LLM が自由な文字列を入れる。候補を固定していない | 3回分の抽出結果を見て、よく出る値を `config.toml` に列挙する |

未実装の機能は次のとおりである。

| 機能 | 必要になる理由 |
|---|---|
| Discord bot（`CLAUDE.md` 4-6、8章 #7） | 段5（ユーザーが理由と適用条件を書き足す工程）の受け皿。これが無いと `CLAUDE.md` 6章の「LLM の提案文言を修正せず承認した割合」も「1日のレビュー所要時間」も測れない |
| 夜間バッチの統合（同 #8） | 毎晩の自動実行。`start_command` / `stop_command` を読む処理もここで書く |
| `direction` の導出 | `speaker_map` が埋まってから。Discord の実装が前提 |
| `importance` の3段階判定 | Discord への投稿時の処理 |

---

## 13. 変更の履歴

### 2026-11-03：抽出の背面を Strata + Qwen3.8-Flash-Next 1つにした

ユーザーの決定により、`ollama_impl.py` と `claude_impl.py` を削除し、`provider` を `strata` のみにした。`anthropic` の依存と `ANTHROPIC_API_KEY` は不要になった。あわせて `Extractor` に能力の宣言（`enforces_json_schema`、`max_input_tokens`、`repair_events`）を足し、`repair.py`、`transport.py`、`tokens.py`、`doctor.py` を追加した。

**この変更で失われた判断手順。** `CLAUDE.md` 4-4 は「Qwen の承認件数が Claude の8割以上なら Qwen に一本化し、未満なら Claude API を使う」としていたが、比較対象の Claude API が無くなったため、この手順は使えない。代わりの判断根拠は `CLAUDE.md` 6章の「LLM の提案文言を修正せず承認した割合」であり、その測定には Discord の実装が必要である。

**記録として残す Claude API の費用。** 抽出には使わないが、`CLAUDE.md` 7章の未検証事項「Claude API の最新単価」への回答として記録する。2026-10-04 時点の Claude Sonnet 5.5（`claude-sonnet-5-5`）は入力 $2.00 / 出力 $10.00（各100万トークン）。2時間の会議1回の抽出は、入力約6万トークンで $0.12、出力約1.5万トークンで $0.15、合計約 $0.27（1ドル150円換算で約40円）の概算だった。`CLAUDE.md` 4-4 の「1回数十円程度」という概算は妥当だった。

### 2026-10-04〜11-03：実装中に見つかった設計の不備4点

| # | 不備 | 対応 |
|---|---|---|
| 1 | `CLAUDE.md` 4-8 の処理順では `direction` が確定できない | 話者ラベルを記録し、`speaker_map` 確定後に導出する設計に変更（4-1） |
| 2 | `v_weekly_metrics` が会議を二重に数えていた。2つの設定で抽出する期間にちょうど発生する | `model_name` を `GROUP BY` に入れた（4-3） |
| 3 | 文字起こしの JSON を保存しておきながら、読み込み直すコマンドが無かった。DB を失うと文字起こしをやり直すしかなかった | `kijun restore-transcript` を追加（6-3）。JSON に不足していた4項目も追加し、日時がマイクロ秒まで入るよう直した |
| 4 | `embeddings` にテキストのハッシュが無く、`check_text` を編集しても古いベクトルが使われ続けた | `text_hash` を追加（4-1） |

### 2026-10-04：初版

`CLAUDE.md` 8章の作業表 #4・#5・#6 と、ユーザーが追加を決めた #9（音声の30日削除）の実装計画として書いた。抽出の背面は Ollama + Qwen3 14B を主候補、Claude API を比較対象としていた。
