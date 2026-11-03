# 設計書：会話録音から判断基準を抽出する仕組み

最終更新：2026-11-03

---

## 0. この文書の位置づけ

`CLAUDE.md` は、この仕組みの**目的と制約**を書いた文書である。2026年9月に claude.ai で行った構想段階の議論の結果であり、何のために作るのか、何を入れてはいけないのか、参加者に何を伝えるのかが書いてある。

この文書は、それをどう実装したかを書いた**設計書**である。ファイルごとの役割、データベースの列、設定値の意味、そして「なぜその形にしたか」を書く。実装済みの部分は現在形で書き、未実装の部分はそう明記する。

初版（2026-10-04）は実装前の計画として書いたが、その後の変更（抽出の背面を Ollama と Claude API から Strata に置き換えた、文字起こしの復元コマンドを足した、埋め込みの持ち方を変えた）を差分で重ねたため、旧設計の痕跡が残っていた。2026-11-03 に実装に合わせて全体を書き直した。同じ日に第2弾（Discord でのレビューと夜間バッチ）を実装し、仕様書（`docs/design-phase2.md`、統合して削除した）の内容を14節と15節に移した。変更の履歴は13節にある。

**実装の状況**

| 対象 | 状況 |
|---|---|
| 取り込み、文字起こし、話者分離、抽出、照合、音声の削除、文字起こしの復元、推論サーバーの実測（第1弾） | 実装済み |
| Discord bot（`CLAUDE.md` 8章 #7）、`direction` の導出、重要度の判定、矛盾の検出（第2弾。14節） | 実装済み。実際の Discord への接続は未確認 |
| 夜間バッチの統合とタスクスケジューラ登録（同 #8。15節） | 実装済み。実機での実行とタスクスケジューラへの登録は未確認 |
| 自動テスト | 564件のうち555件が通る（GPU・ネットワーク・Discord のトークンが要る9件は既定で実行しない） |
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
| 5 | `knowledge_items` の候補 | Discord に投稿し、ユーザーがリアクションと返信で理由・適用条件を書き足す。翌晩のバッチが読み取って反映する | `knowledge_items` の確定、`review_log`、`speaker_map`、`direction` | `kijun discord post` / `kijun discord read`（14節） |

`CLAUDE.md` 1章は「この仕組みの中心は、抽出された指摘に対してユーザー自身が理由と適用条件を書き足す工程である」と書いている。段1から段4はその工程に渡す材料を作るところまでで、段5（第2弾）がその工程の受け皿である。段5ができたことで、`CLAUDE.md` 6章の指標のうち測れていなかった「LLM の提案文言を修正せず承認した割合」と「1日のレビュー所要時間」が、`review_log` と `discord_posts` から測れるようになった（2026-11-03 に抽出の背面を Strata 1つにしたことで失われた判断手順の代わり）。

段1から段5を毎晩自動で回すのが夜間バッチ（`kijun batch run`、15節）である。

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
| `DiscordClient` | `discord/base.py` | `discord/impl.py`（discord.py） | `FakeDiscordClient` |
| `CommandRunner` | `batch/runner.py` | `SubprocessRunner` | `FakeCommandRunner` |
| `ImportanceJudge` | `review/importance.py` | `StrataImportanceJudge` | `FakeImportanceJudge` |
| `ReplyRouter` | `review/reply.py` | `StrataReplyRouter` | `FakeReplyRouter` |
| `SpeakerReplyInterpreter` | `review/speaker.py` | `StrataSpeakerInterpreter` | `FakeSpeakerInterpreter` |

この境界を置く理由は2つある。1つは、開発と自動テストを行う環境（Anthropic のクラウド上の Linux コンテナ）に GPU も CUDA も無く、音声ファイルも1本も無いため、実装を差し込まずに論理を検査する必要があるから。もう1つは、`CLAUDE.md` 4-3 が kotoba-whisper を、4-5 が ruri を、将来の選択肢として残しているから。

### 2-2. プロセスの境界を VRAM の境界にする

`CLAUDE.md` 4-4 に「faster-whisper・pyannote と抽出用のモデルを同時に VRAM に載せない」とある。RTX 4070 Ti の VRAM は12GBで、large-v3（約3GB）と pyannote（約1GB）と 125B の MoE を同時には載せられない。

Python の中で `del model` と `torch.cuda.empty_cache()` を呼ぶ方法は採らない。PyTorch のキャッシュアロケータや CTranslate2 が確保した分が残ることがあり、残ったかどうかを確認しにくいためである。代わりに次の形にする。

- 文字起こしと話者分離を行う `transcribe` と、抽出を行う `extract` を、別の CLI サブコマンドにする。
- 夜間バッチ（15節）は、この2つを**別々のプロセスとして順番に起動する**。プロセスが終了すれば OS が GPU メモリを解放するので、解放されたかどうかを疑う必要がなくなる。
- `transcribe` と `extract` は、Python のモジュールとしても互いを import しない。

Strata は別プロセスのサーバーなので、`transcribe` が動く間も VRAM を保持し続ける可能性がある。Strata にモデルを降ろす手段があるかは未確認なので、設定に `start_command` と `stop_command` を置いてプロセスごと止められるようにした。夜間バッチがこの2つを読んで、Strata を使う手順の前後で実行する（15-2）。

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
- **判定できないリアクションは、推測で決めない。** ✅と❌が両方付いていたら、片方を優先せず、`knowledge_items` を変えず、`processed_at` を埋めず、`review_log` に記録して人に見せる（14-4）。誤って承認したチェック項目は、間違いであることが後から分かりにくいため。
- **返信の原文は、LLM に振り分けさせる前に `review_log` に残す。** 振り分けが間違っても、人が原文から直せるようにするため（14-5）。
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
│   ├── discord/                       # 第2弾: Discord でのレビュー（14節）
│   │   ├── base.py                    # DiscordClient / Reaction / Reply / DiscordError、絵文字の定数
│   │   ├── impl.py                    # discord.py を使う実装（REST のみ。呼び出しごとに接続を開閉）
│   │   ├── format.py                  # 投稿する文面（純粋な文字列処理）
│   │   ├── post.py                    # 4種類の投稿
│   │   ├── read.py                    # リアクションと返信の読み取り
│   │   └── run.py                     # discord read / post の本体、実装の組み立て
│   ├── review/
│   │   ├── llm.py                     # 抽出以外の LLM 用途が共有する Strata 呼び出し（repair を通す）
│   │   ├── importance.py              # 重要度の3段階判定
│   │   ├── reply.py                   # スレッド返信の振り分けと反映
│   │   ├── speaker.py                 # 話者の対応付けの返信の解釈
│   │   ├── direction.py               # speaker_map から direction を導出
│   │   └── contradiction.py           # 矛盾の可能性がある組の検出
│   ├── batch/                         # 第2弾: 夜間バッチ（15節）
│   │   ├── run.py                     # 手順の実行、記録、--dry-run
│   │   ├── runner.py                  # CommandRunner（subprocess の境界）
│   │   ├── strata.py                  # start_command / stop_command の窓と応答待ち
│   │   ├── pending.py                 # どの会議にどの処理が要るかを DB から調べる
│   │   └── task_xml.py                # タスクスケジューラの XML
│   ├── transport.py                   # HTTP の送受信の境界（HttpTransport）と所要時間の計測
│   ├── doctor.py                      # 推論サーバーの実測と、config.toml への推奨値
│   ├── db/
│   │   ├── schema.sql                 # 10テーブルとビュー1つ
│   │   ├── migrate.py                 # スキーマの適用、バージョン管理、旧バージョンからの移行（MIGRATIONS）
│   │   └── repo.py                    # 各テーブルの読み書き、text_hash の計算、review_log.action などの定数
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
└── tests/                             # test_*.py が31ファイル、564件（既定で555件を実行）
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
| `discord_posts` に `channel_id` と `pair_item_id` を足した（スキーマバージョン2、第2弾） | 投稿先のチャンネルID、矛盾の組のもう一方の `item_id` | 読み取りのときにどのチャンネルを見るかが要る。設定値のチャンネルIDは後から変わり得るので、投稿時の値を残す。`pair_item_id` は、既に投稿した矛盾の組を再投稿しない判定に要る（`item_id` と `pair_item_id` の両方で組を表す） |

### 4-2. テーブル一覧

正確な列と型は `src/kijun/db/schema.sql` を見てほしい。ここでは役割と、設計上の判断が入っている点だけを書く。

| テーブル | 役割 | 設計上の判断 |
|---|---|---|
| `schema_version` | 適用済みのスキーマのバージョン | 現在のバージョンは2（`migrate.SCHEMA_VERSION`）。バージョン1の DB には `kijun db init` が `ALTER TABLE` で列を足す（14-9） |
| `conversations` | 会議単位の管理 | `conversation_id` は束の先頭ファイルの `yyyyMMdd_HHmm_<形式>_<種類>`。`transcribed_at` は音声の削除判定（9節）が見る |
| `utterances` | 発話単位の文字起こし | 主キーは `(conversation_id, seq)`。`seq` は会議内で0から始まる通し番号。`speaker_label` は話者分離が失敗した場合 NULL |
| `extracted_requests` | 抽出した指摘・要求1件ごと | `request_id` は `{conversation_id}#{model_name}#{連番}`。同じ会議を2つの設定で抽出しても衝突しない。`stated_reason` と `applies_when` は語られていなければ NULL（6-4参照） |
| `knowledge_items` | チェック項目本体 | `status` は `candidate` / `approved` / `rejected` / `held`。`importance` は抽出の段階では NULL。Discord への投稿の直前に LLM が判定し（14-3）、ユーザーのリアクションが上書きする |
| `item_evidence` | 項目と根拠発言の対応 | 新規候補として作られた場合 `similarity` は NULL |
| `review_log` | 修正履歴 | `log_id` はシーケンス `review_log_seq` から採番。`action` の値は `db/repo.py` の定数1か所で定義し、定義外の値は書けない（14-9）。`item_id` が NOT NULL なので、項目に属さない出来事（話者の対応付け、束ね結果）は入らない |
| `speaker_map` | 話者ラベルと人物の対応 | `person_label` は `self` / `kouhai_a` のように英数字で持つ。ユーザーの返信から入る（14-6）。`self` が `direction` の導出の基準 |
| `discord_posts` | Discord のメッセージと DB レコードの対応 | `post_type` は `new_candidate` / `contradiction` / `speaker_sample` / `bundle_result`。`processed_at` が NULL の行が、次の `read` の対象 |
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

`CLAUDE.md` 6章の指標のうち「LLM の提案文言を修正せず承認した割合」と「1日のレビュー所要時間」は、第2弾で `review_log`（`status_changed` と `check_text_edited`）と `discord_posts`（`posted_at`）に行が入るようになったので計算できる。ビューは足していない。レビュー所要時間は `posted_at` と `reviewed_at` の差から概算できるが、ユーザーが画面を見ていた時間とは一致しないので、運用してからビューの形を決める。

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

束ねた結果は `kijun ingest --dry-run` で表に出す。`CLAUDE.md` 4-7 は Discord に投稿して確認するとしており、第2弾で複数ファイルの会議について投稿するようにした（14-2）。

---

## 6. 文字起こしから抽出まで

### 6-1. 文字起こしと話者分離（`transcribe/`）

`Transcriber` と `Diarizer` は複数ファイルを受け取る。4-7 で束ねた会議が複数ファイルから成るためで、2本目以降の時刻は1本目の長さを足してつなげる。

**複数ファイルは1本につないでから話者分離する。** ファイルごとに pyannote を走らせると `SPEAKER_00` が別人を指してしまうためである。

pyannote には、PyAV（faster-whisper と同じデコーダー）で読んだ波形を渡す。torchaudio が Windows で m4a を読めないことがあるためである。pyannote 4.x にも対応しており、`from_pretrained` の引数名は `token` と `use_auth_token` のうち署名を見て選ぶ。

`HF_TOKEN` が未設定の場合は、`pyannote/speaker-diarization-3.1`、`pyannote/segmentation-3.0`、`pyannote/speaker-diarization-community-1` の3つのモデルページで利用条件に同意する必要があることを含めたエラーメッセージを出して終了する。

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

`timeout_sec` の推奨値は、**推奨する `chunk_minutes` での推定所要時間**から算出する（実機で初めて `doctor` を実行して、実測の長さと推奨する長さが違うと打ち切り時間が足りなくなる誤りが見つかったため）。手順4の2点（約1,000トークンと `chunk_minutes` 相当）から、所要秒数 = 切片 + 傾き × 入力トークン数 の直線を当てはめる（傾き = 秒数の差 / トークン数の差、切片 = 短い方の秒数 − 短い方のトークン数 × 傾き。切片が負なら0）。推奨する `chunk_minutes` を `tokens_for_minutes` でトークン数に直し（実測の入力がそれより多い分、すなわちシステムプロンプト分は足す）、推定秒数の3倍を、下限600秒・分単位切り上げで推奨する。トークン数は概算で、2点からの直線は近似である。2点が揃わない場合、または傾きが0以下（長い入力のほうが速い）の場合は、従来どおり実測の3倍とし、根拠に「推奨する chunk_minutes での所要時間は推定できていない」と書く。

段階を3つに留めているのは、実測に時間をかけすぎないためである（125B の MoE をオフロードして動かすため、1回の応答に数分から十数分かかる見込み）。

判定ロジックと推奨値の算出は、偽の `HttpTransport` を差し込んでクラウド環境でテストしている（46件）。実サーバーへの接続は `requires_network` マーカーを付けてある。

---

## 8. 同じ会議の2回の抽出の比較（`compare/report.py`）

```
uv run kijun compare --conversation <ID> [--model <名前> --model <名前>] [--out <パス>]
```

初版では `CLAUDE.md` 4-4 の「Qwen と Claude API の両方で抽出して承認件数を比較する」を支える道具だったが、2026-11-03 の決定で背面が Strata 1つになったため、**同じ Strata で設定を変えた2回の実行を比べる道具**になった。比べる対象は、量子化の違い、`chunk_minutes` の違い、`enforces_json_schema` の違いなどである。`--model` を2回指定すると比べる2つを明示でき、省略すると設定の `model` を左に置く。

**承認件数そのものは出せない。** ユーザーがレビューして初めて決まる数字であり、レビューの受け皿は Discord（14節）で、承認は `review_log` に入るが、`compare` はそれを読まないからである。出せるのは次の2つである。

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
| `extract.strata.start_command` / `stop_command` | `""` | 文字起こしに VRAM を明け渡すための手段。夜間バッチが読む（15-2）。空なら何もしない。`start_command` は Strata を起動したらすぐに戻るコマンドを書く。`stop_command` は抽出が失敗しても必ず実行する |
| `discord.review_channel_id` / `contradictions_channel_id` | `""` | 空なら `post` と `read` は何もせず、設定が必要であることを出して終了コード 0 で終わる。チャンネルは個人サーバーに作る（`CLAUDE.md` 2章） |
| `discord.max_candidates_per_run` | `20` | 1回の `post` で投稿する新規候補の上限。`CLAUDE.md` 6章の「レビュー所要時間が15分を超え続けたら上限を下げる」の調整用。超えた分は次回に回る（14-2） |
| `discord.max_contradictions_per_run` | `5` | 1回の `post` で投稿する矛盾の組の上限 |
| `discord.contradiction_min_similarity` | `0.70` | 矛盾の候補とみなす類似度の下限。上限は `match.similarity_threshold`。**設計者が考案した規則の閾値で、運用して見直す**（14-7） |
| `discord.speaker_sample_count` | `3` | 話者の対応付けで投稿する発言サンプルの件数（`CLAUDE.md` 4-6） |
| `batch.log_dir` | `./data/batch_logs` | 夜間バッチの各手順の記録（`<日時>.json`）の保存先（15-4） |
| `batch.strata_ready_timeout_sec` | `1800` | **仕様に無い設定を足した。** `start_command` を実行してから Strata が応答するまで待つ秒数。125B の MoE の読み込みには時間がかかり、待たずに抽出を始めると接続できず毎晩失敗する。`start_command` が戻らない場合の打ち切りにも使う。実機で起動にかかる時間を測って詰める |
| `match.query_prefix` / `passage_prefix` | `"query: "` / `"passage: "` | e5 系が要求する接頭辞。ruri に差し替えるなら書き換える（9-1） |
| `match.similarity_threshold` | `0.85` | `CLAUDE.md` 4-5 の初期値 |
| `retention.audio_days` | `30` | `CLAUDE.md` 2章の参加者への説明と一致させる |

秘密情報は設定ファイルに書かず、環境変数で渡す。設定ファイルをうっかりコミットしたときの被害を避けるためである。必要なのは `HF_TOKEN`（pyannote のモデル取得）と `DISCORD_BOT_TOKEN`（Discord bot。Developer Portal で取得する）の2つである。

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
| `discord` | discord.py（無料。運転費用はゼロのまま） |

**torch の CUDA 版について。** `pyproject.toml` に CUDA 版のインデックス指定（`[[tool.uv.index]]` と `[tool.uv.sources]`）を書いてあるが、**コメントアウトしてある**。`uv lock` は extra を使わなくても `download.pytorch.org` に接続し、開発環境の proxy がそれを 403 で拒否して `uv sync` が失敗するためである。Windows では**コメントを外してから** `uv sync --extra transcribe` を実行する。タグは 2026-11-03 に https://pytorch.org/get-started/locally/ で確認した `cu132`（CUDA 13.2、torch 2.15.0、Windows）を書いてある。CUDA 13.4 を選ぶ場合は `cu134`。実機の Strata のエンジンも CUDA 13 系（`cublas64_13.dll`）なので一致する。

この変更は実機の環境に固有なので、**コミットしない**。コミットすると、開発環境の `uv sync` が `download.pytorch.org` に接続できずに失敗する。

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
uv run kijun discord read                                # リアクションと返信を読み取って DB に反映（夜間バッチの手順1）
uv run kijun discord post                                # 新規候補・矛盾・話者の対応付け・束ね結果を投稿（手順8）
uv run kijun derive-direction (--conversation <ID> | --all)   # speaker_map から direction を導出
uv run kijun batch run [--dry-run]                       # 夜間バッチ
uv run kijun batch install-task [--time 23:30] [--out <パス>]  # タスクスケジューラ用の XML を生成
```

`transcribe --audio` は DuckDB に書かず、結果を標準出力と JSON に出すだけである。`CLAUDE.md` 8章 #4 の「1ファイルで faster-whisper と pyannote を試すスクリプト」が、登録を経ずに1本の音声で試せる必要があるためで、ユーザーが最初に実行するコマンドになる。

`match --provider` があるのは、2つの設定で抽出した場合に同じ指摘から項目が二重にできるのを防ぐためである。

### 11-3. テスト

自動テストは31ファイル・564件で、GPU・ネットワーク・Discord のトークンが要る9件は既定で実行しない（`pyproject.toml` の `addopts` に `-m 'not requires_gpu and not requires_network'` を書いてある）。実機では `uv run pytest -m requires_gpu` と `-m requires_network` で実行でき、試験用の音声のパスは環境変数 `KIJUN_TEST_AUDIO` で渡す。

クラウド環境で検証できるのは、第2弾の文面の生成、投稿の順序と件数の上限、リアクションの判定（判定できない場合を含む）、返信の原文の保存、話者の対応付けの解釈、`direction` の導出、矛盾の検出、スキーマの移行、夜間バッチの手順と失敗時の動き、タスクスケジューラの XML、および設定の読み込み、ファイル名の解析、会議単位への束ね、DuckDB のスキーマと読み書き、チャンク分割と発話番号の復元、重複排除、区間の突き合わせ、壊れた JSON の修復、コサイン類似度と閾値の分岐、音声の削除の対象選定、文字起こしの書き出しと読み戻しの往復、`doctor` の判定ロジックと推奨値の算出である。

検証できないのは、faster-whisper、pyannote、Strata、sentence-transformers の実呼び出しと、torch の CUDA 版のインストール、**discord.py の実接続**、**Windows のタスクスケジューラへの登録**である。

### 11-4. 実機での確認の順序

| # | 作業 | 成功の判定 |
|---|---|---|
| 1 | Python 3.12 と uv を入れる | `uv --version` が出る |
| 2 | `uv sync` | `uv run kijun --help` がサブコマンド一覧を出す |
| 3 | `uv run pytest` | 555件通り、9件が除外される |
| 4 | `uv run kijun db init` | `data/kijun.duckdb` に10テーブルとビュー1つができる |
| 5 | Strata を起動し `uv run kijun doctor` | 実測の表が出る。**音声が無くても実行できる。** 出た推奨値を `config.toml` に書き写す |
| 6 | `pyproject.toml` の torch のインデックスを有効にして `uv sync --extra transcribe` | `uv run python -c "import torch; print(torch.cuda.is_available())"` が True |
| 7 | Hugging Face で pyannote の3つのモデルページの利用条件に同意し `HF_TOKEN` を設定 | 次の手順が通る |
| 8 | 5分程度の音声1本で `uv run kijun transcribe --audio <ファイル>` | 日本語の文字起こしと `SPEAKER_00` 等のラベルが出る。処理時間と VRAM 使用量を記録する（別の端末で `nvidia-smi`） |
| 9 | 2時間の音声で `uv run kijun extract` | 抽出が取れる。所要時間とメモリ使用量を記録する |
| 10 | `uv run kijun match` | 初回は全件が新規候補になる |
| 11 | Discord Developer Portal でアプリケーションと bot を作り、Message Content Intent を有効にし、個人サーバーに招待する。`uv sync --extra discord` と `DISCORD_BOT_TOKEN` の設定、`config.toml` にチャンネルIDを書く | `uv run pytest -m requires_network`（`KIJUN_TEST_DISCORD_CHANNEL` に試験用チャンネルのID）で投稿・リアクション・返信の取得が通る |
| 12 | `uv run kijun discord post` | 新規候補と話者の対応付けが Discord に出る。✅などのリアクションとスレッドへの返信を付ける |
| 13 | `uv run kijun discord read` | リアクションが `knowledge_items` に、返信の原文が `review_log` に入る。返信の振り分けには Strata が要る |
| 14 | `uv run kijun batch run --dry-run`、続けて `uv run kijun batch run` | 手順と順序が出る。実行すると `data/batch_logs/` に記録が残り、最後に成否の表が出る |
| 15 | `uv run kijun batch install-task` が作った XML を `schtasks /create /xml` で登録する | タスクスケジューラに登録され、翌晩に起動する（スリープ解除を含む） |

手順5、8、9で記録した値は、12節の未確認事項への回答になる。手順11から15は、14節と15節の未確認事項（Discord への実接続、Strata の起動時間、スリープ解除）への回答になる。結果を `CLAUDE.md` 7章に書き込む。

手順1から4は、音声が1本も無くても、Strata が無くても実行できる。手順5は Strata があれば音声なしで実行できる。手順8から10は音声が必要で、`CLAUDE.md` 8章 #1〜#3（iCloud for Windows の導入、ショートカットの作成、通話の録音テスト）が終わっていることが前提である。

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
| ~~torch の CUDA のタグ~~ | **2026-11-03 確認済み。`cu132`（CUDA 13.2、torch 2.15.0、Windows）** | — |
| 通話を PC スピーカー出力で録音したときに相手の声が十分に入るか | 未確認（`CLAUDE.md` 7章） | 30分の実通話テスト |
| `artifact_type` と `counterpart_role` の値の一覧 | LLM が自由な文字列を入れる。候補を固定していない | 3回分の抽出結果を見て、よく出る値を `config.toml` に列挙する |

第2弾（Discord でのレビューと夜間バッチ）で未解決のまま残ることは次のとおりである。

| 事項 | 現状 | 確認する手段 |
|---|---|---|
| Discord に実際に接続したときの動作 | **未検証。** ユーザーが Developer Portal でアプリケーションと bot を作り、Message Content Intent を有効にし、個人サーバーに招待するまで確認できない（`CLAUDE.md` 4-6）。`impl.py` は REST のみで読み書きする設計だが、`fetch_channel` / `history` / スレッド ID が元のメッセージの ID と同じという前提は実接続で確かめていない | 実機手順11。`requires_network` のテスト |
| 矛盾の検出規則 | 設計者が考案したもので、`CLAUDE.md` に根拠が無い。閾値を設定値にしてあるので運用して見直す | 数週間運用して、投稿された組のうち適用条件の追記につながった割合を見る |
| 重要度の3段階の基準 | プロンプトの文面（3=目的が果たせなくなる、2=守る方が明らかに良い、1=好みや体裁）は設計者が書いたもの。`CLAUDE.md` に基準は無い | 上書きの履歴（`importance_overridden` の `before` と `after`）を見て直す |
| 重要度の回帰による再推定 | `CLAUDE.md` 4-6 の「2〜3か月後」の作業。上書き履歴を残す形だけ決めてある | 2〜3か月分の `review_log` |
| 束ね結果に ❌ が付いたときの再分割 | 自動では行わない。ユーザーが `bundle.gap_minutes` を調整して `kijun ingest` をやり直す | — |
| 1日のレビュー所要時間の測定 | `discord_posts.posted_at` と `review_log.reviewed_at` の差から概算できるが、ユーザーが画面を見ていた時間とは一致しない | 運用してからビューの形を決める |
| Strata の起動にかかる時間 | 未確認。夜間バッチは手順1・6・8で最大3回起動する（15-2）。起動が重ければ、`start_command` を空にして Strata を手で起動しておく運用に切り替える | 実機手順14の記録（`batch_logs` の `1:start`、`6:start`、`8:start` の所要時間） |
| `start_command` が戻るコマンドとして書けるか | 未確認。フォアグラウンドで動き続けるコマンドは `batch.strata_ready_timeout_sec` で打ち切って失敗にする | 実機手順14 |
| タスクスケジューラのスリープ解除 | 未確認。XML の `WakeToRun` で指定したが、PC の電源設定（スリープ解除タイマーの許可）に左右される | 実機手順15の翌朝に `batch_logs` の時刻を見る |
| 話者の対応付けの返信の原文が残らない | `review_log` は `item_id` が NOT NULL で、項目に属さない出来事を入れられない。`speaker_map` の結果だけが残る。LLM が自然文を誤って解釈しても、原文から直せない | 問題になったら `review_log` を `conversation_id` も持てる形にする（スキーマ変更） |
| 処理済みの投稿への後からの返信 | 取り込まない（`processed_at` を埋めた投稿は読み直さない）。承認のリアクションを付けた後で、次の `read` までの間に書いた返信は取り込まれる | 運用で、リアクションと返信を同じ時間帯に付ける |
| 再抽出した後の `direction` | `extract` をやり直した会議の `direction` は NULL に戻る。`kijun derive-direction --conversation <ID>` で導出し直す（自動では行わない） | — |
| 人物ラベルの表示 | 投稿の「（あなた→kouhai_a）」は、`speaker_map` の `person_label` をそのまま出す。`kouhai_a` を「後輩A」と書き換える表は持っていない | 必要になったら `format.py` に辞書を足す |

---

## 13. 変更の履歴

### 2026-11-03：第2弾（Discord でのレビューと夜間バッチ）を実装した

`CLAUDE.md` 8章の #7（Discord bot）と #8（夜間バッチの統合とタスクスケジューラ登録）を実装した。仕様書を先に書き（`docs/design-phase2.md`）、実装後にこの文書の14節と15節へ統合して仕様書を削除した。

- スキーマをバージョン2に上げた（`discord_posts` に `channel_id` と `pair_item_id`）。`migrate.py` に旧バージョンからの移行（`MIGRATIONS`）を持たせた。
- `discord read` / `discord post` / `derive-direction` / `batch run` / `batch install-task` を追加した。
- `parse_with_repair` を `parse_model_with_repair` に一般化し、抽出以外の LLM 用途（重要度、返信の振り分け、話者の解釈）も修復を通すようにした。
- 設計者（Claude）が考案し、`CLAUDE.md` に根拠が無い判断が3つある（`CLAUDE.md` 10章にも記録した）。(1) 矛盾の検出規則（14-7）、(2) 返信の振り分け先に `not_applies_when` を加えたこと（14-5）、(3) 束ね結果に ❌ が付いても自動では分割し直さないこと（14-4）。
- 仕様書と違う実装にした点が3つある。(1) Strata を起動する手順を、手順6だけでなく手順1と8にも広げた（15-2）。(2) 設定 `batch.strata_ready_timeout_sec` を足した（10節）。(3) LLM の用途は仕様書で2つだったが、話者の対応付けの自然文の解釈が3つ目として要った（14-6）。理由は各節に書いた。

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

---

## 14. Discord でのレビュー（第2弾 #7）

`CLAUDE.md` 4-6 を実装した。段5（ユーザーが理由と適用条件を書き足す工程）の受け皿である。

### 14-1. bot は常駐しない

ボタン操作を受け付ける常駐 bot は PC が落ちていると反応しない。そのため、**リアクションと返信を翌晩のバッチでまとめて読み取る**。Discord の処理を2つのサブコマンドに分け、夜間バッチが順に起動する。処理が終われば接続を閉じる。

- `kijun discord read`：前回までに投稿したメッセージのリアクションと返信を読み、DuckDB に反映する（`CLAUDE.md` 4-8 の手順1）。
- `kijun discord post`：新規候補・矛盾の可能性がある組・話者の対応付けの依頼・束ね結果を投稿する（手順8）。

`read` を先に実行するのは、前夜の投稿への反応を取り込んでから、その晩の新しい候補を投稿するためである。

### 14-2. 投稿（`discord/post.py`、`format.py`）

| 種類（`post_type`） | 投稿先 | 内容 |
|---|---|---|
| `new_candidate` | `discord.review_channel_id` | 新規候補を1件1メッセージ。`CLAUDE.md` 4-6 の例の形 |
| `contradiction` | `discord.contradictions_channel_id` | 矛盾の可能性がある組（14-7） |
| `speaker_sample` | review | 会議ごとに各話者の発言サンプル。文字数の多い発話の上位3件（「はい」のような短い発話では誰か分からないため） |
| `bundle_result` | review | 複数ファイルを束ねた会議のファイル一覧 |

`CLAUDE.md` 4-6 は話者の対応付けと束ね結果の投稿先のチャンネルを指定していないので、review に投稿する。

文面で守る点：

- 語られなかった理由・適用条件は、空欄にせず**「記録なし」**と書く。語られなかったことが見えている必要があり、LLM が推測で埋めていないことの確認にもなる（`CLAUDE.md` 1章）。
- 話者と人物の対応（`speaker_map`）が未確定なら、「（SPEAKER_00→SPEAKER_01）」と話者ラベルのまま出す。`self` は「あなた」と出す。
- 重要度を判定できなかったら「重要度案：なし」。
- Discord の1メッセージは2,000文字まで。超えるときは `quote` だけを末尾から切り詰めて「…」を付ける。チェック項目案と理由・適用条件は切り詰めない（レビューの対象だから）。**それでも収まらない文面は投稿せず、失敗として報告する。** 黙って削ると、レビューの対象が見えなくなる。
- 話者の対応付けの文面には、構造化形式の例（`00=self 01=kouhai_a`）と、自分を必ず `self` と書くことを入れる。

投稿の順序と件数（`CLAUDE.md` 6章の「レビュー所要時間が15分を超え続けたら、1回の投稿件数の上限を下げる」に対応）：

- 1回の `post` で投稿する新規候補は `discord.max_candidates_per_run` 件まで。超えた分は次回に回り、**まだ投稿していない件数を標準出力に出す**（投稿に失敗した分も含む）。
- 順序は重要度の降順、同じ重要度なら新しい順。レビューの時間が足りないときに、重要なものが先に目に入るようにするため。重要度が NULL のものは最後。「新しい」は、根拠の抽出結果を最初に紐付けた時刻（候補ができた時刻）で決める。
- 重要度の降順に並べるには、投稿しない分も含めて判定が要る。未判定の項目は投稿の前にまとめて判定し、項目に保存する（次回以降は再判定しない）。

投稿するたびにその場で `discord_posts` に記録する（途中で落ちても、投稿済みのものを再投稿しないため）。1件の失敗で全体を止めず、失敗は報告に残して終了コード1で知らせる。

### 14-3. 重要度の3段階判定（`review/importance.py`）

LLM に、チェック項目の文言と、根拠になった発言（`item_evidence` 経由の `quote` を最大3件）を渡し、1〜3 の整数と根拠の1文を返させる。出力は pydantic で定義し、`repair.py` を通してから検証する（原則2-3）。

- 判定値は `knowledge_items.importance` に入れる。根拠の1文は `review_log` に `importance_proposed`（`after` に根拠）で残す。
- ユーザーのリアクションが決めた値は、`importance_overridden`（`before` に LLM の判定値、`after` にユーザーの値）で残す。**1行から、目的変数（ユーザーの値）と LLM の判定値の両方が取れる形にしてある。** `CLAUDE.md` 4-6 の「2〜3か月後、上書き履歴を目的変数にして、ルールの重みを回帰で推定し直す」のためで、後から形を変えると、それまでの履歴が分析に使えなくなる。
- ユーザーの値が LLM の判定値と同じときも `importance_overridden` を残す（`before` と `after` が同じ行は、LLM の判定への同意を表す）。同じリアクションを読み直しても、直前の記録の `after` と同じなら重複して残さない。LLM が判定できなかった項目（`before` が NULL）にユーザーが値を付けたときも残す。
- 判定に失敗したら `importance` は NULL のまま。処理は止めない。

回帰の説明変数の候補（直近30日の出現回数、成果物の種類、同じ指摘をした相手の人数）は `extracted_requests` と `item_evidence` から後から計算できるので、列は増やしていない。

### 14-4. 読み取りとリアクションの扱い（`discord/read.py`）

`discord_posts` の `processed_at` が NULL の行を、投稿の種類ごとに処理する。リアクションの絵文字は、異体字セレクタ（U+FE0F）を外してから比較する（Discord は「⏸」を「⏸️」の形で返すことがあるため）。

| 状態 | 扱い |
|---|---|
| status のリアクション（✅❌⏸）が1つだけ | そのとおりに `knowledge_items.status` を更新し、`status_changed`（`before` / `after` に status）を残す |
| status のリアクションが**2つ以上** | **`status` を変えない。** `review_log` に `ambiguous_reaction` を残し、`processed_at` を**埋めない**。会議IDと項目IDを標準出力に出す。次回の `read` で再び対象になる（同じ内容のログは重複して残さない） |
| 重要度のリアクション（1️⃣2️⃣3️⃣）が2つ以上 | `importance` を変えない。`ambiguous_reaction` を残して報告する |
| status が無く重要度だけ | 重要度のみ更新する。`processed_at` は埋めない（後から status を付ける可能性があるため） |
| どのリアクションも無い | 何もしない。`processed_at` を埋めない |
| Discord から取得できない（削除された等） | 未処理のまま残し、報告する |

2つ以上のときに片方を優先しないのは原則2-4のためである。誤って承認すると、誤ったチェック項目が「承認済み」として残り、それが間違いであることが後から分かりにくい。

`processed_at` を埋める条件：`new_candidate` は status が1つに決まり、かつ返信の振り分けがすべて済んだとき。`contradiction` は返信が1件以上あり、すべて振り分けが済んだとき（リアクションでは何も変えない）。`speaker_sample` は反映後の `speaker_map` に `self` があり、解釈できない返信が無いとき。`bundle_result` は ❌ または ✅ が付いたとき。

`bundle_result` に ❌ が付いたら、束ね直しが必要であることを標準出力に出す。**自動で分割し直さない。** 会議の束ね方を機械が勝手に変えると、どの音声がどの会議に属するかが分からなくなる。ユーザーが `bundle.gap_minutes` を調整して `kijun ingest` をやり直す運用にする。

### 14-5. 返信の振り分け（`review/reply.py`）

ユーザーの自由な文章を、`reason` / `applies_when` / `not_applies_when` / `check_text` のどれに入れるかを LLM が決める。`CLAUDE.md` 4-6 は3つ（`not_applies_when` を除く）を挙げているが、`knowledge_items` には `not_applies_when` の列があり、「どの状況なら同じ指摘をしないか」は `CLAUDE.md` 1章が暗黙知の中身として挙げる2点のうちの1つなので、振り分け先に加えた。1つの返信が複数の欄に当たることがあるので、出力は欄ごとの配列にする（`{"items": [{"column", "text", "target"}]}`）。プロンプトは、返信に書かれていない理由や条件を推測で足さないよう指示している（`CLAUDE.md` 1章）。

**返信の原文は必ず `review_log` に残す。** 手順は次のとおり。

1. 原文を `reply_received`（`before` に返信の送信時刻、`after` に全文）で残す。**LLM を呼ぶ前に残す。** 既にあれば重複して残さない。
2. 既に `reply_routed` があれば（前の夜に処理済み）何もしない。
3. LLM に振り分けさせる。失敗したら、原文の記録だけが残り、`knowledge_items` は変わらず、`processed_at` も埋まらない。次回の `read` で再試行される。
4. 成功したら項目に反映し、`reply_routed`（`before` に返信の送信時刻、`after` に反映した内容の JSON：`applied` に項目・欄・古い値・新しい値、`unapplied` に反映できなかった部分）を**別の行**で残す。`check_text` を変えたときは `check_text_edited`（`before` / `after` に文言）も残す。

`before` に送信時刻を入れるのは、`reply_received` と `reply_routed` を同じ返信として結ぶ鍵にするためである。

同じ欄に既に値があるときは、**ユーザーの返信で置き換える**。古い値は `reply_routed` の JSON（`old`）に残る。追記にしなかったのは、同じ内容が重なって増え続けるのを避けるため。複数の返信で1つの欄を書き足したい場合は、最後の返信に全部を書く。`check_text` が変わると `embeddings` の `text_hash` が変わるので、次回の `kijun match` でベクトルが再計算される（9-1）。内容を変えた項目は `version` を1つ上げる。

ユーザー本人以外（他のメンバー、bot 自身）の返信は取り込まない（`Reply.author_is_owner`）。

**矛盾の投稿への返信。** 矛盾の投稿は項目 A と B の2つを扱うので、振り分けの各要素に `target`（A か B）を持たせた。仕様書は「どちらの項目に対する追記かを渡す」とだけ書いていた。`target` を特定できない要素は反映せず `unapplied` に残す（返信の原文は残っている）。

### 14-6. 話者の対応付けと `direction`（`review/speaker.py`、`direction.py`）

ユーザーの返信を2通りで受ける。

1. **構造化形式**（`00=self 01=kouhai_a`）：LLM を使わず正規表現で解釈する。確実で、費用も時間もかからない。返信の全体がこの形式の組と区切り（空白、カンマ、読点）だけでできているときだけ、この形式とみなす。
2. **自然な文**（「00は自分、01は後輩A」）：1で解釈できなければ LLM に渡す。自然文から英数字の `person_label` への変換は LLM が行う。

この自然文の解釈は、仕様書が挙げた LLM の用途（重要度、返信の振り分け）に加えて、3つ目の用途である。仕様書の2-7が LLM を使うと書いていたが、1-3の一覧には無かった。

推測しない：会議に存在しない話者ラベルへの対応付けは反映しない。同じ話者に別々の人物が書かれた返信は解釈しない。人物ラベルが英小文字・数字・アンダースコアでなければ反映しない。返信は古い順に適用するので、後の返信が訂正になる。**`self` が1つも決まらない会議は、その旨を出し、`direction` を導出しない。** 投稿も未処理のまま残り、続きの返信で `self` が決まれば次回に導出される。

`direction` の導出（`docs/implementation-plan.md` 4-1 で予告していた処理）。`speaker_map` で `person_label` が `self` の話者を「自分」とする。

| 要求した側 | 言われた側 | `direction` |
|---|---|---|
| 自分 | 自分でない | `self_to_other` |
| 自分でない | 自分 | `other_to_self` |
| どちらも自分でない（第三者間） | | NULL のまま |
| 両方自分（独り言、または話者分離の誤り） | | NULL のまま |

`speaker_map` に無い話者ラベルと、話者ラベルが NULL の側は、自分でないものとして扱う。この導出が無いと、`CLAUDE.md` 5章の「`direction = 相手→自分` の抽出結果は、相手別の要求一覧として扱う」ができず、自分が言ったことと言われたことを分けられない。

`speaker_map` を手で直した後に導出し直せるよう、独立したコマンド `kijun derive-direction (--conversation <ID> | --all)` も持つ。導出結果が NULL でも書き込むので、`speaker_map` を直した後に以前の値が残らない。変更は `review_log` に記録しない（`review_log` はユーザーのレビューの履歴を残す表であり、機械的な導出を混ぜない）。

### 14-7. 矛盾の可能性がある組み合わせ（`review/contradiction.py`）

**この検出規則は `CLAUDE.md` に書かれていない。設計者（Claude）が考案したもので、推奨・未確定である。** 運用して見直す前提で、閾値はすべて設定値にした。

目的は矛盾を証明することではなく、**同じ話題で基準が食い違っている組を人に見せること**である（`CLAUDE.md` 4-6：「多くは矛盾ではなく適用条件の違いであり、適用条件を言語化する材料になる」）。`status` が `approved` または `candidate` の組 (A, B) で、次のすべてを満たすものを投稿する。

1. `artifact_type` が同じ。両方 NULL の場合は同じとみなさない（種類が分からないものどうしを同じ成果物と決めつけない）。
2. `check_text` のコサイン類似度が `discord.contradiction_min_similarity`（0.70）以上。
3. 同じ類似度が `match.similarity_threshold`（0.85）未満。0.85以上は同一項目として統合される側なので、矛盾ではなく重複。
4. `pass_criterion` が両方とも非 NULL で、文言が異なる（NFKC で正規化し、前後の空白を除いて比べる）。
5. まだ `contradiction` として投稿されていない組（A と B の順序は問わない）。

ベクトルは既存の `embeddings` から取る。無い項目、`check_text` が編集されてベクトルが古くなった項目は対象外（この処理のために埋め込みモデルを読み込まない。原則2-2の VRAM の都合）。類似度の高い組から、`discord.max_contradictions_per_run` 件まで投稿する。

### 14-8. `DiscordClient` の実装（`discord/impl.py`）

`post`、`fetch_reactions`、`fetch_thread_replies` の3つだけを持つ。`discord.py` は `deps.py` の仕組みで遅延 import し、無ければ `uv sync --extra discord` を示す。ゲートウェイには接続せず、REST だけで読み書きし、呼び出しごとにログインして閉じる。

- 返信は2通りの形で受ける：メッセージから作ったスレッドの中の発言（スレッドの ID は元のメッセージの ID と同じ）と、チャンネルの中でそのメッセージを参照して返信したメッセージ。参照返信は、チャンネルの履歴を1回だけ読んで振り分ける。
- ユーザー本人の判定は、サーバーの所有者（`guild.owner_id`）の発言で、bot でないこと。個人サーバーなので所有者がユーザー本人である。
- **Message Content Intent を有効にしていないと返信の本文が空で返る。** 本文が空の返信を見つけたら、Developer Portal で有効にするよう示すエラーにする（`CLAUDE.md` 4-6 の bot の準備の2番）。
- bot の準備：Developer Portal でアプリケーションと bot を作ってトークンを取得し、Message Content Intent を有効にして、個人サーバーに招待する。トークンは環境変数 `DISCORD_BOT_TOKEN` で渡す。

### 14-9. データベース

`discord_posts` に `channel_id` と `pair_item_id` を足し、`SCHEMA_VERSION` を 2 に上げた。`schema.sql` の `CREATE TABLE IF NOT EXISTS` は既存のテーブルに列を足さないので、`migrate.py` の `MIGRATIONS` に「バージョン2への移行」として `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` を書いた。記録されたバージョンが古い DB にだけ適用する。バージョン1の DB で、既存の行が残ることを確認している。

`review_log.action` の値は `db/repo.py` に定数で1か所定義し、定義外の値を書こうとすると拒否する。

| `action` | `before` | `after` |
|---|---|---|
| `status_changed` | 変更前の status | 変更後の status |
| `importance_proposed` | NULL | LLM が添えた根拠の1文 |
| `importance_overridden` | LLM の判定値（判定できなかったときは NULL） | ユーザーの値 |
| `reply_received` | 返信の送信時刻（ISO） | 返信の原文の全文 |
| `reply_routed` | 対応する返信の送信時刻（ISO） | 反映した内容の JSON |
| `ambiguous_reaction` | NULL | 判定できなかったリアクションの説明 |
| `check_text_edited` | 変更前の `check_text` | 変更後の `check_text` |

---

## 15. 夜間バッチ（第2弾 #8）

### 15-1. 薄いオーケストレーター

`kijun batch run [--dry-run]` が、`CLAUDE.md` 4-8 の手順を順に実行する。**既存の CLI を `subprocess` で別プロセスとして起動する**（原則2-2）。現在の Python で `kijun.cli` を実行するので、タスクスケジューラの環境で `uv` が PATH に無くても動く。画面と記録には `uv run kijun ...` の形で出す。`--dry-run` は実行する順序と各コマンドを表示するだけで、何も実行せず、DB にも触れない。

| 手順 | 実行するもの | 失敗したとき |
|---|---|---|
| 1 | `kijun discord read` | 続ける（前夜の反応が取り込めなくても、その晩の処理はできる） |
| 2 | `kijun ingest` | **止める**（会議が登録できなければ以降が何もできない） |
| 3-4 | `kijun transcribe --conversation <ID>`（会議ごと） | その会議だけ飛ばして次に進む |
| 5 | （手順3-4のプロセスの終了。OS が GPU メモリを解放する） | — |
| 6 | `start_command` → `kijun extract --conversation <ID>`（会議ごと）→ `stop_command` | その会議だけ飛ばす。`stop_command` は必ず実行する |
| 7 | `kijun match --conversation <ID>`（会議ごと） | その会議だけ飛ばす |
| 8 | `kijun discord post` | 続ける |
| 9 | 処理済み音声を `processed` へ移動（`transcribe` が既に行う） | — |

終了コードは、失敗した手順が1つでもあれば 1。

### 15-2. Strata の起動・停止（`batch/strata.py`）

faster-whisper・pyannote と Strata を同時に VRAM に載せないため、Strata を使う手順の前後で `extract.strata.start_command` と `stop_command` を実行する。

- `start_command` は、Strata を起動したらすぐに戻るコマンドを前提にする。戻らなければ `batch.strata_ready_timeout_sec` で打ち切って失敗にする。
- `start_command` が戻っても Strata の読み込みは終わっていないので、`GET {base_url}/v1/models`（`doctor` と同じ）が応答するまで待つ。仕様書には無かったが、待たずに `extract` を始めると接続できず毎晩失敗する。
- `stop_command` は、起動に失敗しても、窓の中の処理が失敗しても、必ず実行する。
- 空文字のコマンドは何もしない。`start_command` が空なら、どの手順でも Strata を起動しない（手で起動しておく運用）。
- 手順6の窓は会議ごとではなく、抽出する全会議を1回の起動で挟む（125B の MoE の読み込みを会議の数だけ繰り返さないため）。抽出前の会議が無ければ窓を開かない。

**仕様書と違う点（手順1と8にも窓を広げた）。** 仕様書は Strata を起動するのを手順6だけにしていた。しかし手順1（`read`：返信の振り分けと話者の自然文の解釈）と手順8（`post`：重要度の判定）も LLM を使う。手順6の終わりに `stop_command` で Strata を止める以上、手順1と8の時点で Strata は動いていない。そのため手順1と8も、LLM が要るときだけ窓で挟む：手順1は未処理の投稿があるとき、手順8は重要度が未判定の候補があり review チャンネルが設定されているとき。どちらも `start_command` が設定されているときだけである。手順1と8は LLM が使えなくても動く（返信は原文だけ記録して未処理に残り、重要度は NULL で投稿される）ので、Strata が起動できなくてもその手順は実行する。Strata の起動が重ければ、窓は夜に最大3回開くことになる。重い場合は `start_command` を空にして、Strata を手で起動しておく運用にする。

### 15-3. 処理が要る会議の選び方（`batch/pending.py`）

各段の直前に DB から調べ直す（子プロセスが DB を書き換えるため。子プロセスが動いている間は接続を開かない）。

- 文字起こし：`transcribed_at` が NULL の会議。
- 抽出：文字起こし済みで、`model_name` が設定のモデルの抽出結果が無い会議。**0件で済んだ会議を毎晩抽出し直さない**よう、`extract_logs` の記録で済みを見分ける。記録があっても全チャンクが失敗していたら再試行する。一部のチャンクだけ失敗した会議は自動では再試行しない（同じチャンクが毎晩失敗し続けるのを避ける。失敗は抽出の出力と記録に残る）。
- 照合：`reusable` な抽出結果のうち、`item_evidence` に紐付いていないものがある会議。照合は埋め込みモデルを読み込むので、対象が無ければ起動しない。

### 15-4. 実行の記録

各手順（Strata の `start`・`ready`・`stop` を含む）の開始・終了時刻、終了コード、標準出力と標準エラーをまとめた末尾 2,000 文字を、`batch.log_dir`（既定 `./data/batch_logs`）に `<日時>.json` として保存する。最後に全手順の成否を表で標準出力に出す。子プロセスの出力は UTF-8 に固定して取り込む（Windows の既定 cp932 での文字化けを避ける）。実機で未検証の部分が多いので、失敗の記録が最初の数週間の主な情報源になる。

### 15-5. タスクスケジューラへの登録（`batch/task_xml.py`）

`kijun batch install-task [--time 23:30] [--out <パス>]` が XML（UTF-16）を作る。登録そのものは `schtasks /create /tn "kijun nightly batch" /xml <パス>` をユーザーが実行する（管理者権限が必要なことがあるので、コマンドを案内する形にした）。

- `WakeToRun = true`：`CLAUDE.md` 4-8 の「タスクを実行するためにスリープを解除する」。これが無いと PC がスリープしていると実行されない。
- `StartWhenAvailable = true`：PC が電源オフだった場合に、次に起動したときに実行する。
- `MultipleInstancesPolicy = IgnoreNew`：前夜のバッチがまだ動いているときに重ねて起動しない（DuckDB は同時に複数のプロセスから書き込めない）。
- `ExecutionTimeLimit = PT0S`（無制限）：抽出が長い。打ち切りよりも、記録で遅さを見て判断する。
- `LogonType = InteractiveToken`：ユーザーがログオンしている間（画面ロックを含む）に実行する。パスワードを保存しないで済むが、ログオフ中は実行されない。
- 作業ディレクトリは `install-task` を実行したディレクトリ（`config.toml` と `./data` の基準になる）。`--config` を付けて実行したときは、それを引き継ぐ。`uv` のフルパスが見つかればそれを書き、見つからなければ `uv` と書いて、書き換えを案内する。
