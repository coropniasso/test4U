# 設計書：会話録音から判断基準を抽出する仕組み

最終更新：2026-11-05

---

## 0. この文書の位置づけ

`CLAUDE.md` は、この仕組みの**目的と制約**を書いた文書である。2026年9月に claude.ai で行った構想段階の議論の結果であり、何のために作るのか、何を入れてはいけないのか、参加者に何を伝えるのかが書いてある。

この文書は、それをどう実装したかを書いた**設計書**である。ファイルごとの役割、データベースの列、設定値の意味、そして「なぜその形にしたか」を書く。実装済みの部分は現在形で書き、未実装の部分はそう明記する。

初版（2026-10-04）は実装前の計画として書いたが、その後の変更（抽出の背面を Ollama と Claude API から Strata に置き換えた、文字起こしの復元コマンドを足した、埋め込みの持ち方を変えた）を差分で重ねたため、旧設計の痕跡が残っていた。2026-11-03 に実装に合わせて全体を書き直した。同じ日に第2弾（Discord でのレビューと夜間バッチ）を実装し、仕様書（`docs/design-phase2.md`、統合して削除した）の内容を14節と15節に移した。

2026-11-05 に、抽出の背面を Strata から Ollama に移した（実装指示書 `docs/ollama-migration-plan.md` の内容を6-6、7、8-2、10、15節に統合し、指示書は削除した）。Strata の実測の記録は消さずに残してある（6-6の末尾、12節、`CLAUDE.md` 7章）。変更の履歴は13節にある。

**実装の状況**

| 対象 | 状況 |
|---|---|
| 取り込み、文字起こし、話者分離、抽出、照合、音声の削除、文字起こしの復元、推論サーバーの実測（第1弾） | 実装済み |
| Discord bot（`CLAUDE.md` 8章 #7）、`direction` の導出、重要度の判定、矛盾の検出（第2弾。14節） | 実装済み。実際の Discord への接続は未確認 |
| 夜間バッチの統合とタスクスケジューラ登録（同 #8。15節） | 実装済み。実機での実行とタスクスケジューラへの登録は未確認 |
| 自動テスト | 719件のうち709件が通る（GPU・ネットワーク・Discord のトークンが要る10件は既定で実行しない） |
| 実機（Windows + RTX 4070 Ti）での動作確認 | 文字起こしと話者分離は確認済み（`CLAUDE.md` 7章）。**Ollama への移行後の `doctor`・測定・`hotwords` の効果は未確認**。11節に手順を書く |

---

## 1. 全体の流れ

音声ファイル1本が `inbox` に届いてから、チェック項目の候補になるまでの流れと、どのコマンドがどの段を担うかを示す。

| 段 | 入力 | 処理 | 出力 | コマンド |
|---|---|---|---|---|
| 1 | `inbox` の音声ファイル | ファイル名から日時・形式・種類を読み、同じ会議に属するファイルを束ねる | `conversations` の行 | `kijun ingest` |
| 2 | 束ねた音声 | faster-whisper で文字起こし、pyannote で話者分離、両者を突き合わせる | `utterances` の行、文字起こしの JSON、音声を `processed` へ移動 | `kijun transcribe` |
| 3 | `utterances` | `chunk_minutes`（既定12分）ごとのチャンクに分け、Ollama で要求・指摘を抽出する | `extracted_requests` の行 | `kijun extract` |
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
| `Extractor` | `extract/base.py` | `ollama_impl.py` | `FakeExtractor` |
| `HttpTransport` | `transport.py` | `httpx` を使う実装 | 偽の応答を返す実装 |
| `Embedder` | `match/base.py` | `e5_impl.py` | `FakeEmbedder` |
| `DiscordClient` | `discord/base.py` | `discord/impl.py`（discord.py） | `FakeDiscordClient` |
| `CommandRunner` | `batch/runner.py` | `SubprocessRunner` | `FakeCommandRunner` |
| `JsonChat` | `review/llm.py` | `OllamaJsonChat` | 偽の `HttpTransport`（`FakeOllamaTransport`） |
| `ImportanceJudge` | `review/importance.py` | `LlmImportanceJudge` | `FakeImportanceJudge` |
| `ReplyRouter` | `review/reply.py` | `LlmReplyRouter` | `FakeReplyRouter` |
| `SpeakerReplyInterpreter` | `review/speaker.py` | `LlmSpeakerInterpreter` | `FakeSpeakerInterpreter` |
| `ProcessLister` | `batch/defer.py` | `TasklistProcessLister` | `FakeProcessLister` |

この境界を置く理由は2つある。1つは、開発と自動テストを行う環境（Anthropic のクラウド上の Linux コンテナ）に GPU も CUDA も無く、音声ファイルも1本も無いため、実装を差し込まずに論理を検査する必要があるから。もう1つは、`CLAUDE.md` 4-3 が kotoba-whisper を、4-5 が ruri を、将来の選択肢として残しているから。

### 2-2. プロセスの境界を VRAM の境界にする

`CLAUDE.md` 4-4 に「faster-whisper・pyannote と抽出用のモデルを同時に VRAM に載せない」とある。RTX 4070 Ti の VRAM は12GBで、large-v3（約3GB）と pyannote（約1GB）と LLM（9B〜14B で数GB〜10GB）を同時には載せられない。

Python の中で `del model` と `torch.cuda.empty_cache()` を呼ぶ方法は採らない。PyTorch のキャッシュアロケータや CTranslate2 が確保した分が残ることがあり、残ったかどうかを確認しにくいためである。代わりに次の形にする。

- 文字起こしと話者分離を行う `transcribe` と、抽出を行う `extract` を、別の CLI サブコマンドにする。
- 夜間バッチ（15節）は、この2つを**別々のプロセスとして順番に起動する**。プロセスが終了すれば OS が GPU メモリを解放するので、解放されたかどうかを疑う必要がなくなる。
- `transcribe` と `extract` は、Python のモジュールとしても互いを import しない。

LLM は別プロセスのサーバー（Ollama）なので、`transcribe` が動く間も VRAM を保持し続ける可能性がある。**設定 `extract.ollama.keep_alive = "0"` なら、Ollama は応答のたびにモデルを VRAM から下ろす**ので、文字起こしの時点で LLM は載っていない。`keep_alive` を `"0"` 以外にした場合だけ、夜間バッチが手順1の後と手順6の後にモデルを下ろす要求を送る（15-2）。以前の Strata は下ろす手段が確認できず、`start_command` と `stop_command` でプロセスごと止めていた。

### 2-3. 確認できない前提は、設定値と実測コマンドに委ねる

Ollama に移った時点で、次の点は実機で確かめていない。

- 設定のモデル（`qwen3.5:9b` など）が `ollama list` に存在し、GPU（VRAM 12GB）に収まるか
- `format` にスキーマを渡したとき、出力が適合するか（`think` と同時に送ったときの挙動は、版とモデルによって推論にも制約が掛かる報告がある）
- `prompt_eval_count` が返らないことがあるか（返らなければ入力の切り捨てを検出できない）
- トークン数の概算の係数（Strata での実測1点からの推測。Ollama のモデルのトークナイザには当てはまらない）

Strata のときの未確認事項（OpenAI 互換のパス、`response_format`、`usage`、入力長の上限）は、実機で解決した（12節）。

これらを推測で決め打ちしない。すべて `config.toml` の設定値にし、`kijun doctor` が実機で測り、`config.toml` に書くべき値を出す。既定値は設計者の初期値で、実測に基づかない（推奨・未確定）。入力が文脈長を超える設定は、読み込みで拒否する（原則2-4）。

この原則の帰結として、**確認できない背面を吸収する処理は、確認できる場所に寄せる**。具体的には、サーバーが壊れた JSON を返す可能性に対して `extract/repair.py` を置いた（`format` にスキーマを渡しても、出力は必ず通す）。これは外部に一切接続しない純粋な文字列処理なので、壊れ方の見本を使ってクラウド環境で完全にテストできる（27件）。

### 2-4. 録音は録り直せないので、失うより止める

会議の録音は再現できない。取り返しのつかない操作には、次の歯止めを置く。

- ファイル名を解析できない音声は、削除も移動もしない。`inbox` に残したまま処理対象から外し、ファイル名の一覧を出す。
- `kijun purge-audio` は既定で**削除しない**。対象の一覧を表で出すだけで、実際に削除するには `--execute` を付ける。
- `kijun restore-transcript` は、`conversations` に同じ会議の行が既にあれば**上書きせずスキップ**する。壊れたデータベースを直す操作で、正しい行を踏み潰す危険を作らないため。上書きには `--force` を付ける。
- **判定できないリアクションは、推測で決めない。** ✅と❌が両方付いていたら、片方を優先せず、`knowledge_items` を変えず、`processed_at` を埋めず、`review_log` に記録して人に見せる（14-4）。誤って承認したチェック項目は、間違いであることが後から分かりにくいため。
- **返信の原文は、LLM に振り分けさせる前に `review_log` に残す。** 振り分けが間違っても、人が原文から直せるようにするため（14-5）。
- **入力が `num_ctx` を超えて Ollama に切り捨てられた疑いがあれば、結果を使わずに止める。** Ollama はエラーを出さずに先頭を捨てるため、捨てられた発話が抽出結果から静かに消える。設定の読み込み、送る前の警告、応答の `prompt_eval_count` の3か所で止める（6-6）。
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
├── scripts/                           # measure_extraction.py（設定ごとの測定）、variants.example.toml
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
│   │   ├── post.py                    # 5種類の投稿
│   │   ├── read.py                    # リアクションと返信の読み取り
│   │   └── run.py                     # discord read / post の本体、実装の組み立て
│   ├── review/
│   │   ├── llm.py                     # 抽出以外の LLM 用途が共有する Ollama 呼び出し（JsonChat / OllamaJsonChat。repair を通す）
│   │   ├── importance.py              # 重要度の3段階判定
│   │   ├── reply.py                   # スレッド返信の振り分けと反映
│   │   ├── speaker.py                 # 話者の対応付けの返信の解釈
│   │   ├── direction.py               # speaker_map から direction を導出
│   │   └── contradiction.py           # 矛盾の可能性がある組の検出
│   ├── batch/                         # 第2弾: 夜間バッチ（15節）
│   │   ├── run.py                     # 手順の実行、記録、--dry-run
│   │   ├── runner.py                  # CommandRunner（subprocess の境界）
│   │   ├── defer.py                   # ゲームの実行中の延期（ProcessLister、GameDeferrer）
│   │   ├── pending.py                 # どの会議にどの処理が要るかを DB から調べる
│   │   └── task_xml.py                # タスクスケジューラの XML
│   ├── transport.py                   # HTTP の送受信の境界（HttpTransport）と所要時間の計測
│   ├── ollama_api.py                  # Ollama の /api/chat のリクエストの組み立てと応答の読み取り、入力の切り捨ての判定、モデルのアンロード
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
│   │   ├── ollama_impl.py             # Ollama のネイティブ API（/api/chat）
│   │   └── run.py                     # extract の本体
│   ├── match/
│   │   ├── base.py                    # Embedder
│   │   ├── e5_impl.py                 # sentence-transformers
│   │   └── link.py                    # コサイン類似度と閾値による分岐
│   ├── compare/report.py              # 同じ会議の2回の抽出の対照表
│   └── retention/purge.py             # 音声の削除
└── tests/                             # test_*.py が35ファイル、719件（既定で709件を実行）
```

`models.py`、`deps.py`、`table.py`、`transport.py`、`tokens.py` と各パッケージの `run.py` は、どれも1つの理由で分けてある。

- `models.py`：`Utterance` を `transcribe` と `extract` の両方が使うが、原則2-2により互いを import させたくないため、共通の置き場を作った。
- `deps.py`：重い依存を関数の中で import し、無いときに `uv sync --extra transcribe` のように入れるべき extra を示す。モジュールのトップレベルで import しないのは、`kijun --help` が依存の有無に関わらず動く必要があるため。
- `table.py`：日本語を含む表を端末で崩さずに出すため。
- `transport.py`：原則2-3により、HTTP の応答の判定を偽の応答で検査できるようにするため。所要時間の計測もここ1か所に置き、`doctor` の速度の実測で使う。
- `tokens.py`：トークン数の概算の関数を1か所に置くため。`run.py` の入力長の警告と `doctor.py` の入力の引き伸ばしが同じ係数を使う。係数は定数ではなく設定（`extract.tokens_per_character`、`extract.tokens_per_meeting_minute`）から引数で受け取る（6-6の「2026-11-03 実機の訂正」）。
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
| `extracted_requests` に `stated_reason_unverified` と `applies_when_unverified` を足した（スキーマバージョン4、2026-11-05） | 原文照合（6-4-1）で外れたときの、モデルの元の値 | 外れた値を黙って捨てず、後から数えられるようにする。バージョン3の DB には `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` で足す（既存の行は NULL） |
| `discord_posts` に `channel_id` と `pair_item_id` を足した（スキーマバージョン2、第2弾） | 投稿先のチャンネルID、矛盾の組のもう一方の `item_id` | 読み取りのときにどのチャンネルを見るかが要る。設定値のチャンネルIDは後から変わり得るので、投稿時の値を残す。`pair_item_id` は、既に投稿した矛盾の組を再投稿しない判定に要る（`item_id` と `pair_item_id` の両方で組を表す） |

### 4-2. テーブル一覧

正確な列と型は `src/kijun/db/schema.sql` を見てほしい。ここでは役割と、設計上の判断が入っている点だけを書く。

| テーブル | 役割 | 設計上の判断 |
|---|---|---|
| `schema_version` | 適用済みのスキーマのバージョン | 現在のバージョンは4（`migrate.SCHEMA_VERSION`）。バージョン1の DB には `kijun db init` が `ALTER TABLE` で列を足す（14-9） |
| `conversations` | 会議単位の管理 | `conversation_id` は束の先頭ファイルの `yyyyMMdd_HHmm_<形式>_<種類>`。`transcribed_at` は音声の削除判定（9節）が見る |
| `utterances` | 発話単位の文字起こし | 主キーは `(conversation_id, seq)`。`seq` は会議内で0から始まる通し番号。`speaker_label` は話者分離が失敗した場合 NULL |
| `extracted_requests` | 抽出した指摘・要求1件ごと | `request_id` は `{conversation_id}#{model_name}#{連番}`。同じ会議を2つの設定で抽出しても衝突しない。`stated_reason` と `applies_when` は、原文照合（6-4-1）を通ったものだけが入り、語られていない・照合で外れたときは NULL。外れたときの元の値は `stated_reason_unverified` / `applies_when_unverified`（バージョン4で追加）。`transcription_unclear` は、文字起こしの聞き取り誤りで要求の中心が決まらなかった場合に TRUE（バージョン3で追加。`CLAUDE.md` 7章の案A） |
| `knowledge_items` | チェック項目本体 | `status` は `candidate` / `approved` / `rejected` / `held`。`importance` は抽出の段階では NULL。Discord への投稿の直前に LLM が判定し（14-3）、ユーザーのリアクションが上書きする |
| `item_evidence` | 項目と根拠発言の対応 | 新規候補として作られた場合 `similarity` は NULL |
| `review_log` | 修正履歴 | `log_id` はシーケンス `review_log_seq` から採番。`action` の値は `db/repo.py` の定数1か所で定義し、定義外の値は書けない（14-9）。`item_id` が NOT NULL なので、項目に属さない出来事（話者の対応付け、束ね結果）は入らない |
| `speaker_map` | 話者ラベルと人物の対応 | `person_label` は `self` / `kouhai_a` のように英数字で持つ。ユーザーの返信から入る（14-6）。`self` が `direction` の導出の基準 |
| `discord_posts` | Discord のメッセージと DB レコードの対応 | `post_type` は `new_candidate` / `contradiction` / `speaker_sample` / `bundle_result` / `transcription_unclear`。`processed_at` が NULL の行が、次の `read` の対象。`request_id` は `transcription_unclear` の投稿で使う（この種別は `knowledge_items` にならないので `item_id` では記録できない。バージョン3で追加） |
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

**音声のデコードは自前で持つ（`transcribe/audio_decode.py` の `decode_to_mono_16k`）。** PyAV（`av`）を直接使い、16,000Hz・モノラル・float32（-1.0〜1.0）の配列を返す。サンプルレートの定数 `SAMPLE_RATE` はこのモジュールに1つだけ置く。faster-whisper には、パスではなくこの配列を渡す（配列なら faster-whisper は内部の `decode_audio` を呼ばない）。pyannote にも同じ配列を渡す。自前にした理由は3つある。

1. `faster_whisper.audio.decode_audio` は公開 API ではなく、今後も壊れ得る。実機で初めて動かしたとき、新しい PyAV が `av.open()` から `metadata_errors` 引数を削除していて、`TypeError` で落ちた。
2. 文字起こしと話者分離が同じ関数で同じ波形を作るので、両者の時刻が原理的にずれない。以前は別々にデコードしていた。ファイルをつなぐ際のオフセットも、`info.duration` ではなくデコードした配列の長さから計算する。
3. `av` を古い版に固定すると、pyannote.audio 4.x が新しい `av` を要求したときに解決できなくなる。そのため `av` は `transcribe` extra と dev に、上限を付けずに直接の依存として宣言する。

pyannote に PyAV で読んだ波形を渡す理由は、torchaudio が Windows で m4a を読めないことがあるためである。pyannote 4.x にも対応しており、`from_pretrained` の引数名は `token` と `use_auth_token` のうち署名を見て選ぶ。

`HF_TOKEN` が未設定の場合は、`pyannote/speaker-diarization-3.1`、`pyannote/segmentation-3.0`、`pyannote/speaker-diarization-community-1` の3つのモデルページで利用条件に同意する必要があることを含めたエラーメッセージを出して終了する。

**CUDA 12 の DLL（Windows）。** faster-whisper の計算エンジンである CTranslate2 4.8.2 は `cublas64_12.dll`（CUDA 12 の cuBLAS）と cuDNN を要求する。torch 2.15.0+cu132 が同梱するのは `cublas64_13.dll` で名前が違うため、`uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` で CUDA 12 版を併せて入れる（`transcribe` extra に Windows 限定の依存として宣言してあるので、`uv sync --extra transcribe` でも入る）。DLL は `.venv\Lib\site-packages\nvidia\cublas\bin` と `...\cudnn\bin` に入る。コード（`transcribe/cuda_dlls.py` の `register_nvidia_dll_directories()`）が、`nvidia` パッケージの下の `bin` のうち `.dll` を含むものを `os.add_dll_directory()` に登録するので、PATH の手動設定は不要である（PATH への追加はそのウィンドウの間しか有効でなく、タスクスケジューラから起動する夜間バッチでは使えない）。**登録は `import faster_whisper` より前に行う**。import で `ctranslate2` の拡張モジュールが読み込まれ、そこで DLL が解決されるためである。登録したディレクトリは標準エラーに1行出る。`WhisperModel` の構築が cublas / cudnn のエラーで失敗したときは、対処を案内するメッセージを出す。

**`hotwords`（頻出語の誤認識への対策）。** 設定 `transcribe.hotwords`（文字列のリスト）が空でなければ、前後の空白と空の語を除いて空白でつないだ1つの文字列にし、`WhisperModel.transcribe(..., hotwords=...)` に渡す（faster-whisper 1.2.1 の引数。署名はテストで確かめている）。空なら `None` を渡す。例は `["箇条書き", "感嘆符"]` で、`CLAUDE.md` 7章に実際に誤認識された語（「箇条書き」→「過剰書き」、「感嘆符」→「簡単フォーイス」）として記録されている。**既定は空**にした。効果は未検証で、`hotwords` は**その語を実際には言っていない箇所に当てはめる**ことがある（誤挿入）。

*効果の確認の手順（実機。音声が要る）。* `conversations.source_files` で `20261103_1930_f2f_teirei` の音声の場所を調べる（30日の削除期限内なら残っている。見つからなければ確認は飛ばす）。`hotwords` を書いた設定と書かない設定の2つで、`uv run kijun --config <設定> transcribe --audio <音声> --out <出力先>.json` を実行する。`--audio` は DB に書かないので、`utterances` を上書きしない。出力の JSON を比べて、(1) 2語が正しく取れたか、(2) 他の箇所が悪化していないか、(3) 誤挿入が起きていないか、を見る。

**VAD を切り、幻聴を除去する（2026-11-05、実機の測定に基づく）。** iPhone で録音した Discord の通話（36分）で、faster-whisper の Silero VAD（`vad_filter=True`）が発話の約 2/3 を非音声と判定して捨てた。同じ音声・同じモデルで測った結果は次のとおり。

| 設定 | 文字数 | 文字/分 | 所要 |
|---|---|---|---|
| `vad_filter=True`（変更前の既定） | 2,967 | 82 | 47 秒 |
| `vad_filter=False`、`condition_on_previous_text=False` | 9,348 | 260 | 119 秒 |
| 上に `word_timestamps=True`、`hallucination_silence_threshold=2.0` を足す（変更後の既定） | 9,081 | 252 | 148 秒 |

VAD を切ると、Whisper が無音部分に文を作る（「ご視聴ありがとうございました」「早送り」、1353〜1408 秒付近の鎧・猫・スイカについての約12区間の無意味な文など）。対処は2段である。

1. **faster-whisper の組み込みの対処。** `condition_on_previous_text=False`（幻聴が次の区間へ連鎖するのを止める）と、`hallucination_silence_threshold=2.0`（単語の時刻から、2秒以上の無音を挟む区間を飛ばす。`word_timestamps=True` が必要なので、設定が 0 より大きいときは自動で有効にし、0 のときは `word_timestamps=False`・`None` を渡す）。これで上の無意味な区間は消えたが、録音末尾の無音にある3区間（2122 秒「早送り」、2151 秒「早送り」、2158 秒「ご視聴ありがとうございました」）が残った。
2. **除去規則（`transcribe/merge.py` の `filter_hallucinated_segments`）。** 話者分離の後、話者の割り当て・結合（6-2）の前に、区間の「話者区間との重なりの割合」が `drop_max_speaker_overlap`（既定 0.3）未満、かつ `avg_logprob` が `drop_max_avg_logprob`（既定 -0.5）未満の区間を除く。重なりの割合は、区間のうち話者区間の**和集合**に覆われた秒数 ÷ 区間の長さである（話者区間どうしが重なることがあり、重なりの秒数を足すと 1 を超えたため）。この36分の通話では、残った3区間（いずれも重なり 0.00、`avg_logprob` は -0.5 未満）と、意味の無い断片 2 件（「心」「はい」）だけを除き、pyannote が拾えなかったが Whisper が確信していた実発話（「ちょっとこれ富田さんに」重なり 0.00・`avg_logprob` -0.32、「この初日さんのおかげでどうなったとか」重なり 0.00・`avg_logprob` -0.08）は残した。`no_speech_prob` は実発話でも 0.5〜0.8 になり、使えなかった。判断できない場合は除かない（話者区間が空、`avg_logprob` が `None`、`drop_max_speaker_overlap` が 0）。

**除いた区間は黙って消さない。** `transcribe_to_utterances` は `TranscribeResult(utterances, dropped)` を返し、`dropped` の各要素（`DroppedSegment`）は開始・終了秒、本文、`avg_logprob`、話者区間との重なりの割合、理由の文字列を持つ。文字起こしの JSON はこれを最上位のキー `dropped_segments` に書く（除去が無ければ空のリスト）。`kijun transcribe` は実行後に件数と各区間（開始時刻・本文・理由）を表示する。`restore-transcript`（6-3）は `dropped_segments` を読まないので、このキーが無い古い JSON も、あっても復元できる。夜間バッチ（15節）は `kijun transcribe --conversation` を別プロセスで呼ぶので、除いた区間の表示は各手順の標準出力に残る（バッチ側のコードは変えていない）。

**この閾値は1本の通話から決めた設計者の初期値である。** 対面の録音で VAD なし＋この設定が良いかは再測定していない。幻聴は非決定的で、同じ音声を2回処理すると同じ位置に別の作り話が出た。再文字起こしして `dropped_segments` を見て調整する。

faster-whisper の `segments` は generator なので、使う前に list にする。generator のまま次の処理に渡すと、モデルを解放した後に評価されて落ちる。

### 6-2. 区間の突き合わせ（`transcribe/merge.py`）

文字起こしの区間と話者の区間から `utterances` の行を作る。この前に、6-1 の除去規則（`filter_hallucinated_segments`）が幻聴の疑いがある区間を除く。`merge_segments` 自体は除去を知らず、残った区間だけを受け取る。

1. 各文字起こし区間について、時間の重なりが最も長い話者区間のラベルを割り当てる。重なりが全く無ければ `None`。
2. 同じ話者ラベルが連続し、かつ前の区間の終了から次の区間の開始までの間隔が `transcribe.min_utterance_sec` 未満なら、1つの発話に結合する。話者が不明（`None`）の区間どうしは、同じ話者と判断できないので結合しない。
3. 結合後の長さ（新しい発話の終了時刻 − 開始時刻）が `transcribe.max_utterance_sec`（既定30秒）を超えるなら結合せず、その区間から新しい発話を始める。順序は「間隔が近い**かつ**上限を超えない」なら結合である。`0` は上限なし（従来の挙動）。1つの文字起こし区間そのものが上限より長い場合は分割しない。区間の内部の時刻を使っておらず（`word_timestamps=True` にしているのは幻聴の対処のため。単語の時刻は保存しない）、分割すると文字と時刻の対応が壊れるからである。
4. `seq` を0から順に振る。

**上限を設けた根拠（2026-11-03、実機で初めて `kijun transcribe --audio` が最後まで通って判明）**：125秒の読み上げ音声を処理したところ、発話が2件しかできず（0.5〜22.0秒の21.5秒と、23.1〜124.8秒の101.7秒）、101.7秒の発話ができた。読み上げのように間が短い音声では、手順2の結合が際限なく続くためである。上限が無いことには実害が2つある。

- 抽出結果の `seq_from` / `seq_to` がどれも同じ発話を指し、録音のどの位置の発言かが±100秒の精度でしか分からなくなる。`item_evidence` から音声の位置に戻れなくなる。
- `extract/chunk.py` は、1つの発話が `chunk_minutes` を超える場合にその発話だけで1チャンクにし、分割しない。実際の会議で誰かが20分話し続けると1つの発話が20分になり、実測した入力長の上限（約10,000トークン、会議20分相当）を超えて抽出が失敗する。

既定値30秒は、通常の1文〜数文の長さに収まり、かつ位置の精度として許容できる値として置いた初期値である。運用して調整する。

**`min_utterance_sec` の既定を 0.4 から 2.0 にした（2026-11-05、ユーザー承認）。** `word_timestamps=True`（`hallucination_silence_threshold` > 0 で自動的に有効）だと、Whisper の各区間が最後の単語で終わるので、1人の話の中の間が 0.4〜2.6 秒の間隔として現れ、手順2で結合されずに細かく割れた。2本の録音での実測は、約2分の独話が 23 発話（0.4）から 7 発話（2.0）、36分の通話が 255 発話から 136 発話である。2.0 は設計者の実測値で、広い測定ではない。

この処理は純粋な計算なのでクラウド環境でテストできる。

### 6-3. 文字起こしの JSON と復元（`transcribe/restore.py`）

文字起こしの結果を `utterances` に書くと同時に、`paths.transcripts/<conversation_id>.json` に保存する。DuckDB を失ったときに文字起こしをやり直さずに済むようにするためである。

JSON には、`conversations` の行を再構成するのに必要な項目をすべて入れる：`conversation_id`、`recorded_at`、`setting_format`、`setting_kind`、`source_files`、`duration_sec`、`created_at`、`transcribed_at`、`utterances`。日時は `isoformat()` でマイクロ秒まで書く（秒未満を落とすと DuckDB の値と一致せず、復元しても元に戻らない）。`transcript_path` は JSON 自身のパスなので入れず、復元時に読み込んだファイルのパスを入れる。`transcribe --audio` による単発の試し実行の JSON は、会議に属さないので `conversation_id` が null になる。

`kijun restore-transcript` は `conversations` と `utterances` を復元する。`extracted_requests`、`knowledge_items`、`item_evidence` は JSON に入っていないので復元できず、`kijun extract` と `kijun match` のやり直しが必要であることを出力の最後に出す。既にある会議は上書きせずスキップする（原則2-4）。

### 6-4. 抽出プロンプト（`extract/prompt.py`）

`CLAUDE.md` 4-4 の「抽出プロンプトの要件（決定）」を満たす。入力は、話者ラベル付きの発話を `[seq] SPEAKER_00: 本文` の形で並べたもの。発話番号を行頭に入れるのは、抽出結果の `seq_from` / `seq_to` をモデルに書かせるためである。

要件のうち、プロンプトの中で繰り返し書いて守らせているのは次の2つである。

- **`stated_reason` と `applies_when` は、発言中で実際に語られた内容だけを書く。推測で補ってはならない。語られていなければ `null`。** プロンプトにはこの理由も書く。この仕組みの目的はユーザー自身が理由と適用条件を言語化することであり、LLM が推測で埋めるとその機会が失われる（`CLAUDE.md` 1章）。
- **（2026-11-05 追加）`stated_reason` と `applies_when` は、入力の本文から一字一句そのまま写した連続した一部分に限る。** 言い換え・要約・語尾の追加を禁じ、聞き取り誤りも直さずに写させる。本文に無い値はプログラムが捨てる（6-4-1）と、プロンプトに書いて伝える。例は会議と無関係な料理の話（「玉ねぎは先に炒めて。甘みが出るから」）にし、言い換えが誤りである悪い例と、理由が語られていない発言に例の文を写す悪い例を入れた。以前の例（「読む人は忙しいから」）は、qwen3:14b が文字起こしに無いのにそのまま写した（`CLAUDE.md` 7章）。
- **`pass_criterion` は Yes/No で判定できる形にする。** 例として `CLAUDE.md` の「冒頭2行以内に、読み手にしてほしい行動が書かれているか」を入れる。作れない場合は `null`。

`build_messages(chunk, cfg) -> tuple[str, str]`（system と user）という1つの関数にまとめてある。背面を増やしたときに文面が分かれないようにするためで、文面が違えば2回の抽出の比較（8節）が成立しない。

### 6-4-1. 原文照合（`extract/verbatim.py`）

`run_extraction` が、背面から受け取ったチャンクごとの結果に対し、保存の前に `stated_reason` と `applies_when` を原文と照合する。

- `normalize(text)`: Unicode NFKC にし、カテゴリが P（句読点）・Z（区切り）・S（記号）・C（制御）の文字と空白を除き、小文字にする。
- `is_verbatim(value, source)`: `normalize(value)` が空でなく、`normalize(source)` の部分文字列なら真。
- 原文 `source` は、モデルに渡したチャンクの**全発話**の本文を、区切りなしでつないだもの。モデルが返す `seq_from`〜`seq_to` は正確でないので、範囲には絞らない。
- `check_item` は、外れた値を `None` にし、元の値を `CheckedRequest.stated_reason_unverified` / `applies_when_unverified` に入れる。`CheckedRequest` は `ExtractedRequest` を継承した別のクラスで、モデルに渡す JSON スキーマ（`ExtractionResult`）には含まれない。含めると、モデルがこの2項目を自分で埋めようとする。
- 重複区間の除去（`merge_chunk_results`）を通った後の項目から、外した値の一覧（`rejections`）を作る。件数と値は `kijun extract` の標準出力と `extract_logs` の JSON（`verbatim_rejected`）に出す。

言い換えられた本物の理由も外れる。それは承知の上での取捨で、外れた値は `*_unverified` 列に残るので、後から数えられる（`CLAUDE.md` 7章の未検証事項）。`scripts/measure_extraction.py`（8-2）も同じ関数を通し、外した数を結果の表に出す。

### 6-5. チャンク分割（`extract/chunk.py`）

`CLAUDE.md` 4-4 の「20〜30分ごとに区切り、前後を2分程度重ねて分割処理する」を実装する。

- チャンクの境界は発話の境界に丸める。発話を途中で切らない。
- 1つの発話が `chunk_minutes` を超える長さ（長い独話）の場合は、その発話だけで1チャンクにする。
- LLM に見せる発話番号はチャンク内の0始まりで、返ってきた番号を会議全体の `seq` に復元する。チャンク範囲外の番号や逆順の番号は捨て、件数を数える。
- 重複排除：チャンク N と N+1 の重複区間に `seq_from` が収まる抽出結果のうち、後続チャンクから出たものを捨てる。保険として、`quote` の正規化（空白と句読点を除去）が一致し `seq_from` の差が2以内のものを1件にまとめる。**ただし前のチャンクが失敗している場合は捨てない。** 捨てると、その区間の結果がどちらからも得られなくなる。

**`chunk_minutes` と文脈長。** Ollama は `extract.ollama.num_ctx`（既定 16384）の中に入力と出力（`num_predict`、既定 8192）を入れる。そのため既定は `chunk_minutes = 12` / `overlap_minutes = 1` とした（12分 × `tokens_per_meeting_minute` 180 ≒ 2,160 トークン ＋ システムプロンプト ＋ 8192 が収まる）。**推奨・未確定（2026-11-05 設計者の初期値）で、実測に基づかない。** 収まらない設定は読み込みで拒否される。`chunk_minutes` を大きくするには `num_ctx` を上げる必要があり、KV キャッシュが増えて VRAM を使う。GPU に収まるかは `kijun doctor` の検査3で確かめる。以前の Strata では文脈長が 204,800 あり、2時間を1リクエストで渡せる可能性があった（その記述は削除した。当時の実測は12節）。

### 6-6. 抽出の背面（`extract/base.py`、`extract/ollama_impl.py`、`ollama_api.py`、`transport.py`）

2026-11-05 に、背面を Strata（Qwen3.8-Flash-Next）から Ollama に移した（13節）。`Extractor` は6つのメンバーを持つ。

```python
class Extractor(Protocol):
    model_name: str              # request_id と extracted_requests.model_name に使う（例: qwen3.5:9b）
    enforces_json_schema: bool   # format にスキーマを渡すか（設定 extract.ollama.enforces_json_schema）
    max_input_tokens: int | None # 入力に使ってよいトークン数。Ollama では num_ctx - num_predict
    repair_events: list[RepairEvent]  # 出力を修復した記録。run.py が extract_logs に書く
    last_usage: dict | None      # 直近の extract() の使用量（入力・出力のトークン数）。run.py が進捗の表示に使う
    def extract(self, chunk: Chunk) -> ExtractionResult: ...
```

`enforces_json_schema` と `max_input_tokens` を宣言させるのは、原則2-3のためである。背面ごと・設定ごとの挙動をハードコードせず、呼び出し側が判断できるようにする。

`ollama_impl.py` は Ollama のネイティブ API `POST /api/chat` に `transport.py` 経由で投げる。リクエストの組み立てと応答の読み取りは `ollama_api.py` にあり、抽出以外の LLM の用途（`review/llm.py`）、`doctor.py`、`scripts/measure_extraction.py` も同じものを使う。測るものと送るものを食い違わせないためである。

**OpenAI 互換の `/v1/chat/completions` を使わない理由は、`num_ctx` と `keep_alive` をリクエストごとに渡せないためである。** `num_ctx` を送らないと Ollama の既定の文脈長になり、入力が長いとエラーを出さずに先頭から切り捨てられる。

```json
{
  "model": "<extract.ollama.model>",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "stream": false,
  "format": "<ExtractionResult.model_json_schema()>",
  "keep_alive": "<extract.ollama.keep_alive>",
  "think": "<extract.ollama.think>",
  "options": {"temperature": 0, "num_ctx": "<num_ctx>", "num_predict": "<num_predict>"}
}
```

- `format` は `enforces_json_schema` が真のときだけ付ける。`think` は `None` のとき送らない（モデルの既定に任せる）。
- プロンプトは `prompt.py` の `build_messages` をそのまま使う。実装側で文面を足さない。
- `build_payload(cfg, chunk)` はモジュールの関数で、`doctor.py` と `measure_extraction.py` が同じ形を組み立てるのに使う。

**応答の扱い**

| 応答の項目 | 扱い |
|---|---|
| `message.content` | 本文。**スキーマを強制していても**必ず `repair.parse_with_repair` を通してから検証する。保証が実際に効いていれば、修復は何もしないので結果は変わらない |
| `message.thinking` | 推論の本文（`think` が真のとき）。保存しない。文字数を標準エラーに出す |
| `done_reason` | `"length"` なら出力が `num_predict` で切れた。警告を出す。完全な要素が1件も読めなければ**投げ直さず** `ExtractionError`（投げ直しに最大で `num_predict` 分の生成時間がかかるのに、同じ設定では同じように切れる可能性が高い。設定を直す方が速く確実） |
| `prompt_eval_count` | 入力のトークン数。**`>= num_ctx - num_predict` なら `ExtractionError`（投げ直さない）。** 下の「入力の切り捨て」 |
| `eval_count` | 出力のトークン数 |
| `eval_duration`、`total_duration`、`load_duration`、`prompt_eval_duration` | ナノ秒。`ChatReply` が秒に直す。標準エラーの表示と `doctor` の速度の見込みに使う |

`last_usage` は `{"prompt_tokens": prompt_eval_count, "completion_tokens": eval_count}` の形にする。`run.py` が読む形を変えないため。

**入力の切り捨て（原則2-4）。** Ollama は入力が `num_ctx` を超えると、**エラーを出さずに先頭を捨てる**。黙って捨てられた入力から抽出すると、捨てられた発話が抽出結果から静かに消える。そこで3か所で止める。(1) 設定の読み込み時に、`chunk_minutes × tokens_per_meeting_minute ＋ システムプロンプトの概算 ＋ num_predict` が `num_ctx` を超える設定を拒否する（`ConfigError`）。(2) `run.py` が送る前に、概算が `max_input_tokens` を超えるチャンクを警告する（処理は止めない。概算は外れ得る）。(3) 応答の `prompt_eval_count` が `num_ctx - num_predict` に達していたら `ExtractionError` にする。`measure_extraction.py` も同じ判定で、切り捨てられた疑いのあるチャンクを失敗として記録する。**限界：`prompt_eval_count` が返らなかったとき（プロンプトのキャッシュが効いた場合など）は判定できない。** 標準エラーに「確認できません」と出して続ける。`keep_alive = "0"` ならキャッシュは残らないので通常は返ると考えられるが、未確認である。

再試行は読めなかったときだけ1回。2回読めなければ `ExtractionError`。会議全体は落とさず、失敗したチャンクを記録して次に進む。接続できないときのエラー文面には、Ollama が起動しているか（タスクトレイ）、`ollama list` にモデルがあるか、`kijun doctor` で確認することを含める。HTTP 404 は `ollama pull <モデル>` を、`think` を送っているときの HTTP 400 で `think` に触れていれば `extract.ollama.think` を書かないことを案内する。

失敗の記録先は `<db と同じディレクトリ>/extract_logs/<会議ID>__<モデル>.json`（`:` などは `_` に置き換える）。スキーマに保存先の列が無いためファイルに置く。`compare` がこれを読む。

**モデル名の `:` の扱い（設計者の判断）。** Ollama のモデル名には `:` が入る（`qwen3.5:9b`）。`extracted_requests.model_name` には**元の名前を `:` ごと入れる**。`compare` と `match` と `pending` が設定の名前と一致で引くためである。`request_id`（`{会議ID}#{モデル}#{連番}`）にだけ、使える文字（英数字と `. _ + -`）以外を `-` に置き換えた名前を入れる（`extract/run.py` の `request_id_model`）。`run_label` を英数字と `. _ -` に限っている理由が「`request_id` にも入るから」であり、モデル名にも同じ規則を揃えた。`extract_logs` のファイル名は以前から `_` に置き換えていた。

`git` の履歴 7350993 にも Ollama の抽出の実装があるが、そのまま戻さず書き直した。以前の実装には欠陥が3つあった：`num_ctx` を送らない、`transport.py` と `repair.py` を通さない、使用量を記録しない。

`transport.py` の `TransportError` は、応答そのものが得られなかった場合（接続できない、時間切れ）にだけ出す。HTTP のエラー応答（400 など）は `HttpResponse.status_code` で表す。**サーバーが機能に対応していないこと（400）と、サーバーに届かないこと（接続不可）を呼び出し側が区別できるようにするため**で、`doctor` の判定がこの区別に依存している。

#### Strata のときの実機の記録（2026-11-03、記録として残す）

以下は背面が Strata だったときの事実で、Ollama には当てはまらない。比較の基準と経緯として残す。`CLAUDE.md` 7章にも同じ内容がある。

**2026-11-03 実機の訂正：`max_tokens` が無く、トークン数の概算が大きく外れていた。** 実機で初めて `kijun extract` を動かしたところ、次の事実が観測された。

- 2分5秒の音声（1チャンク）が、2,383 秒経っても応答を返さなかった。リクエストに `max_tokens` が無く、推論トークン（`reasoning_content`）を含む生成が `n_ctx`（204,800）まで走り得た。
- 上限なしで最後まで待った実測（curl）：入力 2,325、出力 13,050 トークン（`reasoning_tokens` という項目は返らず、推論は `completion_tokens` に合算される）。実時間 950 秒のうち生成が 932 秒、プロンプト処理が 18 秒。本文の JSON は約1,000トークンで、**残りの約12,000トークンは推論**。`finish_reason` は `stop`。
- したがって `max_tokens` は「本文に必要な長さ」では決められない。小さすぎると推論の途中で打ち切られて JSON が1文字も出ず、`repair.py` は完全な要素が1件も無い出力を修復しない（原則2-4）ので、そのチャンクは失敗になる。既定値 20000 は、この実測 13,050 に余裕を持たせた値で、実測は2回分（6,952 と 13,050）しか無い。`doctor` の推奨値で置き換える。
- 生成速度が同じ profile・モデルで **4.4 倍振れた**（`doctor` 検査4で 61.2〜68.9 tokens/s、上の実測で 14.0 tokens/s）。原因は特定できていない。`timeout_sec` を1回の測定の定数倍で決めてはいけない。
- トークン数の概算（日本語1文字 1.5 トークン）は、`doctor` の記録で概算 4,711 に対し実際 1,646（約2.9倍の過大）、上の実測でも概算 4,825 に対し実際 2,325（約2.07倍）だった。Qwen3.8-Flash-Next は 51B の n-gram 埋め込みテーブルを持つので、日本語のトークン効率が高いと考えられる。係数を設定値にし、既定を `tokens_per_character = 0.55`（実測 1,646 ÷ 約3,141文字 ≒ 0.52 に余裕を持たせた値）、`tokens_per_meeting_minute = 180`（従来の 500 を 0.55/1.5 の比で換算して切り下げ）にした。**この2つは実測1点からの推測であり、`doctor` が `tokens_per_character` を実測して推奨値に出す。**
- 追加の気づき：応答の `usage.prompt_tokens_details.cached_tokens` が 0 で、同じ system プロンプトを使う連続リクエストでもプロンプトキャッシュが効いていない。対応はしていない（12節の未確認事項）。

`extract` は進捗を標準エラーに出す。チャンクごとに、開始時（発話の範囲と推定トークン数）、完了時（所要秒数、応答の `usage` による入力・出力のトークン数、抽出件数。`usage` が取れなければその旨）、失敗時（何チャンク目か、所要秒数）。

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

トップレベルを配列から包み直すのは、`CLAUDE.md` 4-4 のプロンプト要件が「出力は JSON 配列のみ」であり、モデルが素直に従うと配列が返るためである。受け取る形は `{"items": [...]}` にしてある（Ollama の構造化出力がトップレベルの配列を扱いにくかったことに由来する形で、Strata でも、いまの Ollama の `format` でも同じ形を使っている）。

`parse_with_repair(text) -> ParsedExtraction` が、テキストから `ExtractionResult` を作る**唯一の経路**である。修復を通さずに検証する経路を残さないためである。読めなければ `UnrepairableJsonError` を出す。

要素の終わりを数えるとき、文字列の中の `}` や `]` は数えない。引用文に括弧が含まれると切り詰めの位置を誤るためで、テストで確認している。

完全な要素が1件も無いまま切れている場合は修復しない（原則2-4）。

### 6-8. 抽出結果の保存

`extracted_requests` に書く。`request_id` は `{conversation_id}#{model_name}#{連番}`。`direction` はこの時点では NULL（4-1の理由）。`source_type` は `live_conversation`。`stated_reason` / `applies_when` は原文照合（6-4-1）を通った値で、外れたときは NULL にし、元の値を `stated_reason_unverified` / `applies_when_unverified` に入れる。

`max_input_tokens`（Ollama では `num_ctx - num_predict`）が `None` でないとき、チャンクの推定トークン数がそれを超えていたら警告を出す。処理は止めない（概算は外れ得るので、応答の `prompt_eval_count` で最終的に判定する。6-6）。推定には `tokens.py` の係数を使う。`request_id` のモデル名は、使えない文字を `-` に置き換える（6-6）。

---

## 7. 推論サーバーの実測（`doctor.py`）

```
uv run kijun doctor [--out <パス>]
```

原則2-3の実体である。推測を測定に置き換える。固定の日本語の文を繰り返して入力を作るので、**音声が1本も無くても実行できる**（Ollama は起動してモデルを取得しておく）。`db init` の前でも実行できる。

Strata のときは「125B の MoE が64GB の PC で動くか」を測った。Ollama に移った理由は、Strata のプロセスが RAM を約56GB 保持して PC の他の作業を圧迫したことであり、**VRAM 12GB に収まる 9B〜14B のモデルが本当に GPU に収まっているか**が最も大事な検査になる（検査3）。一部がメインメモリに溢れていれば速度が落ち、同じ問題が再発する。

| # | 検査 | 内容 | 合格の条件 |
|---|---|---|---|
| 1 | 接続とモデルの有無 | `GET /api/tags` に設定のモデルがあるか。タグを省いた名前（`qwen3`）は `:latest` として探す | ある。無ければ `ollama pull <モデル>` を案内する。**失敗したら残りは実行しない**（同じ理由で失敗するため） |
| 2 | スキーマへの適合 | 固定の3発話を、抽出と同じ `build_payload` で（設定によらず `format` を付けて）送り、`repair` を通さずに読めるか | 読める（`direct_json`）。修復すれば読める（`repaired`）なら注意、読めないなら失敗。どちらも `enforces_json_schema = false` を推奨する |
| 3 | GPU に収まっているか | 検査2の直後に `GET /api/ps` を読み、対象モデルの `size_vram` と `size` を比べる | `size_vram >= size`。小さければ失敗とし、「一部がメインメモリに溢れている。`num_ctx` を下げるか、小さいモデルにする」と VRAM の割合を出す。**この検査がメモリの問題の再発を防ぐ要である** |
| 4 | 速度 | `chunk_minutes` 相当の長さの入力を送り、`eval_count` と `eval_duration` から tokens/s、所要時間を出す | 合否なし。出力が 50 トークン未満なら速度は誤差が大きいと注意を出す（固定の文を繰り返した入力では、モデルが短い出力しか返さないことがある） |
| 5 | 入力が切り捨てられていないか | 検査4の `prompt_eval_count` を `num_ctx - num_predict` と比べる。`tokens_per_character` の実測値も出す | `prompt_eval_count < num_ctx - num_predict` |

検査3のために、モデルが VRAM に残っている必要がある。doctor 自身の要求だけは `keep_alive` を `"2m"` にし（設定は `"0"` で、そのままだと検査2の直後に下りてしまう）、最後に `POST /api/generate {"model": ..., "keep_alive": 0}` でモデルを下ろす。検査1が失敗したときは何も読み込んでいないので、下ろす要求も送らない。

**2時間の会議の所要時間の見込み**を、2通りで出す。(a) 検査4の1チャンクの所要時間 × チャンク数（`ceil((120 - overlap) / (chunk - overlap))`）。`keep_alive = "0"` のときは、チャンクごとにモデルの読み込みが入るので、検査2で観測した読み込みの時間を足す。(b) **最も遅い測定値**での最悪値：チャンクごとに `読み込み ＋ 入力の処理 ＋ num_predict ÷ 最も遅い生成速度`。生成速度は検査2と検査4のうち出力が 50 トークン以上ある測定の最小を採る。Strata では同じ設定でも生成速度が 4.9 倍振れた実績があるため、1回の測定の値では決めない（`CLAUDE.md` 7章）。

最後に `config.toml` に書くべき推奨値を出し、同じ内容を `<db と同じディレクトリ>/doctor/ollama_<日時>.json` に保存する。

| 推奨値 | 求め方 |
|---|---|
| `enforces_json_schema` | 検査2が `direct_json` なら true、そうでなければ false |
| `tokens_per_character` | 検査4で送った文字数（system と user の合計）と `prompt_eval_count` から実測し、小数第2位に切り上げる。固定の文を繰り返した入力からの値なので、実際の会議ではずれ得る |
| `timeout_sec` | 1チャンクの最悪値の 1.5 倍（下限 600 秒、分単位に切り上げ）。最悪値が出せないとき（出力が短い）は、実測の所要時間の 3 倍 |
| `chunk_minutes` | 検査5が失敗したときだけ。半分にする。再度 doctor を実行して確かめること |

Strata のときの検査（推論の予算の逆算、`max_input_tokens` を求める段階的な入力の引き伸ばし、`max_tokens` と `chunk_minutes` の推奨）は削除した。当時の実測値は `CLAUDE.md` 7章と12節に残してある。

判定ロジックと推奨値の算出は、偽の `HttpTransport` を差し込んでクラウド環境でテストしている（`tests/test_doctor.py`）。実サーバーへの接続は `requires_network`、GPU に収まっているかの確認は `requires_gpu` のマーカーを付けてある。

---

## 8. 設定を変えた抽出の比較

### 8-1. DB に保存された2回の抽出の比較（`compare/report.py`）

```
uv run kijun compare --conversation <ID> [--model <名前> --model <名前>] [--out <パス>]
```

初版では `CLAUDE.md` 4-4 の「Qwen と Claude API の両方で抽出して承認件数を比較する」を支える道具だったが、2026-11-03 の決定で背面が Strata 1つになったため、**同じ背面で設定を変えた2回の実行を比べる道具**になった（2026-11-05 から背面は Ollama）。比べる対象は、モデルの違い、`think` の違い、`chunk_minutes` の違い、`enforces_json_schema` の違いなどである。`--model` を2回指定すると比べる2つを明示でき、省略すると設定の `model` を左に置く。

**承認件数そのものは出せない。** ユーザーがレビューして初めて決まる数字であり、レビューの受け皿は Discord（14節）で、承認は `review_log` に入るが、`compare` はそれを読まないからである。出せるのは次の2つである。

機械的に計算できるもの：モデルごとの抽出件数、`reusable = true` の件数と割合、`stated_reason` / `applies_when` / `pass_criterion` が非 NULL の件数、失敗したチャンク数、両者の対応付いた件数と片方にしか無い件数（`seq_from` の差が2以内かつ `quote` の difflib の類似比0.6以上で突き合わせる）。

`stated_reason` と `applies_when` が非 NULL の件数は、**推測で補っていないかの手がかり**として見る。この値が大きすぎる場合、モデルが語られていない理由を推測で埋めている疑いがある（6-4の要件違反）。

レビュー用の対照表：`seq_from` で並べ、同じ発話に対する2つの出力を左右に並べた Markdown の表。

**2回の実行を DB に並べて残すには `extract.run_label` が必要である。** `kijun extract` は同じ
`(会議ID, model_name)` の行を削除してから書く（`extract/run.py` の `delete_requests`）ので、
`run_label` が空のまま設定だけ変えて2回目を流すと、1回目が消えて比較できない。`run_label` を
入れると `model_name` が `<モデルID>+<label>` になり、2回分が並ぶ。API に送るモデルIDには付かない。

### 8-2. 設定ごとの測定（`scripts/measure_extraction.py`）

```
uv run python scripts/measure_extraction.py --conversation <ID> \
  [--config <パス>] [--variants <TOML>] [--out <ディレクトリ>]
```

8-1 は「DB に保存された結果」を比べる。それとは別に、**設定を変えた複数の実行を1回のコマンドで
行い、生の応答から比べる**道具が要ることが実機で分かった。2026-11-03 に実機で4回の測定を行ったが、
どの2回も条件が揃わなかった。改訂前のプロンプトでは `medium` が推測を混入させ、改訂後は対照の
`xhigh` が推論の予算切れで打ち切られていた。さらに、推論量が変わった原因が「プロンプトの改訂」か
「`reasoning_effort` を明示したこと」かも切り分けられなかった（背面が Strata だったときの話）。

原因は、比べる実行を人が別々のコマンドとして組み立てていたことである。そのたびに、設定以外のものが
変わる余地が残る。このスクリプトは1回の実行で複数の設定を順に投げるので、**設定以外（プロンプト、
`chunk_minutes`、入力の発話）が必ず揃い、結果の違いを設定の違いだけに帰せられる。**

`kijun extract` との違いは3点である。

1. **DB に書かない。** `repo.connect(..., read_only=True)` で開くので、`extracted_requests` を
   書き換えない。`run_label` の付け替えや `delete_requests` による取り違えが起きない。読むのは
   `utterances` だけである。
2. **1回の実行で複数の設定を試す。** 変えられるのは `model` / `think` / `num_ctx` / `num_predict`
   （と、必要なら `timeout_sec`）で、`[[variant]]` の TOML で指定する（`scripts/variants.example.toml` が例）。
   書かなかった項目は `config.toml` の値になる。TOML に null が無いので、`think` を書かないことが
   「variant では指定しない」を表し、`think = false` は明示的な上書きである。`Variant.applied_to` は
   設定を作り直して**検査をやり直す**ので、入力が `num_ctx` に収まらない組み合わせは、測る前に拒否される。
3. **文字起こしに無い語を機械的に検出する。** カタカナ3文字以上と漢字2文字以上の連なりを内容語の
   候補として拾い、会議の文字起こし全文に出てこないものを一覧にする。形態素解析を入れないのは、
   用途が「人が読む候補の一覧を出す」ことであり、取りこぼしより見落としを防ぐ方が重要だからである。
   1文字の漢字を拾わないのは、要約の言い回し（求めた、書く）で必ず出て一覧が読めなくなるためである。
   **検出された語のすべてが誤りではない。** 要約の言い換えで新しい語はできる。モデルが聞き取れない
   語に別の語を当てはめたものか正当な言い換えかは人が判断する。出力にその旨を明記している。

リクエストの本文は `kijun extract` と同じ `build_payload` で作る。測定したものと本番が送るものを
食い違わせないためで、テストでも両者が一致することを確認している。入力が切り捨てられた疑いのある
チャンク（`prompt_eval_count` が `num_ctx - num_predict` に達した）は、失敗として記録する。
切り捨てられた入力から出た結果を並べると、発話が欠けた結果を品質として比べてしまうためである。

`variants.example.toml` は次の4つで、モデル名は**推奨・未確定**（2026-11-05 設計者の初期値）。存在するかは
`ollama list` で確認する。

| label | model | think |
|---|---|---|
| `q35-9b-think` | `qwen3.5:9b` | true |
| `q35-9b-nothink` | `qwen3.5:9b` | false |
| `q3-14b-think` | `qwen3:14b` | true |
| `q3-14b-nothink` | `qwen3:14b` | false |

**比較の基準。** 出力の `.md` には、Strata の結果（2026-11-04、同じ会議 `20261103_1930_f2f_teirei`、同じプロンプト）を並べた表が入る。基準が使えるのはこの会議だけで、別の会議では「比べられない」と出る。

| 項目 | Strata `xhigh` | Strata `medium` |
|---|---|---|
| 抽出件数 | 6 | 6 |
| `stated_reason` / `applies_when` / `pass_criterion` が非 null | 4 / 2 / 4 | 4 / 1 / 4 |
| 文字起こしに無い語の数 | 5 | 17 |

加えて、人が確認する2点を、該当する項目の値として `.md` に出す（判定はしない）。(1) 「簡単フォーイス」の発話（`quote` に「フォーイス」を含む項目）が `transcription_unclear = true`、`pass_criterion` が null、`reusable` が false か。(2) `applies_when` に「参加者が募りたい場合」が取れているか。

**モデルの採否はユーザーが決める。** この道具は判断の材料を並べるだけで、どれがよいとは書かない。

出力は `<out>/<会議ID>_<日時>.json`（全設定の生の結果と要約）と同名の `.md`（人が読む対照表）である。

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
| `transcribe.min_utterance_sec` | `2.0` | 同一話者の連続区間を結合する間隔。話者が不明な区間どうしは結合しない（6-2）。2026-11-05 に 0.4 から変更（単語時刻で発話が細かく割れるため。ユーザー承認） |
| `transcribe.max_utterance_sec` | `30.0` | 結合した発話の長さの上限（秒）。`0` は上限なし。1つの区間そのものが上限より長い場合は分割しない。根拠は 6-2（実機で125秒の読み上げが101.7秒の発話を含む2件になった） |
| `diarize.num_speakers` | `0` | TOML に null が無いため、`0` を「指定なし」として扱う |
| `extract.provider` | `"ollama"` | 値は1つだが `Literal` のままにしてある。将来背面を足すときの型の受け皿 |
| `extract.chunk_minutes` / `overlap_minutes` | `12` / `1` | **推奨・未確定（2026-11-05 設計者の初期値）。** `num_ctx` 16384 の中に入力と出力が収まる大きさ（6-5）。`chunk_minutes > overlap_minutes` と、入力が `num_ctx` に収まることを読み込みで検査する |
| `extract.tokens_per_character` | `0.55` | 入力トークン数の概算の係数。Strata での実測 1,646 ÷ 約3,141文字 ≒ 0.52 に余裕を持たせた値で、**Ollama のモデルには当てはまらない推測**。`doctor` の推奨値で置き換える（7節） |
| `extract.tokens_per_meeting_minute` | `180` | 会議1分あたりの入力トークン数の概算。従来の 500 を 0.55/1.5 の比で換算して切り下げた値で、**推測** |
| `extract.ollama.base_url` | `"http://127.0.0.1:11434"` | `localhost` ではなく数字で書く。Windows では `localhost` が IPv6 の `::1` に解決されることがあり、サーバーが IPv4 だけで待ち受けていると接続拒否になる（`CLAUDE.md` 4-4 で踏んだ問題） |
| `extract.ollama.model` | `"qwen3.5:9b"` | **推奨・未確定。** 12GB の VRAM に収まる 9B〜14B から、測定（8-2）でユーザーが決める。存在するかは `ollama list`。`:` を含む名前の扱いは 6-6 |
| `extract.ollama.num_ctx` | `16384` | 文脈長。リクエストごとに送る。大きくすると KV キャッシュが増えて VRAM を使う。GPU に収まるかは `doctor` の検査3。**推奨・未確定** |
| `extract.ollama.num_predict` | `8192` | 出力の上限（推論を含む）。1024 未満と、`num_ctx` 以上は拒否する。`think` を有効にしたモデルは推論も数える。1チャンクの最悪の所要時間は `num_predict` ÷ 最も遅い生成速度。**推奨・未確定** |
| `extract.ollama.think` | 未指定（`None`） | `think` を送るか。TOML に null が無いので、書かないことで未指定（モデルの既定）を表す。測定（8-2）の対象 |
| `extract.ollama.keep_alive` | `"0"` | 応答のたびにモデルを VRAM から下ろす。原則2-2のため。`"0"` 以外にすると、手順1と6の後にアンロードの要求を送る（15-2）。**推奨・未確定**（応答のたびにモデルの読み込みが入る代わりの値） |
| `extract.ollama.timeout_sec` | `1800` | モデルの読み込みを含む。`doctor` の推奨値で置き換える。**推奨・未確定** |
| `extract.ollama.enforces_json_schema` | `true` | `format` にスキーマを渡す。出力は必ず `repair` を通す。`doctor` の検査2が読めないと出たら false |
| `discord.review_channel_id` / `contradictions_channel_id` | `""` | 空なら `post` と `read` は何もせず、設定が必要であることを出して終了コード 0 で終わる。チャンネルは個人サーバーに作る（`CLAUDE.md` 2章） |
| `discord.max_candidates_per_run` | `20` | 1回の `post` で投稿する新規候補の上限。`CLAUDE.md` 6章の「レビュー所要時間が15分を超え続けたら上限を下げる」の調整用。超えた分は次回に回る（14-2） |
| `discord.max_contradictions_per_run` | `5` | 1回の `post` で投稿する矛盾の組の上限 |
| `discord.contradiction_min_similarity` | `0.70` | 矛盾の候補とみなす類似度の下限。上限は `match.similarity_threshold`。**設計者が考案した規則の閾値で、運用して見直す**（14-7） |
| `discord.speaker_sample_count` | `3` | 話者の対応付けで投稿する発言サンプルの件数（`CLAUDE.md` 4-6） |
| `batch.log_dir` | `./data/batch_logs` | 夜間バッチの各手順の記録（`<日時>.json`）の保存先（15-4） |
| `batch.task_time` | `"23:30"` | 夜間バッチの開始時刻（HH:MM）。`kijun batch install-task` が `--time` を省略されたときに使う（15-5） |
| `batch.defer_if_running` | `[]` | 動いている間は GPU を使う手順を始めないプロセスの実行ファイル名。空なら検出しない。**ユーザーが自分の環境で確認して書く（推奨・未確定）**（15-2） |
| `batch.defer_check_interval_min` | `30` | 延期中に再確認する間隔（分）。**推奨・未確定** |
| `batch.defer_max_hours` | `4` | 延期する時間の上限（バッチ全体の合計）。超えたらその夜の GPU を使う手順を飛ばす。**推奨・未確定** |
| `transcribe.vad_filter` | `false` | 2026-11-05 に `true` から変更。通話の録音で発話の約 2/3 を捨てた（6-1）。対面の録音では再測定していない。設計者の初期値 |
| `transcribe.condition_on_previous_text` | `false` | VAD を切ったときの幻聴の連鎖を止める（6-1）。設計者の初期値 |
| `transcribe.hallucination_silence_threshold` | `2.0` | 秒。0 なら使わない。0 より大きいと `word_timestamps` が自動で有効になり、所要時間が約 1.2 倍（119 → 148 秒、36分の音声）。1本の通話に基づく設計者の初期値（6-1） |
| `transcribe.drop_max_speaker_overlap` / `drop_max_avg_logprob` | `0.3` / `-0.5` | 話者区間との重なりの割合が前者未満、かつ `avg_logprob` が後者未満の区間を除き、JSON の `dropped_segments` に残す。前者が 0 なら除去しない。1本の通話に基づく設計者の初期値（6-1） |
| `transcribe.hotwords` | `[]` | faster-whisper の `hotwords`。空なら渡さない。効果は未検証で、その語を言っていない箇所に当てはめる副作用がある（6-1） |
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
| `transcribe` | av、faster-whisper、pyannote.audio、torch、torchaudio |
| `extract` | httpx |
| `match` | sentence-transformers |
| `discord` | discord.py（無料。運転費用はゼロのまま） |

**torch の CUDA 版について。** `pyproject.toml` に CUDA 版のインデックス指定（`[[tool.uv.index]]` と `[tool.uv.sources]`）を書いてあるが、**コメントアウトしてある**。`uv lock` は extra を使わなくても `download.pytorch.org` に接続し、開発環境の proxy がそれを 403 で拒否して `uv sync` が失敗するためである。Windows では**コメントを外してから** `uv sync --extra transcribe` を実行する。タグは 2026-11-03 に https://pytorch.org/get-started/locally/ で確認した `cu132`（CUDA 13.2、torch 2.15.0、Windows）を書いてある。CUDA 13.4 を選ぶ場合は `cu134`。

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
uv run kijun batch install-task [--time HH:MM] [--out <パス>]  # タスクスケジューラ用の XML を生成（--time の既定は batch.task_time）
```

`transcribe --audio` は DuckDB に書かず、結果を標準出力と JSON に出すだけである。`CLAUDE.md` 8章 #4 の「1ファイルで faster-whisper と pyannote を試すスクリプト」が、登録を経ずに1本の音声で試せる必要があるためで、ユーザーが最初に実行するコマンドになる。

`match --provider` があるのは、2つの設定で抽出した場合に同じ指摘から項目が二重にできるのを防ぐためである。

### 11-3. テスト

自動テストは35ファイル・719件で、GPU・ネットワーク・Discord のトークンが要る10件は既定で実行しない（`pyproject.toml` の `addopts` に `-m 'not requires_gpu and not requires_network'` を書いてある）。実機では `uv run pytest -m requires_gpu` と `-m requires_network` で実行でき、試験用の音声のパスは環境変数 `KIJUN_TEST_AUDIO` で渡す。

クラウド環境で検証できるのは、第2弾の文面の生成、投稿の順序と件数の上限、リアクションの判定（判定できない場合を含む）、返信の原文の保存、話者の対応付けの解釈、`direction` の導出、矛盾の検出、スキーマの移行、夜間バッチの手順と失敗時の動き、タスクスケジューラの XML、および設定の読み込み、ファイル名の解析、会議単位への束ね、DuckDB のスキーマと読み書き、チャンク分割と発話番号の復元、重複排除、区間の突き合わせ、壊れた JSON の修復、コサイン類似度と閾値の分岐、音声の削除の対象選定、文字起こしの書き出しと読み戻しの往復、`doctor` の判定ロジックと推奨値の算出、ゲームの実行中の延期（偽のプロセス一覧と偽の `sleep`）、Ollama の応答の扱い（偽の `HttpTransport`）である。

検証できないのは、faster-whisper、pyannote、Ollama、sentence-transformers の実呼び出しと、torch の CUDA 版のインストール、**discord.py の実接続**、**Windows のタスクスケジューラへの登録**である。

### 11-4. 実機での確認の順序

| # | 作業 | 成功の判定 |
|---|---|---|
| 1 | Python 3.12 と uv を入れる | `uv --version` が出る |
| 2 | `uv sync` | `uv run kijun --help` がサブコマンド一覧を出す |
| 3 | `uv run pytest` | 709件通り、10件が除外される |
| 4 | `uv run kijun db init` | `data/kijun.duckdb` に10テーブルとビュー1つができる |
| 5 | Ollama for Windows を入れ（ollama.com）、`ollama pull qwen3.5:9b` と `ollama pull qwen3:14b` を実行して、`uv run kijun doctor` | 検査1〜5が出る。**検査3（GPU に収まっているか）が「良」であること。** 音声が無くても実行できる。出た推奨値を `config.toml` に書き写す |
| 5b | ゲームを起動した状態で、タスクマネージャーの「詳細」タブの実行ファイル名を確かめ、`batch.defer_if_running` に書く | 手順14のバッチで、ゲームが動いている間は GPU を使う手順が延期される |
| 5c | `uv run python scripts/measure_extraction.py --conversation 20261103_1930_f2f_teirei --variants scripts/variants.example.toml` | 4通りの対照表が出る。Strata の基準（8-2）と並べて、ユーザーがモデルと `think` を決める |
| 5d | `hotwords` の効果の確認（6-1） | 「箇条書き」「感嘆符」が正しく取れ、他の箇所が悪化せず、誤挿入が無い |
| 6 | `pyproject.toml` の torch のインデックスを有効にして `uv sync --extra transcribe` | `uv run python -c "import torch; print(torch.cuda.is_available())"` が True |
| 7 | Hugging Face で pyannote の3つのモデルページの利用条件に同意し `HF_TOKEN` を設定 | 次の手順が通る |
| 8 | 5分程度の音声1本で `uv run kijun transcribe --audio <ファイル>` | 日本語の文字起こしと `SPEAKER_00` 等のラベルが出る。処理時間と VRAM 使用量を記録する（別の端末で `nvidia-smi`） |
| 9 | 2時間の音声で `uv run kijun extract` | 抽出が取れる。所要時間とメモリ使用量を記録する |
| 10 | `uv run kijun match` | 初回は全件が新規候補になる |
| 11 | Discord Developer Portal でアプリケーションと bot を作り、Message Content Intent を有効にし、個人サーバーに招待する。`uv sync --extra discord` と `DISCORD_BOT_TOKEN` の設定、`config.toml` にチャンネルIDを書く | `uv run pytest -m requires_network`（`KIJUN_TEST_DISCORD_CHANNEL` に試験用チャンネルのID）で投稿・リアクション・返信の取得が通る |
| 12 | `uv run kijun discord post` | 新規候補と話者の対応付けが Discord に出る。✅などのリアクションとスレッドへの返信を付ける |
| 13 | `uv run kijun discord read` | リアクションが `knowledge_items` に、返信の原文が `review_log` に入る。返信の振り分けには Ollama が要る |
| 14 | `uv run kijun batch run --dry-run`、続けて `uv run kijun batch run` | 手順と順序が出る。実行すると `data/batch_logs/` に記録が残り、最後に成否の表が出る |
| 15 | `uv run kijun batch install-task` が作った XML を `schtasks /create /xml` で登録する | タスクスケジューラに登録され、翌晩に起動する（スリープ解除を含む） |

手順5、8、9で記録した値は、12節の未確認事項への回答になる。手順11から15は、14節と15節の未確認事項（Discord への実接続、スリープ解除）への回答になる。結果を `CLAUDE.md` 7章に書き込む。

手順1から4は、音声が1本も無くても、Ollama が無くても実行できる。手順5は Ollama とモデルがあれば音声なしで実行できる。手順8から10は音声が必要で、`CLAUDE.md` 8章 #1〜#3（iCloud for Windows の導入、ショートカットの作成、通話の録音テスト）が終わっていることが前提である。

---

## 12. 未確認事項と、確認する手段

2026-11-03 から 11-04 の実機での確認で、この表の多くが解決した。解決したものは取り消し線にして、
実測値を残してある。`CLAUDE.md` 7章にも同じ結論が書いてあり、そちらが正本である。

**まだ確認できていないこと**

| 事項 | 現状 | 確認する手段 |
|---|---|---|
| **Ollama のモデルが GPU（VRAM 12GB）に収まるか、速度、品質** | **未確認（2026-11-05 の移行で新しくなった最大の未確認事項）。** `extract.ollama` の既定（`qwen3.5:9b`、`num_ctx` 16384、`num_predict` 8192、`keep_alive` "0"、`chunk_minutes` 12）は設計者の初期値で、実測に基づかない。モデル名が存在するかも未確認 | `kijun doctor`（検査3が GPU に収まっているか）。`scripts/measure_extraction.py` で品質を比べる（8-2） |
| **Ollama の `format` が出力をスキーマに適合させるか。`think` と同時に送ってよいか** | 未確認。版とモデルによっては、推論にも制約が掛かって壊れる報告がある | `doctor` の検査2。`think` を変えた測定（8-2） |
| `prompt_eval_count` が返らないことがあるか | 未確認。返らなければ、入力の切り捨てを検出できない（6-6） | 実機の抽出の標準エラー（「確認できません」が出るか） |
| ~~`hotwords` の効果と副作用~~ | **2026-11-05 測定。効果が無かった。** 対面の録音で、あり・なしとも「過剰書き」「簡単フォーイス」のまま。誤挿入も無かった（`CLAUDE.md` 7章） | 次の手段は `Transcriber` の背後に別のモデル（kotoba-whisper など）を足して比べること。未着手 |
| `applies_when` の取りこぼし | Strata の `medium` は、文字起こしに実在する「参加者が募りたい場合」を落とした（`CLAUDE.md` 7章）。Ollama のモデルで起きるかは未確認。`applies_when` は `CLAUDE.md` 1章が暗黙知の中身として挙げる2点のうちの1つである | 8-2 の「人が確認する2点」。運用では、レビューでユーザーが `applies_when` を書き足した頻度 |
| `transcription_unclear` の投稿への返信を受け付けるか | **2026-11-04 ユーザーが「まず現状で運用して測る」と決定。** いまはリアクションだけを読み、返信は読まない。`review_log` が `item_id` NOT NULL で記録する先が無いためである。**このため、この投稿は「思い出す機会」は渡すが「書く先」を渡していない。** ✅ を押しても、思い出した内容はどこにも残らない。7章の案Aの目的（本人が正しい項目を書ける機会を渡す）は、半分しか果たせていない | `paths.review_logs` に書き出す実行の記録から、✅ / ❌ / 無反応の件数を数える（数え方と閾値は `CLAUDE.md` 10章）。閾値を超えたら、`review_log.item_id` を NULL 許容にして `request_id` を足し（スキーマ v4）、返信を `knowledge_items` の新規行として起こす |
| 読める語に化けた聞き取り誤りを検出できるか | **未確定。実機で1件見落とした。** 「新しい」が「明日らしい」に化けた箇所を、モデルは `transcription_unclear = false` と判定し、さらに `request_summary` に「明日のリンク」という発言に無い内容を書いた。`reusable = false` なので `knowledge_items` には入らないが、`transcription_unclear = false` なので新しい投稿種別にも出ない。**7章の案Aが塞ごうとした穴に、この発話はそのまま落ちる** | プロンプトに「語そのものは読めても文全体として求めている内容が成り立たないなら疑う」という指示を足した（例は別の作り物にしてある。「明日らしい」を書くと試験に使える語を答えとして教えることになる）。効くかは次の実録音で測る |
| トークン数の概算の係数 | 日本語1文字あたり 0.55 トークン、会議1分あたり 180 トークン。**Strata での実測1点からの推測**で、Ollama のモデルには当てはまらない | `doctor` の検査4・5が `tokens_per_character` を実測して推奨する |
| プロンプトキャッシュ | `usage.prompt_tokens_details.cached_tokens` が 0。同じ system プロンプトの連続リクエストでキャッシュが効いていない | 未対応。気づきとして記録 |
| ゲームの実行ファイル名と、延期の動作 | ユーザーが `batch.defer_if_running` に書く名前が実際のプロセス名と一致するかは未確認。一致しなければゲームが動いていても延期されない | 手順5b。ゲームを起動した状態で `uv run kijun batch run` を試す |
| 通話を PC スピーカー出力で録音したときに相手の声が十分に入るか | 未確認（`CLAUDE.md` 7章） | 30分の実通話テスト |
| `artifact_type` と `counterpart_role` の値の一覧 | LLM が自由な文字列を入れる。候補を固定していない | 3回分の抽出結果を見て、よく出る値を `config.toml` に列挙する |

**解決したこと（2026-11-03〜04 の実機での確認。背面が Strata だったときのもの）**

| 事項 | 結論 |
|---|---|
| ~~Strata にモデルを VRAM から降ろす手段があるか、Strata の起動にかかる時間、`start_command` が戻るコマンドとして書けるか~~ | **2026-11-05 に Ollama へ移ったので問わなくなった。** Ollama は `keep_alive` でモデルを下ろせる（15-2）。`start_command` / `stop_command` / `batch.strata_ready_timeout_sec` は削除した |
| ~~1リクエストの入力長の上限、生成速度の振れの原因~~ | Strata（約10,000トークン＝会議20分相当までは通った。速度は 14.0〜68.9 tokens/s の振れ）の記録。Ollama では `doctor` で測り直す |
| ~~Qwen3.8-Flash-Next が 12GB VRAM + 64GB メモリで動くか~~ | **動く。** 量子化は IQ3_XXS、モデルID は `qwen3.8-flash-next-iq3_xxs`。`strata` プロセスが約56GB（実メモリ約42GB）を保持して動作した。Unsloth の「1bit量子化で最低75GB」は Strata の動的オフロードには当てはまらなかった |
| ~~Strata が `response_format` による JSON スキーマの強制に対応するか~~ | **対応する。** `doctor` の検査3が「200 でスキーマに適合」。`config.toml` は `enforces_json_schema = true`。`repair.py` は保険として残す |
| ~~Strata の OpenAI 互換のパス~~ | `/v1/chat/completions`。`/v1/models`、`/v1/messages`（Anthropic 互換）、`/v1/responses` も同じポートにある |
| ~~Strata の応答に `usage` が含まれるか~~ | 含まれる（`prompt_tokens` と `completion_tokens`）。推論のトークンは `completion_tokens` に合算され、`reasoning_tokens` という項目は返らない |
| ~~`max_tokens` の働き~~ | **推論を途中で切らない。** Strata は `max_tokens` から推論の予算を逆算し、残りを答え用に取り置く（取り置き = `min(max(512, max_tokens // 4), max_tokens // 2)`）。予算に達したら推論を閉じて答えを書かせる。**1024 未満では逆算が働かず、推論が本当に切られる**ので設定の読み込みで拒否している。送らないと予算が文脈長から逆算されて 151,851 になり、2分5秒の音声1チャンクで最悪 2.24 時間かかる。抽出が遅かった原因はこれだった |
| ~~2時間の会議が夜間バッチに収まるか~~ | **収まる。** `max_tokens = 20000` が1チャンクの最悪値を 20000 ÷ 14.0 ≒ 1,429 秒に確定させる。2時間の会議は約6チャンクなので最悪 2.38 時間。23:30 開始なら 01:53 頃 |
| ~~pyannote.audio が Windows + CUDA で動くか~~ | **動く。** 2分5秒の音声を、モデルの読み込みを含めて 18.5 秒で処理。pyannote.audio 4.0.7、torch 2.15.0+cu132。利用条件への同意はモデルページ3つに必要（`CLAUDE.md` 4-3） |
| ~~torch の CUDA のタグ~~ | `cu132`（CUDA 13.2、torch 2.15.0、Windows） |
| ~~Qwen3.8-Flash-Next の量子化版の入手先とタグ~~ | IQ3_XXS。`/v1/models` が返す名前は `qwen3.8-flash-next-iq3_xxs` |
| ~~`reasoning_effort` に `"xhigh"` が有効か~~ | **有効。** 検証を行う `frontend.py` の `EFFORT` 表に `"xhigh"` がある。不正な値への 400 のメッセージが「use none, low, medium or high」と列挙するのは、メッセージの文字列が表の内容を反映していないだけである。**`high` と `xhigh` は同一**（どちらも内部の `xhigh` に写る）。また `chat_template.jinja` が `reasoning_effort|default('xhigh')` なので、**送らないことは `xhigh` を送ることと同じ**である。既定を `"xhigh"` と明示したのは、テンプレートの既定が将来変わったときに挙動が黙って変わるのを防ぐため |
| ~~明示した `reasoning_budget_tokens` が逆算より優先されるか~~ | **優先されるが、そのあと答え用の取り置きで切り下げられる。** 実効の予算 = `min(reasoning_budget_tokens, max_tokens - 取り置き)`。取り置きの額は指定できない。`max_tokens = 48000` で予算 40000 を明示すると 36,000 に切り下がる（ログに `thinking capped at 36000 of max_tokens 48000 ...`）。推論の予算を上げるには `max_tokens` も上げる必要がある |
| ~~抽出の推論量を `medium` にしてよいか~~ | **2026-11-04 ユーザーが `medium` を採用。** 改訂プロンプトでは `medium` でも推測の混入は起きず、速度が約5.9倍。失うのは `applies_when` 1件。所要時間を取る判断をした。残った risk は上の表に移した |
| ~~`max_tokens` の既定 20000 が `xhigh` に足りない~~ | **`medium` を採用したので解消。** medium の推論は 2,641 トークンで、20000 の予算 15,000 に収まる。`xhigh` に戻す場合は 32000 以上が必要で、その旨を設定のコメントに書いてある |
| ~~文字起こしが壊れた発話から作られた項目をユーザーに見せるか~~ | **2026-11-04 ユーザーが案A を採用。実装済み（スキーマ v3）。** `transcription_unclear` 列を足し、Discord の投稿種別 `transcription_unclear` で `#review` に出す。詳細は 14-2 と `CLAUDE.md` 7章 |
| ~~移行した DB と新規に作った DB でスキーマが食い違わないか~~ | **型・NOT NULL・DEFAULT は一致する。違うのは列の順序だけ**（2026-11-04 実機で実データの DB を移行して確認）。新規は `pass_criterion, transcription_unclear, source_type, ...`、移行は `..., extracted_at, transcription_unclear` になる。`ALTER ADD COLUMN` は末尾に足すので、この差は消えない。**無害である**ことを確認した。`repo.py` の INSERT はすべて列名を明示しており（`_REQUEST_COLUMNS` / `_POST_COLUMNS`）、`src/` に `SELECT *` は1件も無いので位置依存が無い。ただし今後 `INSERT INTO ... VALUES` を列名なしで書くと、新規 DB と移行 DB で別の列に入る |
| ~~推論が予算に達したかをどこで確認するか~~ | **サーバープロセスの標準出力の `[strata] thinking:` の行だけ。** API の応答には出ない。Strata の `--log` が指すファイル（`strata-iq3_xxs.log`）にも入らない（そのファイルはエンジンの子プロセスに渡されるもので、`[strata]` の行はサーバープロセスが標準出力に print している）。PowerShell なら `Start-Process -RedirectStandardOutput` で落とす |

第2弾（Discord でのレビューと夜間バッチ）で未解決のまま残ることは次のとおりである。

| 事項 | 現状 | 確認する手段 |
|---|---|---|
| Discord に実際に接続したときの動作 | **未検証。** ユーザーが Developer Portal でアプリケーションと bot を作り、Message Content Intent を有効にし、個人サーバーに招待するまで確認できない（`CLAUDE.md` 4-6）。`impl.py` は REST のみで読み書きする設計だが、`fetch_channel` / `history` / スレッド ID が元のメッセージの ID と同じという前提は実接続で確かめていない | 実機手順11。`requires_network` のテスト |
| 矛盾の検出規則 | 設計者が考案したもので、`CLAUDE.md` に根拠が無い。閾値を設定値にしてあるので運用して見直す | 数週間運用して、投稿された組のうち適用条件の追記につながった割合を見る |
| 重要度の3段階の基準 | プロンプトの文面（3=目的が果たせなくなる、2=守る方が明らかに良い、1=好みや体裁）は設計者が書いたもの。`CLAUDE.md` に基準は無い | 上書きの履歴（`importance_overridden` の `before` と `after`）を見て直す |
| 重要度の回帰による再推定 | `CLAUDE.md` 4-6 の「2〜3か月後」の作業。上書き履歴を残す形だけ決めてある | 2〜3か月分の `review_log` |
| 束ね結果に ❌ が付いたときの再分割 | 自動では行わない。ユーザーが `bundle.gap_minutes` を調整して `kijun ingest` をやり直す | — |
| 1日のレビュー所要時間の測定 | `discord_posts.posted_at` と `review_log.reviewed_at` の差から概算できるが、ユーザーが画面を見ていた時間とは一致しない | 運用してからビューの形を決める |
| タスクスケジューラのスリープ解除 | 未確認。XML の `WakeToRun` で指定したが、PC の電源設定（スリープ解除タイマーの許可）に左右される | 実機手順15の翌朝に `batch_logs` の時刻を見る |
| 話者の対応付けの返信の原文が残らない | `review_log` は `item_id` が NOT NULL で、項目に属さない出来事を入れられない。`speaker_map` の結果だけが残る。LLM が自然文を誤って解釈しても、原文から直せない | 問題になったら `review_log` を `conversation_id` も持てる形にする（スキーマ変更） |
| 処理済みの投稿への後からの返信 | 取り込まない（`processed_at` を埋めた投稿は読み直さない）。承認のリアクションを付けた後で、次の `read` までの間に書いた返信は取り込まれる | 運用で、リアクションと返信を同じ時間帯に付ける |
| 再抽出した後の `direction` | `extract` をやり直した会議の `direction` は NULL に戻る。`kijun derive-direction --conversation <ID>` で導出し直す（自動では行わない） | — |
| 人物ラベルの表示 | 投稿の「（あなた→kouhai_a）」は、`speaker_map` の `person_label` をそのまま出す。`kouhai_a` を「後輩A」と書き換える表は持っていない | 必要になったら `format.py` に辞書を足す |

---

## 13. 変更の履歴

### 2026-11-05：原文照合と `min_utterance_sec`

**背景。** qwen3:14b（`think` あり）の測定で、`stated_reason` に文字起こしに無い値（プロンプトの例の写し）や言い換え、推測が出た。同じ入力でも実行ごとの振れが大きかった（`CLAUDE.md` 7章）。

**ユーザーが決めたこと（`CLAUDE.md` 10章にも記録した）。** (1) `stated_reason` / `applies_when` は文字起こしの本文からそのまま写した値に限り、プログラムで照合する。外れた値は NULL にして元の値を別の列に残す。(2) `transcribe.min_utterance_sec` の既定を 2.0 にする。(3) Windows の torch の CUDA 版 index を `pyproject.toml` で既定で有効にする。

**実装したこと。** `extract/verbatim.py`（6-4-1）、スキーマバージョン4（`extracted_requests` に 2 列）、プロンプトの書き直し（6-4）、`kijun extract` の出力と `extract_logs` への外した値の記録、`scripts/measure_extraction.py` への照合の適用（8-2）、`min_utterance_sec` の既定（6-2）。

### 2026-11-05：抽出の背面を Strata から Ollama に移した

**背景。** Strata（Qwen3.8-Flash-Next、125B の MoE）は動作したが、プロセスが RAM を約56GB 保持し、PC の他の作業を圧迫した。文字起こしと抽出を無料で行うことを前提に、Strata 以外の背面に移した。

**ユーザーが決めたこと（`CLAUDE.md` 10章にも記録した）。**

| 対象 | 決定内容 |
|---|---|
| 抽出の背面 | Ollama で、GPU（VRAM 12GB）に収まる 9B〜14B のモデルを動かす。データは PC の外に出さない（`CLAUDE.md` 2章の参加者への説明を変えない）。無料のクラウド API（Gemini・Groq）と `claude -p` は、同意の取り直しが要るので採らない |
| LLM の他の用途 | 返信の振り分け・重要度の判定・話者の解釈も、同じ Ollama を使う |
| Strata | コードを削除する（`strata_impl.py`、`batch/strata.py`、`doctor` の Strata 用の検査、`StrataConfig`）。`CLAUDE.md` 7章とこの文書にある Strata の実測の記録は残す。経緯と、比較の基準として使うため |
| 文字起こし | faster-whisper ＋ pyannote のまま。加えて、頻出語の誤認識への対策として `hotwords` を設定から渡せるようにする |
| ゲームとの競合 | 夜間バッチの開始時刻を設定で変えられるようにする。加えて、指定したゲームのプロセスが動いている間は、バッチを延期する |

**設計者（Claude）が決めたこと（推奨・未確定）。** 既定値：モデル名 `qwen3.5:9b`、`num_ctx` 16384、`num_predict` 8192、`chunk_minutes` 12 / `overlap_minutes` 1、`keep_alive` "0"、延期の上限 4 時間・確認の間隔 30 分。どれも実測に基づかない初期値である。ほかに、(1) モデル名の `:` を `model_name` に残して `request_id` でだけ置き換えたこと（6-6）、(2) `tasklist` が失敗したら警告して続けること（15-2）、(3) `hotwords` の既定を空にしたこと（6-1）、(4) `keep_alive` が "0" 以外のときのアンロードを実装したこと（15-2）。

**実装したこと。** 設定 `[extract.ollama]` と、入力が `num_ctx` に収まるかの読み込み時の検査（6-6、10節）。`ollama_api.py` と `extract/ollama_impl.py`（6-6）。`review/llm.py` の `JsonChat` / `OllamaJsonChat` とクラス名の変更（`LlmReplyRouter` など。14節の LLM 用途）。夜間バッチの Strata の窓の削除と、ゲームによる延期・開始時刻の設定（15-2、15-5）。`doctor` の書き直し（7節）。測定スクリプトの variant の置き換えと、Strata の基準の併記（8-2）。`transcribe.hotwords`（6-1）。

**指示書と食い違えた判断。** (1) `hotwords` の例（`["箇条書き", "感嘆符"]`）は、`config.example.toml` でコメントとして示し、既定は空のままにした。`tests/test_config.py` が `config.example.toml` と既定値の一致を検査するため、例を有効にすると既定が変わることと、効果が未検証であることによる。(2) 指示書の `build_llm_parts` は、実際の関数名が `build_llm` で、名前は変えていない。(3) 指示書にない追加：`doctor` は検査1が失敗したら残りを実行しない、`doctor` の結果に2時間の会議の見込みを入れた、測定スクリプトに「人が確認する2点」の抽出を入れた、`install-task` のテストが利用者の `config.toml` を読まないよう作業ディレクトリを一時ディレクトリにした。(4) 指示書は `chunk_minutes` の検査を「入力が `num_ctx` を超えるなら拒否」としており、システムプロンプトの概算に `SYSTEM_PROMPT` の文字数 × `tokens_per_character` を使った（概算なので、実際の `prompt_eval_count` での確認が最終判定になる）。

**この変更で失われたもの。** Strata の実測に基づく設定値（`max_tokens = 20000`、`reasoning_effort = "medium"` など）は、Ollama では意味を持たないので削除した。Strata の記録は6-6の末尾と12節、`CLAUDE.md` 7章に残してある。

### 2026-11-04：推論量の測定と、その結果による2つの決定

**測定の道具を足した。** 11-03 の測定は、比べる実行を人が別々のコマンドとして組み立てていたため、どの2回も条件が揃わなかった。`scripts/measure_extraction.py`（8-2）は1回の実行で複数の設定を順に投げるので、設定以外が必ず揃う。文字起こしに無い語を機械的に検出する機能も入れた。

**ユーザーが2つを決めた。**

1. 抽出の推論量を `medium` にする。改訂プロンプトでは `medium` でも推測の混入が起きず、所要が約5.9倍速い。残る risk（`applies_when` の取りこぼし）は12節に移した。
2. `CLAUDE.md` 7章の案A を採る。`transcription_unclear` 列を足し、文字起こしが読めなかった抽出結果を Discord に出す。スキーマをバージョン3にした。

**11-03 の見立てを1つ取り消した。** 「改訂プロンプトが `xhigh` の推論量をほぼ倍にした」は誤りだった。`chat_template.jinja` が `reasoning_effort|default('xhigh')` なので、`reasoning_effort` を送るかどうかで推論量は変わらない。14,967 は推論量ではなく予算15,000の天井そのものだった。

**設計者（Claude）が書いた誤りを2つ直した。** `scripts/variants.example.toml` の「`max_tokens` 48000 は予算 40000 に対して答え用に 8000 残す」は成立しない（取り置きの額は指定できず、予算は 36,000 に切り下がる）。`max_tokens = 20000` の根拠として書いていた「実測 13,050 に余裕を持たせた値」も取り消した（13,050 は `max_tokens` を送らずに測った1点で、同じ入力でも 48,764 以上まで振れていた）。

### 2026-11-03：実機で `kijun extract` を初めて動かして見つかった訂正

6-6 の「2026-11-03 実機の訂正」に事実と根拠を書いた。変更点は次のとおり。`max_tokens` を送る（既定 20000）、`finish_reason = length` を警告する、トークン数の係数を設定値にして `doctor` が実測する、`doctor` の `timeout_sec` を最も遅い生成速度から出して `max_tokens` も推奨する、`extract` の進捗を出す、`cuda_dlls.py` が `os.environ["PATH"]` の先頭にも足す（`os.add_dll_directory()` は `LoadLibraryEx` に `LOAD_LIBRARY_SEARCH_USER_DIRS` を付けた呼び出しにしか効かず、CTranslate2 の読み込みには効かなかった）、`nvidia-cublas-cu12` を Windows の wheel がある 12.9.2.x に収める、テストが実データの `data/` に書かないようにする（`cfg` フィクスチャが `batch.log_dir` も一時ディレクトリにし、Path 型の設定がすべて `tmp_path` の下にあることを検査するテストを足した）。

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
| `transcription_unclear` | review | 文字起こしが読めず、要求の中心が決まらなかった抽出結果（`CLAUDE.md` 7章の案A）。チェック項目案は書かない（書けないことがこの投稿の内容である）。引用は直さずそのまま出す。これらは `reusable = false` なので 9-1 の照合の対象から外れ、`knowledge_items` にならない。投稿するのは、発話したのが利用者本人であり、壊れた引用を見れば自分が何と言ったか分かるためである。上限は `discord.max_transcription_unclear_per_run`（既定5） |

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

`processed_at` を埋める条件：`new_candidate` は status が1つに決まり、かつ返信の振り分けがすべて済んだとき。`contradiction` は返信が1件以上あり、すべて振り分けが済んだとき（リアクションでは何も変えない）。`speaker_sample` は反映後の `speaker_map` に `self` があり、解釈できない返信が無いとき。`bundle_result` は ❌ または ✅ が付いたとき。`transcription_unclear` も ❌ または ✅ が付いたとき（この種別は返信を読まない。対応する `knowledge_items` が無く、`review_log` は `item_id` が NOT NULL なので記録する先が無いためである。そのことは投稿の文面にも書いてある）。

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
| 6 | `kijun extract --conversation <ID>`（会議ごと） | その会議だけ飛ばす |
| 7 | `kijun match --conversation <ID>`（会議ごと） | その会議だけ飛ばす |
| 8 | `kijun discord post` | 続ける |
| 9 | 処理済み音声を `processed` へ移動（`transcribe` が既に行う） | — |

GPU を使う手順（1、3-4、6、7、8の一部）の直前には、ゲームの実行を確かめる（15-2）。終了コードは、失敗した手順が1つでもあれば 1。飛ばした手順（ゲームの延期の上限に達した、対象が無い、など）は失敗に数えない。

### 15-2. ゲームの実行中の延期（`batch/defer.py`）と、モデルのアンロード

2026-11-05 に Strata から Ollama に移ったため、バッチは LLM の起動も停止もしない（以前は `extract.strata.start_command` / `stop_command` と `batch.strata_ready_timeout_sec` で挟んでいた。`batch/strata.py` は削除した）。`keep_alive = "0"` なら Ollama が応答のたびにモデルを VRAM から下ろすので、手順3-4（文字起こし）の時点で LLM は載っていない。原則2-2は変えていない。

**keep_alive が "0" 以外のとき。** この前提が崩れる。手順1の後と手順6の後に、モデルを下ろす要求（`POST /api/generate {"model": ..., "keep_alive": 0}`）を送る（`ollama_api.unload_model`）。記録のステップ名は `1:unload`、`6:unload`。失敗しても処理は止めず、失敗として記録する。手順6の後は、抽出する会議があったときだけ送る。`keep_alive = "0"` のときは送らない。

**ゲームの実行中は延期する。** PC でゲーム（F1 25、League of Legends など）が動いていると、GPU を使う手順と VRAM を奪い合う。設定 `batch.defer_if_running` に実行ファイル名を挙げると、GPU を使う手順の直前に、そのどれかが動いていないかを確かめる。

```toml
[batch]
task_time = "23:30"            # kijun batch install-task が --time を省略されたときに使う
defer_if_running = []          # タスクマネージャーの「詳細」タブの実行ファイル名（大文字小文字は区別しない）
defer_check_interval_min = 30  # 延期中に再確認する間隔（分）
defer_max_hours = 4            # 延期する時間の上限（バッチ全体の合計）
```

- **GPU を使う手順**は、手順1（未処理の投稿があるとき。返信の振り分けに LLM を使う）、手順3-4（会議ごと）、手順6（会議ごと）、手順7（会議ごと）、手順8（重要度が未判定の候補があり review チャンネルが設定されているとき）。手順2（ingest）は GPU を使わないので、延期の影響を受けない。
- 動いていれば、`defer_check_interval_min` ごとに再確認して待つ。**待った時間はバッチ全体で合計**し、`defer_max_hours` を超えたら、残りの GPU を使う手順をすべて `skipped` にして終える（`note` に「ゲームの実行中のため延期の上限に達した」とプロセス名）。上限に達した後は、検出も待機もしない。1回の待機は残りの時間を超えない（上限 45 分・間隔 30 分なら 30 分と 15 分）。`defer_max_hours = 0` は待たずに飛ばす。
- **飛ばした会議は、翌晩に `batch/pending.py` が未処理として拾う。** 飛ばしただけで DB は変わらないため、このために新しい仕組みは要らない。`tests/test_batch.py` で、2晩目に文字起こし・抽出が実行されることを確認している。飛ばしたこと自体は失敗ではない（終了コードは 0）。
- 待ったときは `StepRecord`（ステップ名は `3-4:defer` のように `<手順>:defer`）に、待った時間と原因のプロセス名を残す。
- プロセスの検出は `ProcessLister`（Protocol）の背後に置き、Windows の実装は `tasklist /FO CSV /NH` の1列目を読む（`TasklistProcessLister`）。新しい依存（psutil）は足さない。`tasklist` の出力は UTF-8 で読めなければ cp932 で読む。テストでは `FakeProcessLister` と、`sleep` を差し替えて実時間を待たない。
- `defer_if_running` が空なら、検出そのものを行わない。
- **`tasklist` が失敗して一覧が得られないとき**は、警告を出して「動いていない」とみなして続ける（設計者の判断）。検出の失敗で、バッチが毎晩止まるよりよいと考えた。ゲームの実行を見逃す可能性は残る。
- 実行ファイル名は、ユーザーが自分の環境で確認して書く（推奨・未確定）。名前が違えば、ゲームが動いていても延期されない。

**手順1と8は LLM が使えなくても動く**（返信は原文だけ記録して未処理に残り、重要度は NULL で投稿される）。LLM が要るとき以外は、ゲームの確認を挟まない。

### 15-3. 処理が要る会議の選び方（`batch/pending.py`）

各段の直前に DB から調べ直す（子プロセスが DB を書き換えるため。子プロセスが動いている間は接続を開かない）。

- 文字起こし：`transcribed_at` が NULL の会議。
- 抽出：文字起こし済みで、`model_name` が設定のモデルの抽出結果が無い会議。**0件で済んだ会議を毎晩抽出し直さない**よう、`extract_logs` の記録で済みを見分ける。記録があっても全チャンクが失敗していたら再試行する。一部のチャンクだけ失敗した会議は自動では再試行しない（同じチャンクが毎晩失敗し続けるのを避ける。失敗は抽出の出力と記録に残る）。
- 照合：`reusable` な抽出結果のうち、`item_evidence` に紐付いていないものがある会議。照合は埋め込みモデルを読み込むので、対象が無ければ起動しない。

### 15-4. 実行の記録

各手順（ゲームによる待機 `<手順>:defer` と、モデルのアンロード `<手順>:unload` を含む）の開始・終了時刻、終了コード、標準出力と標準エラーをまとめた末尾 2,000 文字を、`batch.log_dir`（既定 `./data/batch_logs`）に `<日時>.json` として保存する。最後に全手順の成否を表で標準出力に出す。子プロセスの出力は UTF-8 に固定して取り込む（Windows の既定 cp932 での文字化けを避ける）。実機で未検証の部分が多いので、失敗の記録が最初の数週間の主な情報源になる。

### 15-5. タスクスケジューラへの登録（`batch/task_xml.py`）

`kijun batch install-task [--time HH:MM] [--out <パス>]` が XML（UTF-16）を作る。`--time` を省略すると設定の `batch.task_time`（既定 23:30）を使う。登録そのものは `schtasks /create /tn "kijun nightly batch" /xml <パス>` をユーザーが実行する（管理者権限が必要なことがあるので、コマンドを案内する形にした）。

- `WakeToRun = true`：`CLAUDE.md` 4-8 の「タスクを実行するためにスリープを解除する」。これが無いと PC がスリープしていると実行されない。
- `StartWhenAvailable = true`：PC が電源オフだった場合に、次に起動したときに実行する。
- `MultipleInstancesPolicy = IgnoreNew`：前夜のバッチがまだ動いているときに重ねて起動しない（DuckDB は同時に複数のプロセスから書き込めない）。
- `ExecutionTimeLimit = PT0S`（無制限）：ゲームによる延期の待ち時間（最大 `batch.defer_max_hours`）を含めると所要時間が読めないため。打ち切りよりも、記録で遅さを見て判断する。
- `LogonType = InteractiveToken`：ユーザーがログオンしている間（画面ロックを含む）に実行する。パスワードを保存しないで済むが、ログオフ中は実行されない。
- 作業ディレクトリは `install-task` を実行したディレクトリ（`config.toml` と `./data` の基準になる）。`--config` を付けて実行したときは、それを引き継ぐ。`uv` のフルパスが見つかればそれを書き、見つからなければ `uv` と書いて、書き換えを案内する。
