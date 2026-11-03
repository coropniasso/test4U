# kijun

会話の録音から、自分の判断基準を「用途別のチェックリスト」として文書化するための個人用ツール。
目的と制約は `CLAUDE.md`、実装の設計は `docs/implementation-plan.md` に書いてある。

この版で動くのは、次の作業と、それらが共有する基盤である。

| 作業表 | 内容 | サブコマンド |
|---|---|---|
| #4 | 文字起こし（faster-whisper）と話者分離（pyannote.audio） | `transcribe` |
| #5 | Strata（Qwen3.8-Flash-Next）による抽出と、設定を変えた2回の実行の比較 | `doctor`、`extract`、`compare` |
| #6 | DuckDB のスキーマ | `db init` |
| #7 | Discord でのレビュー（投稿と、リアクション・返信の読み取り）。`direction` の導出、重要度の判定、矛盾の検出を含む | `discord read`、`discord post`、`derive-direction` |
| #8 | 夜間バッチの統合とタスクスケジューラへの登録 | `batch run`、`batch install-task` |
| #9 | 音声の30日削除 | `purge-audio` |

#7 と #8 は自動テストで確認しているが、**実際の Discord への接続と、Windows のタスクスケジューラへの登録はまだ確認していない**（クラウド環境に Discord のトークンも Windows も無いため）。下の「ユーザーが実機で行う確認の順序」の手順17以降で確認する。

## 導入

Windows の実機では Python 3.12 を入れる（pyannote と torch の対応が確実な版）。パッケージ管理と実行は uv で行う。

```powershell
uv sync                          # 基盤だけ。重い依存（torch など）は入らない
uv run pytest                    # GPU とネットワークが要るテストは実行されない
uv run kijun --help
copy config.example.toml config.toml   # 必要な項目だけ書き換える。config.toml は git に入れない
```

重い依存は用途別の extra に分けてある。必要になったときに入れる。

| 用途 | コマンド | 入るもの |
|---|---|---|
| 文字起こしと話者分離 | `uv sync --extra transcribe` | faster-whisper、pyannote.audio、torch、torchaudio |
| 抽出（Strata への通信）と `doctor` | `uv sync --extra extract` | httpx |
| 埋め込みによる照合 | `uv sync --extra match` | sentence-transformers |
| Discord でのレビュー | `uv sync --extra discord` | discord.py |

複数の extra を同時に入れるときは `uv sync --extra transcribe --extra extract` のように並べる。
extra が足りない状態でサブコマンドを実行すると、どの extra を入れればよいかを示すエラーで終了する。

### torch を CUDA 版にする（Windows）

Windows で `uv sync --extra transcribe` をそのまま実行すると CPU 版の torch が入り、`torch.cuda.is_available()` が False になる。
CUDA 版を入れるには、`pyproject.toml` の末尾近くにある、コメントアウトした `[[tool.uv.index]]` と `[tool.uv.sources]` を有効にする。
有効にしていない理由は、uv が `uv sync` のたびに、そこに書いた index（download.pytorch.org）へ接続して全 extra の情報を取得するため、
その host に接続できない環境（外向き通信を制限したクラウド環境など）では、extra を使わなくても `uv sync` が失敗するからである。

1. https://pytorch.org/get-started/locally/ を開く。
2. Stable、Windows、Pip、Python、CUDA を選ぶ。表示される `pip install ... --index-url https://download.pytorch.org/whl/cuXXX` の `cuXXX` が、現在のタグである。
   2026-11-03 に確認した値は `cu132`（CUDA 13.2、torch 2.15.0、Windows）。CUDA 13.4 を選ぶと `cu134` になる。
   RTX 4070 Ti はどちらでも動くが、新しいほうはより新しいドライバを要求するので、迷ったら `cu132` を使う。画面に出たタグを優先する。
3. `pyproject.toml` のコメントアウトした部分の `#` を外す。
4. 4 か所の `cu132` が、2 で確認したタグと違っていれば書き換える（`[[tool.uv.index]]` の `name` と `url`、`[tool.uv.sources]` の `index` 名の 2 行）。
5. `uv sync --extra transcribe` を実行する。
6. `uv run python -c "import torch; print(torch.cuda.is_available())"` が `True` になることを確認する。

### pyannote の利用条件と Hugging Face のトークン

1. Hugging Face のアカウントでアクセストークンを作る（https://huggingface.co/settings/tokens）。
2. https://huggingface.co/pyannote/speaker-diarization-3.1 のページで利用条件に同意する。
3. https://huggingface.co/pyannote/segmentation-3.0 のページでも利用条件に同意する。
4. https://huggingface.co/pyannote/speaker-diarization-community-1 のページでも利用条件に同意する。これは pyannote.audio 4.x で必要になる。`speaker-diarization-3.1` のパイプラインが内部で、このリポジトリの `plda/xvec_transform.npz`（話者の声の特徴を比較するファイル）を読むためである。3.1 を使っていても必要になる。
5. PowerShell で `$env:HF_TOKEN = "hf_..."` を実行する。

3つのどれか1つでも同意していないと、モデルの取得が失敗する（`Cannot access gated repo` と出る）。

実機で確認した事実（pyannote.audio 4.0.7、2026-11-03）:

- 3つとも、同意した直後に許可された。手動の承認待ちはない。
- gated repo の同意フォームでは Company/university と Website が必須だが、内容は検証されない。個人なら `Individual` と、自分の GitHub のプロフィール URL を書けばよい。

抽出は Strata（ローカルの推論サーバー）に対して行うので、API キーは要らない。秘密情報は `HF_TOKEN` だけで、環境変数で渡し、設定ファイルには書かない。

### Discord bot の準備

Discord でのレビュー（`discord read` / `discord post`）には、Discord の bot が要る。**プロジェクトのサーバーではなく、自分だけの個人サーバーに作る**（メッセージに他のメンバーの発言が引用されるため。`CLAUDE.md` 2章）。bot が使うのは Discord の REST だけで、常駐しない。夜間バッチが動く間だけ接続する。

1. https://discord.com/developers/applications でアプリケーションを作り、Bot を追加してトークンを取得する。
2. Bot の設定で **Message Content Intent を有効にする。** 無効のままだと、返信の本文が空で返ってきて読めない（`discord read` が、有効にするよう示すエラーを出す）。
3. OAuth2 の URL Generator で scope に `bot` を選び、権限に「チャンネルを見る」「メッセージを送信」「メッセージ履歴を読む」を付けて、個人サーバーに招待する。
4. 個人サーバーに `#review`（新規候補・話者の対応付け・束ね結果）と `#contradictions`（矛盾の可能性がある組）を作る。Discord の設定でデベロッパーモードを有効にし、各チャンネルを右クリックして「チャンネルIDをコピー」する。
5. `config.toml` の `[discord]` に `review_channel_id` と `contradictions_channel_id` を書く（数字の文字列）。
6. `uv sync --extra discord` を実行する。
7. PowerShell で `$env:DISCORD_BOT_TOKEN = "..."` を実行する。トークンは環境変数で渡し、設定ファイルには書かない。タスクスケジューラから実行する場合は、ユーザー環境変数に登録する（`setx DISCORD_BOT_TOKEN "..."`）。

投稿へのリアクションと返信は、次のとおり使う。

| 操作 | 意味 |
|---|---|
| ✅ / ❌ / ⏸ | 承認 / 却下 / 保留。**1つだけ付ける。** 複数付けると判定できないので何も変えず、`discord read` が会議IDと項目IDを出す |
| 1️⃣ / 2️⃣ / 3️⃣ | 重要度（LLM の案を上書きする） |
| 投稿への返信（スレッド、または返信） | 理由・適用条件・適用しない条件・文言の修正を、自由な文章で書く。原文は必ず `review_log` に残る |
| 話者の対応付けへの返信 | `00=self 01=kouhai_a` と書く（確実）。自分を必ず `self` と書く。「00は自分、01は後輩A」のような文章でも読み取る |
| 矛盾の投稿への返信 | A か B のどちらについてかが分かるように書く（例：「Aは社外向けの告知のとき。Bは社内連絡のとき」） |
| 束ね結果への ❌ | 束ね方の誤り。自動では分割し直さない。`bundle.gap_minutes` を調整して `kijun ingest` をやり直す |

### Strata と Qwen3.8-Flash-Next

抽出の背面は、Strata で Qwen3.8-Flash-Next をローカル実行する（2026-11-03 決定。Ollama と Claude API は使わない）。

- Strata は MoE 向けのローカル推論エンジンで、動的オフロードで大きい MoE を一般向けの PC で動かす。API サーバーの既定ポートは 8080。
- Qwen3.8-Flash-Next は 125B の MoE（1トークンあたりの活性は約 6B）で、ネイティブの文脈長は 262,144 トークン。

**動くかどうかは確かめられていない。** Unsloth はこのモデルに「1bit量子化で最低 75GB の RAM」としており、この PC は 64GB で 11GB 足りない。また Strata の API 仕様（構造化出力に対応するか、OpenAI 互換のパスが何か）も、実装時に確認できていない。

そのため、設定には推測値を書かず、**`uv run kijun doctor` で実測してから `config.toml` に書く**手順にしてある。`doctor` は DuckDB も音声も GPU 用の依存も要らないので、Strata を起動すればすぐ実行できる。下の確認の順序では、torch を入れる前に `doctor` を実行する。

動かなかった場合は、量子化を下げる・オフロードの設定を変えるなどでローカル実行の維持を試す。クラウドへの退避は実装していない。

## ユーザーが実機で行う確認の順序

クラウド環境には GPU も音声ファイルも無いので、GPU・ネットワーク・外部プロセスに触る処理は、実機で次の順に確認する。

| # | 作業 | 成功の判定 |
|---|---|---|
| 1 | Python 3.12 と uv を入れる | `uv --version` が出る |
| 2 | `uv sync` | `uv run kijun --help` がサブコマンド一覧を出す |
| 3 | `uv run pytest` | すべて通る（GPU とネットワークが要るテストはスキップされる） |
| 4 | `uv run kijun db init` | `data/kijun.duckdb` ができ、テーブルが10個とビューが1つある |
| 5 | Strata を入れ、Qwen3.8-Flash-Next を取得して API サーバーを起動する | サーバーが起動する（既定ポート 8080）。**ここが最大の関門**。Unsloth の 75GB に対してこの PC は 64GB なので、起動しない・極端に遅い可能性がある |
| 6 | `uv sync --extra extract` | `httpx` が入る |
| 7 | `uv run kijun doctor` | 接続・スキーマ適合・速度・最大入力長の実測結果と、推奨値が出る。接続できなければ `config.toml` の `base_url` と `chat_path` を直して再実行する |
| 8 | `doctor` が出した推奨値を `config.toml` に書く | `enforces_json_schema` / `max_input_tokens` / `timeout_sec` / `chunk_minutes` / `overlap_minutes` の5つ |
| 9 | `doctor` の結果を `CLAUDE.md` 7章に書き込む | 未検証事項2件（Strata がスキーマ適合を保証するか、このモデルが 64GB で動くか）への回答になる |
| 10 | `pyproject.toml` の torch のインデックスのコメントを外し、タグを確認して `uv sync --extra transcribe` | `uv run python -c "import torch; print(torch.cuda.is_available())"` が True |
| 11 | Hugging Face で pyannote の3つのモデルページの利用条件に同意し、`HF_TOKEN` を設定 | 次の手順が通る |
| 12 | 5分程度の音声1本で `uv run kijun transcribe --audio <ファイル>` | 日本語の文字起こしと `SPEAKER_00` 等のラベルが出る。処理時間と VRAM 使用量を記録する |
| 13 | 音声を inbox に置いて `uv run kijun ingest` → `uv run kijun transcribe --conversation <ID>` | `utterances` に発話が入る |
| 14 | `uv run kijun extract --conversation <ID>` | 抽出が `extracted_requests` に入る。`data/extract_logs/` の JSON で、修復したチャンク数と失敗したチャンク数を確認する |
| 15 | `uv sync --extra match` して `uv run kijun match --conversation <ID> --dry-run` | 最類似の既存項目と類似度の一覧が出る（初回は既存項目が0件なので全件が新規候補） |
| 16 | `config.toml` の `model` を変えてもう一度 `extract` し、`uv run kijun compare --conversation <ID>` | 対照表の Markdown が出る |
| 17 | 上の「Discord bot の準備」を行う | `uv run pytest -m requires_network`（`KIJUN_TEST_DISCORD_CHANNEL` に試験用チャンネルのID）で、投稿・リアクションの取得・返信の取得が通る |
| 18 | `uv run kijun discord post` | 話者の対応付けと新規候補が Discord に出る |
| 19 | Discord で ✅ とスレッドへの返信を付け、話者の対応付けに `00=self 01=...` と返信してから `uv run kijun discord read` | `knowledge_items` の status、`review_log` の返信の原文、`speaker_map`、`direction` に反映される。返信の振り分けには Strata が要る |
| 20 | `uv run kijun batch run --dry-run`、続けて `uv run kijun batch run` | 手順と順序が出る。実行すると `data/batch_logs/` に記録が残り、最後に成否の表が出る |
| 21 | `uv run kijun batch install-task` が作った XML を、表示された `schtasks /create /xml` のコマンドで登録する | タスクスケジューラに登録され、翌晩にスリープ解除を含めて起動する。翌朝 `data/batch_logs/` を見る |

手順5から9を手順10より前に置いてある理由は2つある。(1) Strata が 64GB で動くかどうかが最も不確かな部分で、動かないなら他の作業の前提が変わる。(2) `doctor` は音声も DB も GPU 用の依存も要らないので、torch の CUDA 版（数GB）を入れる前に、抽出が成立するかどうかを知れる。

補足:

- 手順4の注意（スキーマバージョン1の DB が既にある場合）: 第1弾で作った DB には、`kijun db init` が `ALTER TABLE` で `discord_posts` の `channel_id` と `pair_item_id` を足す（バージョン2への移行）。既存の行は残る。何度実行しても安全。
- 手順4の注意（DB ファイルが既にある場合）: 埋め込みの `text_hash` 列とビュー `v_weekly_metrics` の修正より前に `kijun db init` を実行して作った DB ファイルがある場合、`embeddings` テーブルには `text_hash` 列が無い。`db init` は `CREATE TABLE IF NOT EXISTS` なので、既存のテーブルに列を足さない。
  - ビュー `v_weekly_metrics` は `CREATE OR REPLACE` なので、`kijun db init` をもう一度実行すれば新しい定義になる。
  - `embeddings` は、再計算できるキャッシュにすぎない。次の2コマンドで、他のテーブルの内容を残したまま作り直せる。
    `uv run python -c "import duckdb; duckdb.connect('data/kijun.duckdb').execute('DROP TABLE embeddings')"` を実行してから、`uv run kijun db init` を実行する（DB のパスは `[paths].db` に合わせる）。
  - DB ファイルを削除してしまった場合は、`uv run kijun db init` でスキーマを作り直してから、`uv run kijun restore-transcript --all` で `conversations` と `utterances` を文字起こしの JSON（`[paths].transcripts`）から復元する。文字起こしをやり直す必要は無い（GPU 時間を使わない）。
  - ただし、`extracted_requests`、`knowledge_items`、`item_evidence`、`speaker_map` などは JSON に入っていないので復元されない。復元後に `kijun extract` と `kijun match` をやり直すこと。レビュー済みの `knowledge_items` を失うので、DB ファイルの削除は最後の手段にする（先に上の `DROP TABLE embeddings` で直す）。
- 手順4の10テーブルは、CLAUDE.md 5章の8テーブルに、`schema_version` と `embeddings` の2テーブルを足したものである。ビューは `v_weekly_metrics`（抽出モデルごとの週次の指標）の1つ。
- 手順12の `--audio` は DuckDB に書かず、結果を標準出力と JSON に出すだけである。VRAM 使用量は、実行中に別の端末で `nvidia-smi` を実行して記録する。
- 手順12で記録した処理時間と VRAM 使用量は、CLAUDE.md 7章の未検証事項「pyannote.audio が Windows + CUDA 環境で問題なく動くか」への回答になる。手順7の `doctor` の結果は「Strata が JSON スキーマに従った出力を保証するか」と「Qwen3.8-Flash-Next が 12GB VRAM ＋ 64GB RAM で動くか、動く場合の tokens/s と最大入力長」への回答になる。どちらも結果を CLAUDE.md 7章に書き込む。
- 手順8で `chunk_minutes` を 180 にできるのは、`doctor` の手順5で2時間（約6万トークン）相当が1リクエストで通った場合だけである。既定の `chunk_minutes = 25` では、`doctor` が試す最大（4倍）でも100分相当にしかならず、2時間は確かめられない。確かめるには `chunk_minutes` を 30 以上にして `doctor` をもう一度実行する。`doctor` の出力にもそのことが出る。
- 手順14で修復したチャンクが多い場合、背面が壊れた JSON を返している。`doctor` の手順3が「200 でスキーマに適合」と出ていれば `enforces_json_schema = true` にすると減る可能性がある。
- `uv run pytest -m requires_gpu`、`uv run pytest -m requires_network` で、実機でだけ動く確認用のテストを実行できる（`KIJUN_TEST_AUDIO` に試験用の音声のパスを設定する）。

### VRAM の扱い

faster-whisper・pyannote と Qwen3.8-Flash-Next は、同時に VRAM に載せない。`transcribe` と `extract` は別のプロセスとして順に実行する。
プロセスが終われば OS が GPU メモリを解放する。

Strata に Ollama の `keep_alive: 0` に相当するアンロードの手段があるかは確認できていない。そのため、Strata をプロセスごと起動・停止する手段を設定（`extract.strata.start_command` / `stop_command`）で持たせてある。夜間バッチ（`kijun batch run`）がこの2つを読み、Strata を使う手順（Discord の返信の読み取り、抽出、Discord への投稿）の前後で実行する。`start_command` は Strata を起動したらすぐに戻るコマンド（バックグラウンドで起動するもの）を書く。戻ったあと、Strata が応答するまで `batch.strata_ready_timeout_sec` を上限に待つ。空にすると、バッチは Strata を起動も停止もしない（手で起動しておく運用）。夜間バッチを使わず手で実行するときは、必要なら手で Strata を止めてから `transcribe` を実行する。

## CLI のサブコマンド

```
uv run kijun db init                                   # スキーマを適用する
uv run kijun ingest [--dry-run]                        # inbox を走査し、会議単位に束ねて conversations に登録
uv run kijun transcribe --conversation <ID>            # 文字起こしと話者分離。utterances に書く
uv run kijun transcribe --audio <ファイル>              # 単発の試し実行（作業表 #4 の用途）
uv run kijun restore-transcript (--conversation <ID> | --all) [--force] [--dry-run]
                                                       # 文字起こしの JSON から conversations と utterances を復元
uv run kijun doctor [--out <パス>]                      # Strata の実測（接続・スキーマ適合・速度・最大入力長）
uv run kijun extract --conversation <ID> [--provider strata]
uv run kijun compare --conversation <ID> [--out <パス>] [--model <名前> --model <名前>]
                                                       # 同じ会議を2回抽出した結果の対照表（作業表 #5）
uv run kijun match --conversation <ID> [--dry-run]     # 既存項目との照合
uv run kijun purge-audio [--execute]                   # 音声の30日削除
uv run kijun discord read                              # Discord のリアクションと返信を読み取って DB に反映（作業表 #7）
uv run kijun discord post                              # 新規候補・矛盾・話者の対応付け・束ね結果を Discord に投稿（作業表 #7）
uv run kijun derive-direction (--conversation <ID> | --all)
                                                       # speaker_map から direction（自分→相手 / 相手→自分）を導出
uv run kijun batch run [--dry-run]                     # 夜間バッチ（作業表 #8）。--dry-run は順序とコマンドを表示するだけ
uv run kijun batch install-task [--time 23:30] [--out <パス>]
                                                       # タスクスケジューラ登録用の XML を生成（登録は schtasks をユーザーが実行）
```

全体に共通して `--config <パス>` で設定ファイルを指定できる（省略すると `./config.toml`、無ければ既定値）。

| サブコマンド | 動作 |
|---|---|
| `db init` | `data/kijun.duckdb` にスキーマを適用する。何度実行しても安全。 |
| `ingest` | `[paths].inbox` のファイルを `yyyyMMdd_HHmm_<形式>_<種類>.m4a` として解析し、同じ日で、前のファイルの終了から30分以内のものを同じ会議にまとめて `conversations` に登録する。`--dry-run` は登録せず、束ねた結果を表で出す。解析できないファイルは、削除も移動もせず、一覧に出す。 |
| `transcribe --conversation <ID>` | 登録済みの会議の音声を文字起こしして話者を付け、`utterances` に書く。同じ内容を `[paths].transcripts/<ID>.json` に保存し、音声を `[paths].processed` に移す。 |
| `transcribe --audio <ファイル>` | 1本の音声を試す。DuckDB には書かない。 |
| `restore-transcript` | `[paths].transcripts` の JSON から、`conversations` と `utterances` を復元する（DB を失ったときに、文字起こしをやり直さずに済む）。`--conversation <ID>` で1会議、`--all` で全部。`conversations` に同じ会議が既にあれば、上書きせずスキップする（`--force` で上書き）。`extracted_requests`、`knowledge_items`、`item_evidence` は復元されないので、`extract` と `match` をやり直す。`--dry-run` は書かずに、復元される内容を出す。 |
| `doctor` | Strata を実測し、`config.toml` に書くべき値を出す。(1) `/v1/models` に接続できるか (2) `response_format` なしの出力がそのまま読めるか・修復が要るか (3) `response_format` ありでスキーマに適合するか (4) tokens/s と所要秒数 (5) 1リクエストに入る最大の入力長。DuckDB も音声も要らないので最初に実行できる。結果は `data/doctor/strata_<日時>.json` にも保存する。 |
| `extract` | 発話を `chunk_minutes` ごと（`overlap_minutes` 重ねる）に分け、Strata で要求・指摘を抽出し、`extracted_requests` に書く。出力は必ず修復（前置きの散文、コードフェンス、トップレベルが配列、末尾が切れている）を通してから検証する。読めなければ同じチャンクを1回だけ投げ直し、2回失敗したらそのチャンクを失敗として記録して次に進む。 |
| `compare` | 同じ会議を**設定を変えて2回抽出した結果**を、数字の表と、左右に並べた対照表の Markdown にする。比べる対象は `model_name` が違う2組なので、2回目の抽出の前に `config.toml` の `extract.strata.model` をその設定を表す名前（例: `qwen3.8-flash-next-q8`）に変えておく。同じ名前のまま2回抽出すると、2回目が1回目を置き換える。 |
| `match` | `reusable = true` の抽出結果を、既存のチェック項目とコサイン類似度で照合する。閾値（既定 0.85）以上なら既存項目に紐付け、未満なら `status = 'candidate'` の新規候補を作る。`--dry-run` は書かず、類似度の降順の表を出す。`--provider` で、どのモデルの抽出結果を対象にするかを選ぶ（省略すると設定の `extract.provider`）。 |
| `discord read` | 前回までに投稿したメッセージのリアクションと返信を読む。✅❌⏸が1つだけ付いていれば status を更新し、**2つ以上付いていたら何も変えずに会議IDと項目IDを出す**（次回また対象になる）。1️⃣2️⃣3️⃣は重要度の上書き。返信は原文を `review_log` に残してから LLM で `reason` / `applies_when` / `not_applies_when` / `check_text` に振り分ける。話者の対応付けの返信は `speaker_map` に反映し、`self` が決まった会議の `direction` を導出する。`[discord]` のチャンネルIDが空なら何もせず、設定が必要だと出して終わる。 |
| `discord post` | 新規候補（重要度の降順、同じなら新しい順に、`max_candidates_per_run` 件まで。残りの件数を出す）、矛盾の可能性がある組、話者の対応付けの依頼、複数ファイルを束ねた会議の一覧を投稿する。重要度が未判定の候補は投稿の前に LLM で判定する。`DISCORD_BOT_TOKEN` が要る。 |
| `derive-direction` | `speaker_map` から `extracted_requests.direction` を導出する。`self` が決まっていない会議は導出しない。`speaker_map` を手で直した後や、`extract` をやり直した後に使う。 |
| `batch run` | 夜間バッチ。`discord read` → `ingest` → `transcribe`（会議ごと）→ `extract`（会議ごと。前後で `start_command` / `stop_command`）→ `match`（会議ごと）→ `discord post` を、別プロセスとして順に起動する。`ingest` が失敗したら止め、それ以外は1つの会議の失敗で止めない。各手順の記録を `batch.log_dir` に JSON で保存し、最後に成否の表を出す。失敗があれば終了コード 1。 |
| `batch install-task` | Windows のタスクスケジューラ用の XML（毎日 `--time` に実行。`WakeToRun` と `StartWhenAvailable` が true）を作り、登録用の `schtasks /create /xml` のコマンドを案内する。登録そのものはしない。 |
| `purge-audio` | 文字起こし完了から `retention.audio_days`（既定 30）日以上たった音声の一覧を出す。`--execute` を付けたときだけ削除する。文字起こしの JSON は削除しない。 |

## 既知の制約

- 承認件数そのものは `compare` の出力には出ない。承認の記録は `review_log`（`status_changed` と `check_text_edited`）にあり、「LLM の提案文言を修正せず承認した割合」（`CLAUDE.md` 6章）は、そこから数える。
- 判定できないリアクション（✅と❌が両方など）は、推測で決めずに未処理のまま残る。標準出力（夜間バッチでは `batch_logs`）に出る会議IDと項目IDを見て、リアクションを1つに直す。
- 承認のリアクションを付けて処理された後の返信は取り込まれない。リアクションと返信は、同じ時間帯に付ける。
- 話者の対応付けの返信の原文は残らない（`review_log` は項目に属する出来事だけを入れる）。結果の `speaker_map` だけが残る。
- `extract` をやり直した会議の `direction` は NULL に戻る。`derive-direction` で導出し直す。
- `artifact_type` と `counterpart_role` の値は、LLM が自由な文字列を入れる。3回分の抽出結果を見て、よく出る値を決める。
- 矛盾の検出規則と重要度の3段階の基準は、設計者（Claude）が考案したもので、`CLAUDE.md` に根拠が無い。運用して見直す。
- 失敗したチャンクの記録と、出力を修復した記録は、DuckDB のスキーマに列が無いので、DB と同じディレクトリの `extract_logs/` に JSON で保存する。`compare` と夜間バッチ（0件で済んだ抽出を毎晩やり直さないため）がここを読む。
- 夜間バッチは、Strata を最大3回起動する（返信の読み取り、抽出、投稿）。起動が重いときは `start_command` を空にして、Strata を手で起動しておく。

## テスト

```powershell
uv run pytest                        # GPU とネットワークが要るテストを除いて実行
uv run pytest -m requires_gpu        # 実機で、GPU を使う確認だけ実行
uv run pytest -m requires_network    # 実機で、Strata・Hugging Face を使う確認だけ実行
```

GPU・ネットワーク・外部プロセスに触る処理は、すべて `typing.Protocol` で定義したインターフェース
（`Transcriber`、`Diarizer`、`Extractor`、`Embedder`、`DiscordClient`、`CommandRunner` と、LLM の3用途の `ImportanceJudge`、`ReplyRouter`、`SpeakerReplyInterpreter`）の背後に置いてある。テストでは `tests/fakes.py` の偽の実装を使う。
