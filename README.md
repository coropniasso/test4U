# kijun

会話の録音から、自分の判断基準を「用途別のチェックリスト」として文書化するための個人用ツール。
目的と制約は `CLAUDE.md`、実装の設計は `docs/implementation-plan.md` に書いてある。

この版で動くのは、次の作業と、それらが共有する基盤である。

| 作業表 | 内容 | サブコマンド |
|---|---|---|
| #4 | 文字起こし（faster-whisper）と話者分離（pyannote.audio） | `transcribe` |
| #5 | Ollama による抽出と、設定を変えた2回の実行の比較 | `doctor`、`extract`、`compare` |
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
| 抽出（Ollama への通信）と `doctor` | `uv sync --extra extract` | httpx |
| 埋め込みによる照合 | `uv sync --extra match` | sentence-transformers |
| Discord でのレビュー | `uv sync --extra discord` | discord.py |

複数の extra を同時に入れるときは `uv sync --extra transcribe --extra extract` のように並べる。
extra が足りない状態でサブコマンドを実行すると、どの extra を入れればよいかを示すエラーで終了する。

### torch を CUDA 版にする（Windows）

Windows で `uv sync --extra transcribe` をそのまま実行すると CPU 版の torch が入り、`torch.cuda.is_available()` が False になる。
`pyproject.toml` の `[[tool.uv.index]]` と `[tool.uv.sources]` が、Windows の torch を CUDA 版（`cu132`）にする設定で、2026-11-05 から既定で有効になっている。
ただし uv は `uv sync` のたびに、そこに書いた index（download.pytorch.org）へ接続して全 extra の情報を取得するため、
その host に接続できない環境（外向き通信を制限したクラウド環境など）では、extra を使わなくても `uv sync` が失敗する。その環境では、`pyproject.toml` の該当する 8 行をコメントアウトする。
同期は `uv sync --all-extras` で行う（extra を一部だけ指定すると、他の extra のパッケージが消える）。

1. https://pytorch.org/get-started/locally/ を開く。
2. Stable、Windows、Pip、Python、CUDA を選ぶ。表示される `pip install ... --index-url https://download.pytorch.org/whl/cuXXX` の `cuXXX` が、現在のタグである。
   2026-11-03 に確認した値は `cu132`（CUDA 13.2、torch 2.15.0、Windows）。CUDA 13.4 を選ぶと `cu134` になる。
   RTX 4070 Ti はどちらでも動くが、新しいほうはより新しいドライバを要求するので、迷ったら `cu132` を使う。画面に出たタグを優先する。
3. `pyproject.toml` の 4 か所の `cu132` が、2 で確認したタグと違っていれば書き換える（`[[tool.uv.index]]` の `name` と `url`、`[tool.uv.sources]` の `index` 名の 2 行）。
4. `uv sync --all-extras` を実行する。
5. `uv run python -c "import torch; print(torch.cuda.is_available())"` が `True` になることを確認する。

### CUDA 12 の DLL を入れる（Windows）

faster-whisper の CTranslate2 は `cublas64_12.dll` と cuDNN（CUDA 12）を要求する。torch の CUDA 13 の DLL とは名前が違うので、両方を入れる。

```powershell
uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

`transcribe` extra にも Windows 限定の依存として宣言してあるので、`uv sync --extra transcribe` でも入る。
入った DLL のディレクトリは、コードが `os.add_dll_directory()` で登録する。**PowerShell で `$env:PATH` を手動で設定する必要はない。**
起動時に標準エラーへ「NVIDIA の DLL ディレクトリを登録: ...」と出る。「0 件」と出たら、上のコマンドが実行されていない。

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

抽出は Ollama（ローカルの推論サーバー）に対して行うので、API キーは要らない。秘密情報は `HF_TOKEN` だけで、環境変数で渡し、設定ファイルには書かない。

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

### Ollama の準備

抽出の背面は **Ollama** で、GPU（VRAM 12GB）に収まる 9B〜14B のモデルをローカル実行する（2026-11-05 決定。それ以前の Strata は、プロセスが RAM を約56GB 保持して PC の他の作業を圧迫したため、削除した）。返信の振り分け・重要度の判定・話者の解釈も、同じ Ollama を使う。音声も文字起こしも PC の外に出さない。

**インストールとモデルの取得は、ユーザーが行う**（実装を担当するセッションは行わない）。

1. Ollama for Windows をインストールする（https://ollama.com から）。インストール後、タスクトレイに常駐する。
2. モデルを取得する。

   ```powershell
   ollama pull qwen3.5:9b
   ollama pull qwen3:14b
   ollama list        # 取得できていることを確認する
   ```

   モデル名は設計者（Claude）の初期値で、**存在するかは確かめていない**。無ければ Ollama のライブラリ（https://ollama.com/library）で近い名前を探して決める。名前を変えたら `config.toml` の `[extract.ollama] model` も変える。
3. `uv sync --extra extract` を実行する（httpx が入る）。
4. `config.toml` に `[extract.ollama]` を書く（`config.example.toml` に全項目とコメントがある）。`base_url` は `http://127.0.0.1:11434` と**数字で書く**（`localhost` は Windows で IPv6 の `::1` に解決されることがあり、接続拒否になる）。
5. `uv run kijun doctor` を実行する。**検査3（モデルが GPU に収まっているか）が「良」であること。** 一部がメインメモリに溢れていると出たら、`num_ctx` を下げるか、小さいモデルにする。
6. 品質を比べる。`uv run python scripts/measure_extraction.py --conversation 20261103_1930_f2f_teirei --variants scripts/variants.example.toml` が、モデルと `think` を変えた4通りの対照表を出す。結果には Strata の基準（同じ会議・同じプロンプト）も並ぶ。**採否はユーザーが決める。**

古い設定が残っていると、設定の読み込みが失敗する。`config.toml` から次を削除する：`[extract.strata]` の節、`[extract] provider = "strata"`、`[batch] strata_ready_timeout_sec`。

**Strata が常駐していれば終了する。** Strata のデスクトップアプリが動いていると、エンジンが約56GBを保持したままになる。トレイのメニューから「終了」する（ウィンドウを閉じるだけでは終わらない）。必要ならアンインストールする（ユーザーの判断）。

### ゲームとの競合

夜間バッチの開始時刻は設定 `batch.task_time`（既定 23:30）で変えられる（`kijun batch install-task` が `--time` を省略されたときに使う）。加えて、`batch.defer_if_running` に実行ファイル名を挙げると、そのプロセスが動いている間は GPU を使う手順を始めずに待つ。実行ファイル名は、ゲームを起動した状態で、**タスクマネージャーの「詳細」タブ**に出る名前を書く（F1 25、League of Legends など。大文字小文字は区別しない）。待つのは `defer_check_interval_min`（既定30分）ごとの再確認で、バッチ全体で合計 `defer_max_hours`（既定4時間）を超えたら、その夜の残りの GPU を使う手順を飛ばす。飛ばした会議は翌晩に拾われる。

## ユーザーが実機で行う確認の順序

クラウド環境には GPU も音声ファイルも無いので、GPU・ネットワーク・外部プロセスに触る処理は、実機で次の順に確認する。

| # | 作業 | 成功の判定 |
|---|---|---|
| 1 | Python 3.12 と uv を入れる | `uv --version` が出る |
| 2 | `uv sync` | `uv run kijun --help` がサブコマンド一覧を出す |
| 3 | `uv run pytest` | すべて通る（GPU とネットワークが要るテストはスキップされる） |
| 4 | `uv run kijun db init` | `data/kijun.duckdb` ができ、テーブルが10個とビューが1つある |
| 5 | 上の「Ollama の準備」の 1〜2（Ollama のインストールと `ollama pull`） | `ollama list` にモデルが出る |
| 6 | `uv sync --extra extract` | `httpx` が入る |
| 7 | `uv run kijun doctor` | 検査1〜5の結果と、推奨値が出る。**検査3が「良」であること。** 接続できなければ `config.toml` の `base_url` を直して再実行する |
| 8 | `doctor` が出した推奨値を `config.toml` に書く | `enforces_json_schema` / `timeout_sec` / `tokens_per_character`（検査5が失敗したときは `chunk_minutes` も） |
| 9 | `doctor` の結果を `CLAUDE.md` 7章に書き込む | 「Ollama のモデルが GPU に収まるか、速度、品質」への回答になる |
| 10 | `pyproject.toml` の torch のインデックスのタグ（既定で有効。`cu132`）を確認して `uv sync --all-extras` | `uv run python -c "import torch; print(torch.cuda.is_available())"` が True |
| 11 | Hugging Face で pyannote の3つのモデルページの利用条件に同意し、`HF_TOKEN` を設定 | 次の手順が通る |
| 12 | 5分程度の音声1本で `uv run kijun transcribe --audio <ファイル>` | 日本語の文字起こしと `SPEAKER_00` 等のラベルが出る。処理時間と VRAM 使用量を記録する |
| 13 | 音声を inbox に置いて `uv run kijun ingest` → `uv run kijun transcribe --conversation <ID>` | `utterances` に発話が入る |
| 14 | `uv run kijun extract --conversation <ID>` | 抽出が `extracted_requests` に入る。`data/extract_logs/` の JSON で、修復したチャンク数と失敗したチャンク数を確認する |
| 15 | `uv sync --extra match` して `uv run kijun match --conversation <ID> --dry-run` | 最類似の既存項目と類似度の一覧が出る（初回は既存項目が0件なので全件が新規候補） |
| 16 | `config.toml` の `model` を変えてもう一度 `extract` し、`uv run kijun compare --conversation <ID>` | 対照表の Markdown が出る |
| 17 | 上の「Discord bot の準備」を行う | `uv run pytest -m requires_network`（`KIJUN_TEST_DISCORD_CHANNEL` に試験用チャンネルのID）で、投稿・リアクションの取得・返信の取得が通る |
| 18 | `uv run kijun discord post` | 話者の対応付けと新規候補が Discord に出る |
| 19 | Discord で ✅ とスレッドへの返信を付け、話者の対応付けに `00=self 01=...` と返信してから `uv run kijun discord read` | `knowledge_items` の status、`review_log` の返信の原文、`speaker_map`、`direction` に反映される。返信の振り分けには Ollama が要る |
| 20 | `uv run kijun batch run --dry-run`、続けて `uv run kijun batch run` | 手順と順序が出る。実行すると `data/batch_logs/` に記録が残り、最後に成否の表が出る |
| 21 | `uv run kijun batch install-task` が作った XML を、表示された `schtasks /create /xml` のコマンドで登録する | タスクスケジューラに登録され、翌晩にスリープ解除を含めて起動する。翌朝 `data/batch_logs/` を見る |
| 22 | ゲームを起動した状態で `uv run kijun batch run` を試す（`batch.defer_if_running` に実行ファイル名を書いてから） | `data/batch_logs/` に `<手順>:defer` の記録が残り、ゲームを終了すると再開する（短くするには `defer_check_interval_min` を小さくする） |
| 23 | `hotwords` の効果の確認。`20261103_1930_f2f_teirei` の音声（`conversations.source_files` で場所を調べる。30日の削除期限内なら残っている）を、`transcribe.hotwords = ["箇条書き", "感嘆符"]` を書いた設定と書かない設定で `uv run kijun --config <設定> transcribe --audio <音声> --out <出力先>.json` | 「箇条書き」「感嘆符」が正しく取れ、他の箇所が悪化せず、その語を言っていない箇所に当てはめていない（DB の `utterances` は変わらない） |

手順5から9を手順10より前に置いてある理由は2つある。(1) モデルが GPU（VRAM 12GB）に収まるかどうかが最も不確かな部分で、収まらないなら他の作業の前提が変わる。(2) `doctor` は音声も DB も GPU 用の依存も要らないので、torch の CUDA 版（数GB）を入れる前に、抽出が成立するかどうかを知れる。

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
- 手順12で記録した処理時間と VRAM 使用量は、CLAUDE.md 7章の未検証事項「pyannote.audio が Windows + CUDA 環境で問題なく動くか」への回答になる。手順7の `doctor` の結果は「Ollama のモデルが GPU に収まるか、速度、品質」と「`format` が出力をスキーマに適合させるか」への回答になる。どちらも結果を CLAUDE.md 7章に書き込む。
- 既定の `chunk_minutes = 12` / `overlap_minutes = 1` と `num_ctx = 16384` は設計者の初期値で、実測に基づかない。入力が `num_ctx` に収まらない設定は、読み込みで拒否される（Ollama は入力が文脈長を超えると、エラーを出さずに先頭を捨てるため）。
- 手順14で修復したチャンクが多い場合、背面が壊れた JSON を返している。`doctor` の検査2が「適合」と出ていれば `enforces_json_schema = true` にすると減る可能性がある。
- `uv run pytest -m requires_gpu`、`uv run pytest -m requires_network` で、実機でだけ動く確認用のテストを実行できる（`KIJUN_TEST_AUDIO` に試験用の音声のパスを設定する）。`requires_gpu` には、Ollama のモデルが GPU に収まっているかの確認も含む。

### VRAM の扱い

faster-whisper・pyannote と LLM（Ollama）は、同時に VRAM に載せない。`transcribe` と `extract` は別のプロセスとして順に実行する。
プロセスが終われば OS が GPU メモリを解放する。

Ollama は、設定 `extract.ollama.keep_alive` が `"0"`（既定）なら、応答のたびにモデルを VRAM から下ろす。したがって文字起こしの時点で LLM は載っていない。夜間バッチは LLM の起動も停止もしない（Strata のときは `start_command` / `stop_command` で挟んでいたが、削除した）。`keep_alive` を `"0"` 以外にしたときは、夜間バッチが手順1と手順6の後にモデルを下ろす要求を送る。夜間バッチを使わず手で実行するときは、`keep_alive` を `"0"` のままにしておくか、`ollama stop <モデル名>` で手で下ろしてから `transcribe` を実行する。

## CLI のサブコマンド

```
uv run kijun db init                                   # スキーマを適用する
uv run kijun ingest [--dry-run]                        # inbox を走査し、会議単位に束ねて conversations に登録
uv run kijun transcribe --conversation <ID>            # 文字起こしと話者分離。utterances に書く
uv run kijun transcribe --audio <ファイル>              # 単発の試し実行（作業表 #4 の用途）
uv run kijun restore-transcript (--conversation <ID> | --all) [--force] [--dry-run]
                                                       # 文字起こしの JSON から conversations と utterances を復元
uv run kijun doctor [--out <パス>]                      # Ollama の実測（接続・スキーマ適合・GPU に収まるか・速度・入力の切り捨て）
uv run kijun extract --conversation <ID> [--provider ollama]
uv run kijun compare --conversation <ID> [--out <パス>] [--model <名前> --model <名前>]
                                                       # 同じ会議を2回抽出した結果の対照表（作業表 #5）
uv run kijun match --conversation <ID> [--dry-run]     # 既存項目との照合
uv run kijun purge-audio [--execute]                   # 音声の30日削除
uv run kijun discord read                              # Discord のリアクションと返信を読み取って DB に反映（作業表 #7）
uv run kijun discord post                              # 新規候補・矛盾・話者の対応付け・束ね結果を Discord に投稿（作業表 #7）
uv run kijun derive-direction (--conversation <ID> | --all)
                                                       # speaker_map から direction（自分→相手 / 相手→自分）を導出
uv run kijun batch run [--dry-run]                     # 夜間バッチ（作業表 #8）。--dry-run は順序とコマンドを表示するだけ
uv run kijun batch install-task [--time HH:MM] [--out <パス>]
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
| `doctor` | Ollama を実測し、`config.toml` に書くべき値を出す。(1) 接続できるか・設定のモデルがあるか (2) `format` にスキーマを付けて、出力がそのまま読めるか (3) **モデルが GPU に収まっているか**（`/api/ps` の `size_vram` と `size`） (4) tokens/s と所要秒数、2時間の会議の見込み (5) 入力が `num_ctx` を超えて切り捨てられていないか。最後にモデルを下ろす。DuckDB も音声も要らないので最初に実行できる。結果は `data/doctor/ollama_<日時>.json` にも保存する。 |
| `extract` | 発話を `chunk_minutes` ごと（`overlap_minutes` 重ねる）に分け、Ollama で要求・指摘を抽出し、`extracted_requests` に書く。入力が切り捨てられた疑い（`prompt_eval_count` が `num_ctx - num_predict` に達した）があれば、そのチャンクを失敗にする。出力は必ず修復（前置きの散文、コードフェンス、トップレベルが配列、末尾が切れている）を通してから検証する。読めなければ同じチャンクを1回だけ投げ直し、2回失敗したらそのチャンクを失敗として記録して次に進む。 |
| `compare` | 同じ会議を**設定を変えて2回抽出した結果**を、数字の表と、左右に並べた対照表の Markdown にする。比べる対象は `model_name` が違う2組なので、2回目の抽出の前に `config.toml` の `extract.run_label` に設定を表す名前（例: `nothink`）を入れておく（`model_name` が `qwen3.5:9b+nothink` になる）。同じ名前のまま2回抽出すると、2回目が1回目を置き換える。設定ごとの測定を1コマンドで行うには `scripts/measure_extraction.py` を使う。 |
| `match` | `reusable = true` の抽出結果を、既存のチェック項目とコサイン類似度で照合する。閾値（既定 0.85）以上なら既存項目に紐付け、未満なら `status = 'candidate'` の新規候補を作る。`--dry-run` は書かず、類似度の降順の表を出す。`--provider` で、どのモデルの抽出結果を対象にするかを選ぶ（省略すると設定の `extract.provider`）。 |
| `discord read` | 前回までに投稿したメッセージのリアクションと返信を読む。✅❌⏸が1つだけ付いていれば status を更新し、**2つ以上付いていたら何も変えずに会議IDと項目IDを出す**（次回また対象になる）。1️⃣2️⃣3️⃣は重要度の上書き。返信は原文を `review_log` に残してから LLM で `reason` / `applies_when` / `not_applies_when` / `check_text` に振り分ける。話者の対応付けの返信は `speaker_map` に反映し、`self` が決まった会議の `direction` を導出する。`[discord]` のチャンネルIDが空なら何もせず、設定が必要だと出して終わる。 |
| `discord post` | 新規候補（重要度の降順、同じなら新しい順に、`max_candidates_per_run` 件まで。残りの件数を出す）、矛盾の可能性がある組、話者の対応付けの依頼、複数ファイルを束ねた会議の一覧を投稿する。重要度が未判定の候補は投稿の前に LLM で判定する。`DISCORD_BOT_TOKEN` が要る。 |
| `derive-direction` | `speaker_map` から `extracted_requests.direction` を導出する。`self` が決まっていない会議は導出しない。`speaker_map` を手で直した後や、`extract` をやり直した後に使う。 |
| `batch run` | 夜間バッチ。`discord read` → `ingest` → `transcribe`（会議ごと）→ `extract`（会議ごと）→ `match`（会議ごと）→ `discord post` を、別プロセスとして順に起動する。GPU を使う手順の直前には、`batch.defer_if_running` のゲームが動いていないかを確かめ、動いていれば待つ（上限を超えたら残りの GPU を使う手順を飛ばす）。`ingest` が失敗したら止め、それ以外は1つの会議の失敗で止めない。各手順の記録を `batch.log_dir` に JSON で保存し、最後に成否の表を出す。失敗があれば終了コード 1。 |
| `batch install-task` | Windows のタスクスケジューラ用の XML（毎日 `--time`（省略すると `batch.task_time`）に実行。`WakeToRun` と `StartWhenAvailable` が true）を作り、登録用の `schtasks /create /xml` のコマンドを案内する。登録そのものはしない。 |
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
- `batch.defer_if_running` に書く実行ファイル名が実際のプロセス名と違うと、ゲームが動いていても延期されない。ユーザーが「詳細」タブで確認して書く。
- `hotwords` の効果は未検証で、既定は空である。

## テスト

```powershell
uv run pytest                        # GPU とネットワークが要るテストを除いて実行
uv run pytest -m requires_gpu        # 実機で、GPU を使う確認だけ実行
uv run pytest -m requires_network    # 実機で、Ollama・Hugging Face を使う確認だけ実行
```

GPU・ネットワーク・外部プロセスに触る処理は、すべて `typing.Protocol` で定義したインターフェース
（`Transcriber`、`Diarizer`、`Extractor`、`Embedder`、`DiscordClient`、`CommandRunner` と、LLM の3用途の `ImportanceJudge`、`ReplyRouter`、`SpeakerReplyInterpreter`）の背後に置いてある。テストでは `tests/fakes.py` の偽の実装を使う。
