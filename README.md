# kijun

会話の録音を文字起こしし、議事録にまとめて Markdown で保存し、Discord に投稿する個人用ツール。
目的と制約は `CLAUDE.md` に書いてある。2026-11-06 に、目的をチェックリストの作成から議事録の作成に変えた。以前の文書は `docs/history/` にある。

処理の流れは、録音（iPhone）→ 文字起こしと話者分離（faster-whisper、pyannote.audio）→ 議事録の生成（Ollama）→ Markdown の保存と Discord への投稿である。毎晩のバッチで自動実行し、手で実行するコマンドもある。

| 作業 | サブコマンド |
|---|---|
| 文字起こしと話者分離 | `transcribe` |
| 議事録の生成 | `minutes`、`doctor` |
| DuckDB のスキーマ | `db init` |
| Discord への投稿 | `discord post` |
| 夜間バッチの統合とタスクスケジューラへの登録 | `batch run`、`batch install-task` |
| 音声の30日削除 | `purge-audio` |

**実際の Discord への接続と、Windows のタスクスケジューラへの登録は、まだ確認していない。**

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
| 議事録の生成（Ollama への通信）と `doctor` | `uv sync --extra extract` | httpx |
| Discord への投稿 | `uv sync --extra discord` | discord.py |

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

議事録の生成は Ollama（ローカルの推論サーバー）に対して行うので、API キーは要らない。秘密情報は `HF_TOKEN` と `DISCORD_BOT_TOKEN` だけで、環境変数で渡し、設定ファイルには書かない。

### Discord bot の準備

議事録の Discord への投稿（`discord post`）には、Discord の bot が要る。**プロジェクトのサーバーではなく、自分だけの個人サーバーに作る**（`CLAUDE.md` 2章）。bot が使うのは Discord の REST だけで、常駐しない。夜間バッチが動く間だけ接続し、本文とファイル1つを投稿する。リアクションや返信は読まないので、Message Content Intent は要らない。

1. https://discord.com/developers/applications でアプリケーションを作り、Bot を追加してトークンを取得する。
2. OAuth2 の URL Generator で scope に `bot` を選び、権限に「チャンネルを見る」「メッセージを送信」「ファイルを添付」を付けて、個人サーバーに招待する。
3. 個人サーバーに議事録用のチャンネルを1つ作る。Discord の設定でデベロッパーモードを有効にし、チャンネルを右クリックして「チャンネルIDをコピー」する。
4. `config.toml` の `[discord]` に `minutes_channel_id` を書く（数字の文字列。空のままだと投稿されない）。
5. `uv sync --extra discord` を実行する。
6. PowerShell で `$env:DISCORD_BOT_TOKEN = "..."` を実行する。トークンは環境変数で渡し、設定ファイルには書かない。タスクスケジューラから実行する場合は、ユーザー環境変数に登録する（`setx DISCORD_BOT_TOKEN "..."`）。

**実際の Discord への接続は、まだ確認していない**（トークンが未設定のため）。

### Ollama の準備

議事録の生成の背面は **Ollama** で、GPU（VRAM 12GB）に収まる 9B〜14B のモデルをローカル実行する。音声も文字起こしも PC の外に出さない。

**インストールとモデルの取得は、ユーザーが行う**。

1. Ollama for Windows をインストールする（https://ollama.com から）。インストール後、タスクトレイに常駐する。
2. モデルを取得する（`ollama pull qwen3.5:9b`、`ollama list` で確認する）。
3. `uv sync --extra extract` を実行する（httpx が入る）。
4. `config.toml` に `[llm]` を書く（`config.example.toml` に全項目とコメントがある）。`base_url` は `http://127.0.0.1:11434` と**数字で書く**（`localhost` は Windows で IPv6 の `::1` に解決されることがあり、接続拒否になる）。`qwen3.5:9b` では `think = false` を書く。
5. `uv run kijun doctor` を実行する。**検査3（モデルが GPU に収まっているか）が「良」であること。** 一部がメインメモリに溢れていると出たら、`num_ctx` を下げるか、小さいモデルにする。

### 古い config.toml からの移行

2026-11-06 に目的をチェックリストの作成から議事録の作成に変えたため、`config.toml` の次の項目は読み込みで拒否される（未知のキーを拒否する仕組みのため）。

| 古い項目 | 新しい項目 |
|---|---|
| `[extract]`（`chunk_minutes`、`overlap_minutes`、`run_label` ほか） | `[minutes]` の `chunk_minutes`（重なりは 0 で固定） |
| `[extract.ollama]` | `[llm]`（同じ項目名） |
| `[match]` | 削除 |
| `[discord]` の `review_channel_id`、`contradictions_channel_id`、`max_*`、`contradiction_min_similarity`、`speaker_sample_count` | `[discord]` の `minutes_channel_id` |
| `[paths]` の `review_logs` | 削除（`minutes` を追加） |

書き換え後の内容の例が `data/config.toml.proposed` にある（このリポジトリの作業者が作った提案で、`config.toml` 自体は書き換えていない）。

### ゲームとの競合

夜間バッチの開始時刻は設定 `batch.task_time`（既定 23:30）で変えられる（`kijun batch install-task` が `--time` を省略されたときに使う）。加えて、`batch.defer_if_running` に実行ファイル名を挙げると、そのプロセスが動いている間は GPU を使う手順（文字起こし、議事録）を始めずに待つ。実行ファイル名は、ゲームを起動した状態で、**タスクマネージャーの「詳細」タブ**に出る名前を書く（大文字小文字は区別しない）。待つのは `defer_check_interval_min`（既定30分）ごとの再確認で、バッチ全体で合計 `defer_max_hours`（既定4時間）を超えたら、その夜の残りの GPU を使う手順を飛ばす。飛ばした会議は翌晩に拾われる。

## ユーザーが実機で行う確認の順序

| # | 作業 | 成功の判定 |
|---|---|---|
| 1 | Python 3.12 と uv を入れる | `uv --version` が出る |
| 2 | `uv sync --all-extras` | `uv run kijun --help` がサブコマンド一覧を出す |
| 3 | `uv run pytest` | すべて通る（GPU とネットワークが要るテストはスキップされる） |
| 4 | `uv run kijun db init` | `data/kijun.duckdb` ができる（新規は4テーブル。バージョン4の DB は、古いテーブルを残したまま `minutes` が足される） |
| 5 | Ollama のインストールと `ollama pull qwen3.5:9b` | `ollama list` にモデルが出る |
| 6 | `uv run kijun doctor` | 検査1〜5の結果と推奨値が出る。**検査3が「良」であること** |
| 7 | torch の CUDA 版の確認（上の「torch を CUDA 版にする」）と、Hugging Face のトークン | `torch.cuda.is_available()` が True |
| 8 | 音声を inbox に置いて `uv run kijun ingest` → `uv run kijun transcribe --conversation <ID>` | `utterances` に発話が入る |
| 9 | `uv run kijun minutes --conversation <ID>` | `data/minutes/<ID>.md` ができる。内容を読んで、作り話や取りこぼしを確認する |
| 10 | 「Discord bot の準備」を行い、`uv run kijun discord post` | Discord に本文と Markdown の添付ファイルが投稿される |
| 11 | `uv run kijun batch run --dry-run`、続けて `uv run kijun batch run` | 手順と順序が出る。実行すると `data/batch_logs/` に記録が残り、最後に成否の表が出る |
| 12 | `uv run kijun batch install-task` が作った XML を、表示された `schtasks /create /xml` のコマンドで登録する | 翌晩にスリープ解除を含めて起動する。翌朝 `data/batch_logs/` を見る |
| 13 | ゲームを起動した状態で `uv run kijun batch run` を試す | `data/batch_logs/` に `<手順>:defer` の記録が残り、ゲームを終了すると再開する |

補足:

- DB ファイルを失ったときは、`uv run kijun db init` でスキーマを作り直してから、`uv run kijun restore-transcript --all` で `conversations` と `utterances` を文字起こしの JSON（`[paths].transcripts`）から復元する。文字起こしをやり直す必要は無い。`minutes` の行は復元されないので、`uv run kijun minutes --pending` で議事録を作り直す（Markdown のファイルがあっても、上書きされる）。
- `--audio` の試し実行（`transcribe --audio`）は DuckDB に書かず、結果を標準出力と JSON に出すだけである。VRAM 使用量は、実行中に別の端末で `nvidia-smi` を実行して記録する。
- `minutes.chunk_minutes = 15` と `llm.num_ctx = 16384` は設計者の初期値で、実測に基づかない。入力が `num_ctx` に収まらない設定は、読み込みで拒否される（Ollama は入力が文脈長を超えると、エラーを出さずに先頭を捨てるため）。
- `uv run pytest -m requires_gpu`、`uv run pytest -m requires_network` で、実機でだけ動く確認用のテストを実行できる（`KIJUN_TEST_AUDIO` に試験用の音声のパスを設定する）。

### VRAM の扱い

faster-whisper・pyannote と LLM（Ollama）は、同時に VRAM に載せない。`transcribe` と `minutes` は別のプロセスとして順に実行する。プロセスが終われば OS が GPU メモリを解放する。

Ollama は、設定 `llm.keep_alive` が `"0"`（既定）なら、応答のたびにモデルを VRAM から下ろす。したがって文字起こしの時点で LLM は載っていない。夜間バッチは LLM の起動も停止もしない。`keep_alive` を `"0"` 以外にしたときは、夜間バッチが議事録の手順の後にモデルを下ろす要求を送る。

## CLI のサブコマンド

```
uv run kijun db init                                   # スキーマを適用する（古い DB は移行する。データは消さない）
uv run kijun ingest [--dry-run]                        # inbox を走査し、会議単位に束ねて conversations に登録
uv run kijun transcribe --conversation <ID>            # 文字起こしと話者分離。utterances に書く
uv run kijun transcribe --audio <ファイル>              # 単発の試し実行
uv run kijun restore-transcript (--conversation <ID> | --all) [--force] [--dry-run]
                                                       # 文字起こしの JSON から conversations と utterances を復元
uv run kijun minutes (--conversation <ID> | --pending) # 議事録（Markdown）を作る。--conversation は作り直し
uv run kijun doctor [--out <パス>]                      # Ollama の実測（接続・スキーマ適合・GPU に収まるか・速度・入力の切り捨て）
uv run kijun discord post                              # 未投稿の議事録を Discord に投稿
uv run kijun purge-audio [--execute]                   # 音声の30日削除
uv run kijun batch run [--dry-run]                     # 夜間バッチ。--dry-run は順序とコマンドを表示するだけ
uv run kijun batch install-task [--time HH:MM] [--out <パス>]
                                                       # タスクスケジューラ登録用の XML を生成（登録は schtasks をユーザーが実行）
```

全体に共通して `--config <パス>` で設定ファイルを指定できる（省略すると `./config.toml`、無ければ既定値）。

| サブコマンド | 動作 |
|---|---|
| `db init` | `data/kijun.duckdb` にスキーマを適用する。何度実行しても安全。古い DB には `minutes` を足すだけで、古いテーブル（`extracted_requests` など）には触れない。 |
| `ingest` | `[paths].inbox` のファイルを `yyyyMMdd_HHmm_<形式>_<種類>.m4a` として解析し、同じ日で、前のファイルの終了から30分以内のものを同じ会議にまとめて `conversations` に登録する。`--dry-run` は登録せず、束ねた結果を表で出す。解析できないファイルは、削除も移動もせず、一覧に出す。 |
| `transcribe --conversation <ID>` | 登録済みの会議の音声を文字起こしして話者を付け、`utterances` に書く。同じ内容を `[paths].transcripts/<ID>.json` に保存し、音声を `[paths].processed` に移す。 |
| `restore-transcript` | `[paths].transcripts` の JSON から、`conversations` と `utterances` を復元する。`conversations` に同じ会議が既にあれば、上書きせずスキップする（`--force` で上書き）。 |
| `minutes` | 発話を `minutes.chunk_minutes` 分ごと（重なり 0）に分け、チャンクごとに Ollama で議題・要点・決定事項・ToDo・未決事項を作り、つなげた結果からもう1回の呼び出しで概要を作る。`[paths].minutes/<ID>.md` に書き、`minutes` テーブルに記録する。1つでもチャンクが失敗したら、その会議の議事録は作らない（欠けた議事録を黙って出さないため）。`--pending` は、議事録の無い文字起こし済みの会議すべて。 |
| `doctor` | Ollama を実測し、`config.toml` に書くべき値を出す。(1) 接続できるか・設定のモデルがあるか (2) `format` にスキーマを付けて、出力がそのまま読めるか (3) **モデルが GPU に収まっているか**（`/api/ps` の `size_vram` と `size`） (4) tokens/s と所要秒数、2時間の会議の見込み (5) 入力が `num_ctx` を超えて切り捨てられていないか。最後にモデルを下ろす。結果は `data/doctor/ollama_<日時>.json` にも保存する。 |
| `discord post` | 未投稿（`minutes.posted_at` が空）の議事録を投稿する。本文は見出し・概要・決定事項・ToDo を 2000 文字以内に収め、Markdown 全体を添付する。投稿済みは再投稿しない。`DISCORD_BOT_TOKEN` が要る。`discord.minutes_channel_id` が空なら何もせず、設定が必要だと出して正常終了する。 |
| `batch run` | 夜間バッチ。`ingest` → `transcribe`（会議ごと）→ `minutes`（会議ごと）→ `discord post` を、別プロセスとして順に起動する。GPU を使う手順の直前には、`batch.defer_if_running` のゲームが動いていないかを確かめ、動いていれば待つ。`ingest` が失敗したら止め、それ以外は1つの会議の失敗で止めない。各手順の記録を `batch.log_dir` に JSON で保存し、最後に成否の表を出す。失敗があれば終了コード 1。 |
| `batch install-task` | Windows のタスクスケジューラ用の XML（毎日 `--time`（省略すると `batch.task_time`）に実行。`WakeToRun` と `StartWhenAvailable` が true）を作り、登録用の `schtasks /create /xml` のコマンドを案内する。登録そのものはしない。 |
| `purge-audio` | 文字起こし完了から `retention.audio_days`（既定 30）日以上たった音声の一覧を出す。`--execute` を付けたときだけ削除する。文字起こしの JSON と議事録は削除しない。 |

## 既知の制約

- 議事録の品質（作り話、取りこぼし）は、qwen3.5:9b ではまだ十分に測っていない。`CLAUDE.md` 7章に実機での所見を書いてある。
- 議事録の生成に失敗した会議は、`minutes` に行が作られないので、翌晩も対象になる（毎晩再試行する）。
- 作り直した議事録（`minutes --conversation`）は、投稿済みの会議なら自動では再投稿されない。
- 話者は `SPEAKER_00` のようなラベルのまま扱う。人名への対応付けはしない。
- `batch.defer_if_running` に書く実行ファイル名が実際のプロセス名と違うと、ゲームが動いていても延期されない。ユーザーが「詳細」タブで確認して書く。
- `hotwords` の効果は未検証で、既定は空である。

## テスト

```powershell
uv run pytest                        # GPU とネットワークが要るテストを除いて実行
uv run pytest -m requires_gpu        # 実機で、GPU を使う確認だけ実行
uv run pytest -m requires_network    # 実機で、Ollama・Hugging Face・Discord を使う確認だけ実行
```

GPU・ネットワーク・外部プロセスに触る処理は、`typing.Protocol` で定義したインターフェース（`Transcriber`、`Diarizer`、`JsonChat`、`DiscordClient`、`CommandRunner`）の背後に置いてある。テストでは `tests/fakes.py` の偽の実装を使う。
