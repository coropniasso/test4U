# kijun

会話の録音から、自分の判断基準を「用途別のチェックリスト」として文書化するための個人用ツール。
目的と制約は `CLAUDE.md`、実装の設計は `docs/implementation-plan.md` に書いてある。

この版（第1弾）で動くのは、次の4つの作業と、それらが共有する基盤である。

| 作業表 | 内容 | サブコマンド |
|---|---|---|
| #4 | 文字起こし（faster-whisper）と話者分離（pyannote.audio） | `transcribe` |
| #5 | Qwen（Ollama）と Claude API による抽出と、結果の比較 | `extract`、`compare` |
| #6 | DuckDB のスキーマ | `db init` |
| #9 | 音声の30日削除 | `purge-audio` |

Discord bot（#7）と、夜間バッチの統合・タスクスケジューラへの登録（#8）は、まだ作っていない（第2弾）。

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
| Ollama での抽出 | `uv sync --extra extract-ollama` | httpx |
| Claude API での抽出 | `uv sync --extra extract-claude` | anthropic |
| 埋め込みによる照合 | `uv sync --extra match` | sentence-transformers |

複数の extra を同時に入れるときは `uv sync --extra transcribe --extra extract-ollama` のように並べる。
extra が足りない状態でサブコマンドを実行すると、どの extra を入れればよいかを示すエラーで終了する。

### torch を CUDA 版にする（Windows）

Windows で `uv sync --extra transcribe` をそのまま実行すると CPU 版の torch が入り、`torch.cuda.is_available()` が False になる。
CUDA 版を入れるには、`pyproject.toml` の末尾近くにある、コメントアウトした `[[tool.uv.index]]` と `[tool.uv.sources]` を有効にする。
有効にしていない理由は、uv が `uv sync` のたびに、そこに書いた index（download.pytorch.org）へ接続して全 extra の情報を取得するため、
その host に接続できない環境（外向き通信を制限したクラウド環境など）では、extra を使わなくても `uv sync` が失敗するからである。

1. https://pytorch.org/get-started/locally/ を開く。
2. Stable、Windows、Pip、Python、CUDA を選ぶ。表示される `pip install ... --index-url https://download.pytorch.org/whl/cuXXX` の `cuXXX` が、現在のタグである。
   RTX 4070 Ti は CUDA 12 系で動く。計画書では `cu126` と書いたが、これは書いた時点の推測なので、画面に出たタグを優先する。
3. `pyproject.toml` のコメントアウトした部分の `#` を外す。
4. 3 か所の `cu126` を、2 で確認したタグに書き換える（`[[tool.uv.index]]` の `name` と `url`、`[tool.uv.sources]` の `index` 名の 2 行）。
5. `uv sync --extra transcribe` を実行する。
6. `uv run python -c "import torch; print(torch.cuda.is_available())"` が `True` になることを確認する。

### pyannote の利用条件と Hugging Face のトークン

1. Hugging Face のアカウントでアクセストークンを作る（https://huggingface.co/settings/tokens）。
2. https://huggingface.co/pyannote/speaker-diarization-3.1 のページで利用条件に同意する。
3. https://huggingface.co/pyannote/segmentation-3.0 のページでも利用条件に同意する。両方に同意しないと、モデルの取得が失敗する。
4. PowerShell で `$env:HF_TOKEN = "hf_..."` を実行する。

Claude API で抽出する場合は、`$env:ANTHROPIC_API_KEY = "sk-ant-..."` を設定する。
Console で月額の上限金額を設定してから使うこと。秘密情報は環境変数だけで渡し、設定ファイルには書かない。

## ユーザーが実機で行う確認の順序

クラウド環境には GPU も音声ファイルも無いので、GPU・ネットワーク・外部プロセスに触る処理は、実機で次の順に確認する。

| # | 作業 | 成功の判定 |
|---|---|---|
| 1 | Python 3.12 と uv を入れる | `uv --version` が出る |
| 2 | `uv sync` | `uv run kijun --help` がサブコマンド一覧を出す |
| 3 | `uv run pytest` | すべて通る（GPU とネットワークが要るテストはスキップされる） |
| 4 | `uv run kijun db init` | `data/kijun.duckdb` ができ、テーブルが10個とビューが1つある |
| 5 | `pyproject.toml` の torch のインデックスを確認して `uv sync --extra transcribe` | `uv run python -c "import torch; print(torch.cuda.is_available())"` が True |
| 6 | Hugging Face で pyannote の2つのモデルページの利用条件に同意し、`HF_TOKEN` を設定 | 次の手順が通る |
| 7 | 5分程度の音声1本で `uv run kijun transcribe --audio <ファイル>` | 日本語の文字起こしと `SPEAKER_00` 等のラベルが出る。処理時間と VRAM 使用量を記録する（CLAUDE.md 7章の未検証事項） |
| 8 | `ollama list` で Qwen のタグを確認し、`config.toml` に書く | — |
| 9 | 2時間の音声で `uv run kijun extract --provider ollama` | 抽出が JSON として取れる。VRAM が足りるかを記録する（CLAUDE.md 7章の未検証事項） |
| 10 | `ANTHROPIC_API_KEY` を設定し、Console で月額上限を設定して `--provider claude` | 抽出が取れる。実際の費用を記録し、計画書 7-6 の概算と比べる |
| 11 | `uv run kijun compare` | 対照表の Markdown が出る |

補足:

- 手順4の注意（DB ファイルが既にある場合）: 埋め込みの `text_hash` 列とビュー `v_weekly_metrics` の修正より前に `kijun db init` を実行して作った DB ファイルがある場合、`embeddings` テーブルには `text_hash` 列が無い。`db init` は `CREATE TABLE IF NOT EXISTS` なので、既存のテーブルに列を足さない。
  - ビュー `v_weekly_metrics` は `CREATE OR REPLACE` なので、`kijun db init` をもう一度実行すれば新しい定義になる。
  - `embeddings` は、再計算できるキャッシュにすぎない。次の2コマンドで、他のテーブルの内容を残したまま作り直せる。
    `uv run python -c "import duckdb; duckdb.connect('data/kijun.duckdb').execute('DROP TABLE embeddings')"` を実行してから、`uv run kijun db init` を実行する（DB のパスは `[paths].db` に合わせる）。
  - 試しに作っただけの DB なら、DB ファイルを削除して `kijun db init` をやり直してもよい。文字起こしの JSON（`[paths].transcripts`）は DB ファイルとは別に残る。ただし、JSON から DB の `utterances` を読み込み直すコマンドは、この版には無い。文字起こし済みの会議が入った DB を削除すると、その会議は `kijun transcribe` をやり直さない限り DB に戻らない。GPU 時間を無駄にしないよう、文字起こし済みのデータがある DB は、削除せず上の方法で直すこと。
- 手順4の10テーブルは、CLAUDE.md 5章の8テーブルに、`schema_version` と `embeddings` の2テーブルを足したものである。ビューは `v_weekly_metrics`（抽出モデルごとの週次の指標）の1つ。
- 手順7の `--audio` は DuckDB に書かず、結果を標準出力と JSON に出すだけである。VRAM 使用量は、実行中に別の端末で `nvidia-smi` を実行して記録する。
- 手順7、9で記録した処理時間と VRAM 使用量は、CLAUDE.md 7章の未検証事項「Qwen の最新版と、VRAM 12GB での実際のメモリ使用量・処理時間」と「pyannote.audio が Windows + CUDA 環境で問題なく動くか」への回答になる。結果を CLAUDE.md 7章に書き込む。
- 手順10の Claude 側の抽出は、動作確認として `claude-sonnet-5-5`（`effort = "low"`）を使う。承認件数が足りなければ `config.toml` の `extract.claude.effort` を `medium` に上げる。
- `uv run pytest -m requires_gpu`、`uv run pytest -m requires_network` で、実機でだけ動く確認用のテストを実行できる（`KIJUN_TEST_AUDIO` に試験用の音声のパスを設定する）。

### VRAM の扱い

faster-whisper・pyannote と Qwen は、同時に VRAM に載せない。`transcribe` と `extract` は別のプロセスとして順に実行する。
プロセスが終われば OS が GPU メモリを解放する。Ollama は、リクエストに `keep_alive: 0` を付けて、1回の抽出が終わるとモデルをアンロードさせる。

## CLI のサブコマンド

```
uv run kijun db init                                   # スキーマを適用する
uv run kijun ingest [--dry-run]                        # inbox を走査し、会議単位に束ねて conversations に登録
uv run kijun transcribe --conversation <ID>            # 文字起こしと話者分離。utterances に書く
uv run kijun transcribe --audio <ファイル>              # 単発の試し実行（作業表 #4 の用途）
uv run kijun extract --conversation <ID> --provider {ollama,claude}
uv run kijun compare --conversation <ID> [--out <パス>]  # Qwen と Claude の対照表（作業表 #5）
uv run kijun match --conversation <ID> [--dry-run]     # 既存項目との照合
uv run kijun purge-audio [--execute]                   # 音声の30日削除
```

全体に共通して `--config <パス>` で設定ファイルを指定できる（省略すると `./config.toml`、無ければ既定値）。

| サブコマンド | 動作 |
|---|---|
| `db init` | `data/kijun.duckdb` にスキーマを適用する。何度実行しても安全。 |
| `ingest` | `[paths].inbox` のファイルを `yyyyMMdd_HHmm_<形式>_<種類>.m4a` として解析し、同じ日で、前のファイルの終了から30分以内のものを同じ会議にまとめて `conversations` に登録する。`--dry-run` は登録せず、束ねた結果を表で出す。解析できないファイルは、削除も移動もせず、一覧に出す。 |
| `transcribe --conversation <ID>` | 登録済みの会議の音声を文字起こしして話者を付け、`utterances` に書く。同じ内容を `[paths].transcripts/<ID>.json` に保存し、音声を `[paths].processed` に移す。 |
| `transcribe --audio <ファイル>` | 1本の音声を試す。DuckDB には書かない。 |
| `extract` | 発話を25分ごと（前後2分重ねる）に分け、Ollama または Claude API で要求・指摘を抽出し、`extracted_requests` に書く。`--provider` を省略すると設定の `extract.provider`。失敗したチャンクは記録して次に進む。 |
| `compare` | 同じ会議の Qwen と Claude の抽出結果を、数字の表と、左右に並べた対照表の Markdown にする。 |
| `match` | `reusable = true` の抽出結果を、既存のチェック項目とコサイン類似度で照合する。閾値（既定 0.85）以上なら既存項目に紐付け、未満なら `status = 'candidate'` の新規候補を作る。`--dry-run` は書かず、類似度の降順の表を出す。`--provider` で、どのモデルの抽出結果を対象にするかを選ぶ（省略すると設定の `extract.provider`）。 |
| `purge-audio` | 文字起こし完了から `retention.audio_days`（既定 30）日以上たった音声の一覧を出す。`--execute` を付けたときだけ削除する。文字起こしの JSON は削除しない。 |

## 第1弾の制約

- 承認件数は、レビューして初めて決まる数字なので、`compare` の出力には出ない。第1弾では、出力した Markdown に手で印を付ける（Discord での記録は第2弾）。
- `extracted_requests.direction`（自分→相手 / 相手→自分）は、話者と人物の対応付け（`speaker_map`、第2弾の Discord）が済むまで確定できないので、NULL のままにする。
- `knowledge_items.importance`（重要度）は、Discord への投稿時に判定する処理なので、第2弾まで NULL のままにする。
- `artifact_type` と `counterpart_role` の値は、LLM が自由な文字列を入れる。3回分の抽出結果を見て、よく出る値を決める。
- 1つの会議が複数ファイルに分かれたときの束ね結果の確認は、第1弾では `ingest --dry-run` の標準出力で行う。
- 失敗したチャンクの記録は、DuckDB のスキーマに列が無いので、DB と同じディレクトリの `extract_logs/` に JSON で保存する。`compare` がここを読む。

## テスト

```powershell
uv run pytest                        # GPU とネットワークが要るテストを除いて実行
uv run pytest -m requires_gpu        # 実機で、GPU を使う確認だけ実行
uv run pytest -m requires_network    # 実機で、Ollama・Claude API・Hugging Face を使う確認だけ実行
```

GPU・ネットワーク・外部プロセスに触る処理は、すべて `typing.Protocol` で定義したインターフェース
（`Transcriber`、`Diarizer`、`Extractor`、`Embedder`）の背後に置いてある。テストでは `tests/fakes.py` の偽の実装を使う。
