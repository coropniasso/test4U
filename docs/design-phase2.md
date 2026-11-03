# 第2弾の設計：Discord でのレビューと夜間バッチ

作成日：2026-11-03
対象：`CLAUDE.md` 8章の作業表 #7（Discord bot）と #8（夜間バッチの統合とタスクスケジューラ登録）
この文書は実装のための仕様である。実装が終わったら内容を `docs/implementation-plan.md` に「実装済みの記述」として統合し、この文書は削除する。

---

## 0. この2つを作る意味

`docs/implementation-plan.md` 1節の表で、段5（ユーザーが抽出された指摘に理由と適用条件を書き足す工程）が未実装である。`CLAUDE.md` 1章はこの工程を「この仕組みの中心」と呼んでいる。#7 はその工程の受け皿を作る。

#7 ができると、`CLAUDE.md` 6章の指標のうち測れていなかった2つ（「LLM の提案文言を修正せず承認した割合」「1日のレビュー所要時間」）が測れるようになる。これは、2026-11-03 に抽出の背面を Strata 1つにしたことで失われた判断手順の代わりである（`CLAUDE.md` 10章）。

#8 は、#4〜#7 を毎晩自動で回す。

---

## 1. 守る前提

### 1-1. bot は常駐しない

`CLAUDE.md` 4-6 の「PCの稼働時間の制約への対応」に従う。ボタン操作を受け付ける常駐 bot は PC が落ちていると反応しないので、**リアクションと返信を翌晩のバッチでまとめて読み取る**。

したがって Discord の処理は2つのサブコマンドに分ける。夜間バッチが順に起動し、処理が終われば接続を閉じる。

- `kijun discord read`：前回までに投稿したメッセージのリアクションと返信を読み取り、DuckDB に反映する（4-8 の手順1）
- `kijun discord post`：新規候補・矛盾の可能性がある組み合わせ・話者の対応付けの依頼・束ね結果を投稿する（4-8 の手順8）

`read` を先に実行するのは `CLAUDE.md` 4-8 の処理順どおりである。前夜の投稿への反応を取り込んでから、その晩の新しい候補を投稿する。

### 1-2. 既存の4つの原則をそのまま適用する

`docs/implementation-plan.md` 2節の原則をそのまま使う。特に次の3点を守る。

- **原則2-1（差し替え可能な境界）**：Discord に触る処理は `DiscordClient` という Protocol の背後に置く。実際の Discord に接続せずにテストする。
- **原則2-3（確認できない前提は設定値と実測に委ねる）**：LLM に渡す処理は `enforces_json_schema` と `repair.py` を通す。抽出と同じ扱いにする。
- **原則2-4（失うより止める）**：判定できない入力は、推測で決めずに未処理のまま残す。1-5 と 2-4 に具体を書く。

### 1-3. LLM の使い道が2つ増える

抽出に加えて、次の2つで LLM を使う。どちらも `transport.py` と `config.extract.strata` の設定を再利用する（同じ Strata のサーバーに投げる）。

| 用途 | 入力 | 出力 | 根拠 |
|---|---|---|---|
| 重要度の3段階判定 | チェック項目の文言と、その根拠になった発言 | 1〜3 の整数と、根拠を述べた1文 | `CLAUDE.md` 4-6「初期は、LLM が3段階で判定して根拠を1文添え、ユーザーがリアクションで上書きする」 |
| スレッド返信の振り分け | ユーザーが書いた自由な文章 | `reason` / `applies_when` / `check_text` のどれに入れるかと、その内容 | `CLAUDE.md` 4-6「返信はLLMに渡して `reason` / `applies_when` / `check_text` に振り分ける」 |

`Extractor` は再利用しない。入出力のスキーマが違うためである。`extract/base.py` と同じ形の Protocol を用途ごとに作る。

**返信の原文は必ず `review_log` に残す。** LLM の振り分けが間違っても、原文から人が直せるようにするためである。振り分けの結果だけを保存してはいけない。

### 1-4. 投稿件数に上限を置く

`CLAUDE.md` 6章の「レビュー所要時間が1日15分を超え続けたら、1回の投稿件数の上限を下げる」に対応する。1回の `post` で投稿する新規候補の件数を設定値 `discord.max_candidates_per_run`（既定20）で制限する。上限を超えた分は投稿せず、次回に回す。投稿していない候補の件数を標準出力に出す。

投稿の順序は、重要度の降順、同じ重要度なら新しい順にする。レビューの時間が足りないときに、重要なものが先に目に入るようにするためである。

### 1-5. 判定できないリアクションは、推測で決めない

`CLAUDE.md` 4-6 のリアクションは ✅承認 / ❌却下 / ⏸保留 / 1️⃣2️⃣3️⃣重要度 である。実際には次の状態が起こり得る。

| 状態 | 扱い |
|---|---|
| status のリアクションが1つだけ | そのとおりに `knowledge_items.status` を更新する |
| status のリアクションが**2つ以上**（✅と❌が両方など） | **status を変えない。** `review_log` に「判定できなかった」と記録し、`discord_posts.processed_at` を**埋めない**。次回の `read` で再び対象になる。標準出力に会議IDと項目IDを出す |
| 重要度のリアクションが2つ以上 | 同様に `importance` を変えない |
| status が無く重要度だけ | 重要度のみ更新する |
| どのリアクションも無い | 何もしない。`processed_at` を埋めない（後から付ける可能性があるため） |

2つ以上のときに片方を優先しないのは原則2-4のためである。誤って承認すると、誤ったチェック項目が「承認済み」として残り、それが間違いであることが後から分かりにくい。判定を保留して人に見せるほうが安全である。

---

## 2. #7 Discord bot

### 2-1. ファイル構成

```
src/kijun/discord/
├── base.py         # Protocol: DiscordClient、Reaction、Reply
├── impl.py         # discord.py を使う実装
├── format.py       # 投稿する文面を作る（純粋な文字列処理。テストできる）
├── post.py         # 4種類の投稿
├── read.py         # リアクションと返信の読み取り
└── run.py          # CLI の本体
src/kijun/review/
├── importance.py   # 重要度の3段階判定（LLM）
├── reply.py        # スレッド返信の振り分け（LLM）
├── speaker.py      # 話者の対応付けの返信の解釈、speaker_map への反映
├── direction.py    # speaker_map から extracted_requests.direction を導出する
└── contradiction.py # 矛盾の可能性がある組み合わせの検出（埋め込み）
```

### 2-2. `DiscordClient`（`discord/base.py`）

```python
@dataclass(frozen=True)
class Reaction:
    emoji: str
    count: int

@dataclass(frozen=True)
class Reply:
    author_is_owner: bool   # サーバーの所有者（ユーザー本人）の返信か
    content: str
    created_at: datetime

class DiscordClient(Protocol):
    def post(self, channel_id: str, content: str) -> str: ...
    def fetch_reactions(self, channel_id: str, message_ids: list[str]) -> dict[str, list[Reaction]]: ...
    def fetch_thread_replies(self, channel_id: str, message_ids: list[str]) -> dict[str, list[Reply]]: ...
```

`post` は `message_id` を返す。呼び出し側が `discord_posts` に記録する。

`Reply.author_is_owner` を持たせるのは、ユーザー本人以外（他のメンバー、bot 自身）の返信を取り込まないためである。`CLAUDE.md` 2章により、このチャンネルはユーザー個人のサーバーに作るので他人は居ない想定だが、bot 自身の投稿を自分で読み戻すことは起こり得る。

`impl.py` は `discord.py` を使う。`deps.py` の仕組みで遅延 import し、依存が無ければ `uv sync --extra discord` を示す。接続は処理の間だけ開き、終わったら閉じる（1-1）。**Message Content Intent を有効にしていないと返信の本文が読めない**ので、本文が空で返ってきた場合は、Developer Portal で Message Content Intent を有効にする必要があることを含めたエラーメッセージを出す（`CLAUDE.md` 4-6 の bot の準備の2番）。

### 2-3. 投稿する文面（`discord/format.py`）

`CLAUDE.md` 4-6 のメッセージの例に合わせる。文面を作る処理は外部に接続しないので、すべてテストする。

新規候補の文面は `CLAUDE.md` 4-6 の例をそのまま使う。

```
【新規候補 #0142】重要度案：2
場面：9/20 プロジェクト定例／後輩の告知文レビュー
発言：「募集要項の最初に、何をしてほしいかを書いて」（あなた→後輩A）
チェック項目案：告知文の冒頭2行以内に、読み手にしてほしい行動が書かれているか
理由：記録なし
適用条件：記録なし
→ ✅❌⏸でstatus、1️⃣2️⃣3️⃣で重要度。理由・条件・修正はスレッドに返信
```

守る点を列挙する。

- 「理由：記録なし」「適用条件：記録なし」は、`stated_reason` / `applies_when` が NULL のときの表示である。**NULL を空欄にせず「記録なし」と書く。** 語られなかったことが見えている必要がある（`CLAUDE.md` 1章）。LLM が推測で埋めていないことの確認にもなる。
- 「（あなた→後輩A）」の部分は `direction` と `speaker_map` から作る。`speaker_map` が未確定なら「（SPEAKER_00→SPEAKER_01）」とラベルのまま出す。推測しない。
- 「場面」は `conversations.recorded_at` の月日と `setting_kind` の日本語表示から作る。`setting_kind` の日本語への対応（`teirei` → 定例、`sagyou` → 作業、`zatsudan` → 雑談）は `format.py` に辞書を置き、未知の種類は英数字のまま出す。
- 「重要度案」は LLM の判定値。判定できなかった場合は「重要度案：なし」と出す。
- Discord の1メッセージは2,000文字までである。超える場合は `quote` を末尾から切り詰め、切り詰めたことが分かる印（`…`）を付ける。チェック項目案と理由・適用条件は切り詰めない。これらがレビューの対象だからである。

### 2-4. 読み取り（`discord/read.py`）

1. `discord_posts` から `processed_at` が NULL の行を取り出す。
2. `post_type` ごとに処理を分ける。
   - `new_candidate`：リアクションから `status` と `importance` を決める（1-5 の表）。スレッドの返信があれば `review/reply.py` に渡して振り分ける。
   - `speaker_sample`：返信を `review/speaker.py` に渡して `speaker_map` に反映する。
   - `bundle_result`：❌ が付いていたら、束ね直しが必要であることを標準出力に出す。**自動で分割し直さない。** 会議の束ね方を機械が勝手に変えると、どの音声がどの会議に属するかが分からなくなる。ユーザーが `config.toml` の `bundle.gap_minutes` を調整して `kijun ingest` をやり直す運用にする。
   - `contradiction`：返信があれば、どちらの項目に対する `applies_when` の追記かを `review/reply.py` に渡す。リアクションでは何も変えない。
3. 変更はすべて `review_log` に記録する。`before` と `after` に変更前後の値を入れる。
4. 処理できた投稿の `discord_posts.processed_at` を埋める。判定できなかったものは埋めない（1-5）。
5. `speaker_map` に変更があった場合、最後に `review/direction.py` を呼んで `extracted_requests.direction` を導出する。

### 2-5. 重要度の判定（`review/importance.py`）

LLM に、チェック項目の文言（`check_text`）と、その根拠になった発言（`item_evidence` 経由で引いた `extracted_requests.quote` を最大3件）を渡し、1〜3 の整数と根拠1文を返させる。

- 出力のスキーマは pydantic で定義し、`repair.py` を通してから検証する（原則2-3）。
- 判定に失敗したら `importance` を NULL のままにする。処理は止めない。
- 判定値は `knowledge_items.importance` に入れ、根拠の1文は `review_log` に `action = 'importance_proposed'`、`after` に根拠の文を入れて記録する。
- ユーザーのリアクションが上書きしたら、`action = 'importance_overridden'`、`before` に LLM の判定値、`after` にユーザーの値を入れて記録する。

**この上書きの履歴の形を今決めておく理由**は、`CLAUDE.md` 4-6 の「2〜3か月後、上書き履歴を目的変数にして、ルールの重みを回帰で推定し直す」のためである。後から形を変えると、それまでの履歴が分析に使えなくなる。目的変数（ユーザーの値）と、LLM の判定値の両方が、1行から取れる形にしておく。

回帰の説明変数の候補（`CLAUDE.md` 4-6）は「直近30日の出現回数」「成果物の種類」「同じ指摘をした相手の人数」である。これらは `extracted_requests` と `item_evidence` から後から計算できるので、いま列を増やす必要はない。

### 2-6. スレッド返信の振り分け（`review/reply.py`）

ユーザーの自由な文章を、`reason` / `applies_when` / `check_text` のどれに入れるかを LLM が決める。

- 1つの返信が複数の欄に対応することがある（「冒頭に書かないと読み飛ばされるから。ただし社内向けの連絡では不要」は `reason` と `not_applies_when` の両方）。出力は欄ごとの配列にする。
- 対応する欄は `reason`、`applies_when`、`not_applies_when`、`check_text` の4つにする。`CLAUDE.md` 4-6 は3つを挙げているが、`knowledge_items` には `not_applies_when` の列があり（`CLAUDE.md` 5章）、「どの状況なら同じ指摘をしないか」は `CLAUDE.md` 1章が暗黙知の中身として挙げている2点のうちの1つである。振り分け先に無いと記録できない。
- **返信の原文を `review_log` に `action = 'reply_received'`、`after` に全文を入れて必ず記録する。** そのうえで、振り分けの結果を `action = 'reply_routed'` として別の行に記録する。LLM の振り分けが間違っても原文から直せるようにするため（1-3）。
- 振り分けに失敗したら、原文の記録だけ残し、`knowledge_items` は変えない。`processed_at` は埋めず、次回の `read` で再試行する。
- `check_text` を変更した場合、`embeddings` の `text_hash` が変わるので次回の `kijun match` でベクトルが再計算される。これは既存の仕組みで自動的に起こる（`docs/implementation-plan.md` 9-1）。

### 2-7. 話者の対応付け（`review/speaker.py`）

会議ごとに、各 `speaker_label` の発言サンプルを3件ずつ投稿する。サンプルは**その話者の発話のうち文字数が多い上位3件**を選ぶ。「はい」「そうですね」のような短い発話では誰か分からないためである。

ユーザーの返信を2通りの形式で受ける。

1. **構造化した形式**（投稿の文面で案内する）：`00=self 01=kouhai_a` のように書かれていれば、LLM を使わずに正規表現で解釈する。確実で、費用も時間もかからない。
2. **自然な文章**（`CLAUDE.md` 4-6 が想定している形）：「00は自分、01は後輩A」。1で解釈できなかった場合に LLM に渡す。

`person_label` は英数字で持つ（`docs/implementation-plan.md` 4-2）。自然文から英数字への変換は LLM が行う。ただし **`self`（ユーザー本人）だけは特別扱いする**。`direction` の導出がこれに依存するためで、`self` が1つも決まらなかった会議は、その旨を標準出力に出し、`direction` を導出しない。

投稿の文面には、構造化形式の例を必ず入れる。

```
【話者の対応付け】20260920_1930_f2f_teirei
SPEAKER_00 の発言サンプル：
  「募集要項の最初に、何をしてほしいかを書いて」
  （以下2件）
SPEAKER_01 の発言サンプル：
  （3件）
→ このメッセージに返信してください。
   書き方（確実）：00=self 01=kouhai_a
   自然な文でも読み取ります：「00は自分、01は後輩A」
   自分を必ず self と書いてください（自分→相手の判定に使います）
```

### 2-8. `direction` の導出（`review/direction.py`）

`docs/implementation-plan.md` 4-1 で予告していた処理である。`speaker_map` が埋まった会議について、`extracted_requests` の `direction` を導出する。

- `speaker_from_label` に対応する `person_label` が `self` なら `self_to_other`
- `speaker_to_label` に対応する `person_label` が `self` なら `other_to_self`
- どちらも `self` でない（第三者間のやり取り）なら NULL のままにする
- 両方 `self`（独り言、または話者分離の誤り）なら NULL のままにする

独立したコマンド `kijun derive-direction [--conversation <ID> | --all]` としても実行できるようにする。`speaker_map` を手で直した後に再実行できるようにするためである。

変更は `review_log` に記録しない。`review_log` はユーザーのレビューの履歴を残す表であり、機械的な導出はそこに混ぜない。

### 2-9. 矛盾の可能性がある組み合わせ（`review/contradiction.py`）

**この検出規則は `CLAUDE.md` に書かれていない。設計者（Claude）が考案したものであり、推奨・未確定である。** ユーザーが運用して見直す前提で、閾値はすべて設定値にする。

`CLAUDE.md` 4-6 は「過去の指摘と矛盾する可能性がある発言の組み合わせを投稿する。多くは矛盾ではなく適用条件の違いであり、適用条件を言語化する材料になる」としている。つまり目的は矛盾を証明することではなく、**同じ話題で基準が食い違っている組を人に見せること**である。

規則：`knowledge_items` のうち `status` が `approved` または `candidate` の組 (A, B) で、次のすべてを満たすものを投稿する。

1. `artifact_type` が同じ（違う成果物なら基準が違って当然）
2. `check_text` のコサイン類似度が `discord.contradiction_min_similarity`（既定0.70）以上
3. 同じ類似度が `match.similarity_threshold`（既定0.85）**未満**（0.85以上は同一項目として統合される側なので、矛盾ではなく重複）
4. `pass_criterion` が両方とも非 NULL で、文言が異なる
5. まだ `discord_posts` に `post_type = 'contradiction'` として投稿されていない組

1回の `post` で投稿する組の数も `discord.max_contradictions_per_run`（既定5）で制限する。

ベクトルは既存の `embeddings` から取る。無ければその項目は対象外にする（この処理のために埋め込みモデルを読み込まない。原則2-2の VRAM の都合）。

### 2-10. データベースの変更

`discord_posts` は既に存在する。1つだけ列を足す。

| 変更 | 内容 | 理由 |
|---|---|---|
| `discord_posts` に `channel_id VARCHAR` を足す | 投稿先のチャンネルID | 読み取りのときにどのチャンネルを見ればよいかが必要。`post_type` から引けるが、チャンネルIDは設定値なので後から変わり得る。投稿時の値を記録しておく |
| `discord_posts` に `pair_item_id VARCHAR` を足す | 矛盾の組のもう一方の `item_id` | 2-9 の条件5（既に投稿した組を再投稿しない）の判定に必要。`item_id` と `pair_item_id` の両方で組を表す |

`schema.sql` を直接書き換え、`SCHEMA_VERSION` を 2 に上げる。既存の DB がある場合に `CREATE TABLE IF NOT EXISTS` が列を足さない問題は、`migrate.py` に「バージョン1から2への移行」として `ALTER TABLE ... ADD COLUMN` を書いて解決する。README の手順4の注記も直す。

`review_log` の `action` に使う値を1か所に定数で定義する。`status_changed`、`importance_proposed`、`importance_overridden`、`reply_received`、`reply_routed`、`ambiguous_reaction`、`check_text_edited`。

---

## 3. #8 夜間バッチ

### 3-1. 薄いオーケストレーターにする

`CLAUDE.md` 4-8 の手順1〜9を順に実行する。**既存の CLI を `subprocess` で別プロセスとして起動する。** 原則2-2（プロセスの境界を VRAM の境界にする）を守るためで、1つの Python プロセスの中で全部を呼ぶと faster-whisper と Strata が同時に VRAM を使う危険が出る。

```
uv run kijun batch run [--dry-run]
```

`--dry-run` は、実行する順序と各コマンドを表示するだけで何も実行しない。

| 手順 | 実行するもの | 失敗したとき |
|---|---|---|
| 1 | `kijun discord read` | 続ける（前夜の反応が取り込めなくても、その晩の処理はできる） |
| 2 | `kijun ingest` | **止める**（会議が登録できなければ以降が何もできない） |
| 3-4 | `kijun transcribe --conversation <ID>`（会議ごと） | その会議だけ飛ばして次の会議に進む |
| 5 | （手順3-4のプロセスの終了。OS が GPU メモリを解放する） | — |
| 6 | `extract.strata.start_command` → `kijun extract --conversation <ID>` → `stop_command` | その会議だけ飛ばす |
| 7 | `kijun match --conversation <ID>` | その会議だけ飛ばす |
| 8 | `kijun discord post` | 続ける |
| 9 | 処理済み音声を `processed` へ移動（`transcribe` が既に行う） | — |

手順6の `start_command` / `stop_command` を読む処理をここで実装する。`docs/implementation-plan.md` 10節で「設定に置いただけで、読む処理は未実装」と書いていた箇所である。空文字なら何もしない。`stop_command` は、手順6が失敗しても必ず実行する（Strata が VRAM を保持したまま残らないようにするため）。

手順2で止める以外は、1つの会議の失敗で全体を止めない。`CLAUDE.md` 4-8 は毎晩動く前提なので、1本の音声の問題で他の音声が処理されないのは困る。

### 3-2. 実行の記録

各手順の開始時刻・終了時刻・終了コード・標準出力の末尾を `batch.log_dir`（既定 `./data/batch_logs`）に `<日時>.json` として保存する。最後に全手順の成否を表で標準出力に出す。

ユーザーが翌朝これを見て、どの段で失敗したかが分かるようにする。実機で未検証の部分が多いので、失敗の記録が最初の数週間の主な情報源になる。

### 3-3. タスクスケジューラへの登録

```
uv run kijun batch install-task [--time 23:30] [--out <パス>]
```

Windows のタスクスケジューラに登録するための XML を生成する。登録そのものは `schtasks /create /xml` でユーザーが実行する（管理者権限が必要なので、コマンドを生成して案内する形にする）。

XML に必ず入れる設定は次の2つである。

- `<WakeToRun>true</WakeToRun>`：`CLAUDE.md` 4-8 の「タスクを実行するためにスリープを解除する」。これが無いと PC がスリープしていると実行されない。
- `<StartWhenAvailable>true</StartWhenAvailable>`：PC が電源オフだった場合に、次に起動したときに実行する。

XML の生成は純粋な文字列処理なのでテストする。生成した XML が整形式であることを `xml.etree.ElementTree` で検証するテストを書く。

---

## 4. 設定の追加

```toml
[discord]
# Discord のチャンネルID（数字の文字列）。空文字なら post と read は何もせず、
# 設定が必要であることを出して終わる。
review_channel_id = ""
contradictions_channel_id = ""
# 1回の post で投稿する新規候補の上限（CLAUDE.md 6章のレビュー所要時間の調整用）。
max_candidates_per_run = 20
# 1回の post で投稿する矛盾の可能性がある組の上限。
max_contradictions_per_run = 5
# 矛盾の候補とみなす類似度の下限。上限は match.similarity_threshold（0.85）で、
# それ以上は重複として扱う。この規則は設計者が考案したもので、運用して見直す。
contradiction_min_similarity = 0.70
# 話者の対応付けで投稿する発言サンプルの件数（CLAUDE.md 4-6 は3件）。
speaker_sample_count = 3

[batch]
# 各手順の記録を置く場所。
log_dir = "./data/batch_logs"
```

必要な資格情報が1つ増える。

| 環境変数 | 用途 |
|---|---|
| `DISCORD_BOT_TOKEN` | Discord bot。Developer Portal で取得する |

依存が1つ増える。`discord = ["discord.py>=2.4"]` を extra に足す。無料である（運転費用はゼロのまま）。

---

## 5. テストで必ず確認すること

実際の Discord に接続せず、偽の `DiscordClient` と偽の LLM を使って確認する。

| # | 対象 | 確認する内容 |
|---|---|---|
| 1 | 文面の生成（2-3） | `stated_reason` / `applies_when` が NULL のとき「記録なし」と出ること。`speaker_map` が未確定なら話者ラベルのまま出ること。2,000文字を超えたら `quote` だけが切り詰められ、チェック項目案と理由・適用条件は切り詰められないこと |
| 2 | 判定できないリアクション（1-5） | status が2つ以上付いていたら `knowledge_items` が変わらず、`processed_at` が埋まらず、`review_log` に記録されること。次回の `read` で再び対象になること |
| 3 | 返信の原文の保存（1-3、2-6） | 振り分けが失敗しても原文が `review_log` に残ること。成功時も原文と振り分け結果が別の行で残ること |
| 4 | 話者の対応付け（2-7） | `00=self 01=kouhai_a` 形式が LLM を使わずに解釈されること。自然文は LLM に渡ること。`self` が決まらない会議で `direction` を導出しないこと |
| 5 | `direction` の導出（2-8） | 自分→相手、相手→自分、第三者間（NULL のまま）、両方 self（NULL のまま）の4通り |
| 6 | 矛盾の検出（2-9） | 類似度が0.70未満なら対象外、0.85以上なら対象外、`artifact_type` が違えば対象外、`pass_criterion` が同じなら対象外、既に投稿済みの組は再投稿されないこと |
| 7 | 投稿件数の上限（1-4） | 上限を超えた分が投稿されず、次回に回ること。重要度の降順に投稿されること |
| 8 | スキーマの移行（2-10） | バージョン1の DB に `ALTER TABLE` が適用され、既存の行が残ること |
| 9 | バッチの手順（3-1） | 手順2が失敗したら止まること。手順3-4が1会議で失敗しても次の会議に進むこと。手順6が失敗しても `stop_command` が実行されること |
| 10 | タスクスケジューラの XML（3-3） | 整形式であること。`WakeToRun` と `StartWhenAvailable` が true であること |

5番と2番を特に重視する。5番は `direction` が決まらないと `CLAUDE.md` 5章の「相手別の要求一覧」が作れないため、このプロジェクトの目的に直結する。2番は、誤って承認したチェック項目が「承認済み」として残ると、それが間違いであることが後から分かりにくいためである。

---

## 6. この設計で未解決のまま残ること

| 事項 | 現状 |
|---|---|
| Discord に実際に接続したときの動作 | 未検証。ユーザーが Developer Portal でアプリケーションと bot を作り、Message Content Intent を有効にし、個人サーバーに招待するまで確認できない（`CLAUDE.md` 4-6） |
| 矛盾の検出規則 | 設計者が考案したもので、`CLAUDE.md` に根拠が無い。閾値を設定値にしてあるので運用して見直す |
| 重要度の回帰による再推定 | `CLAUDE.md` 4-6 の「2〜3か月後」の作業。上書き履歴を残す形だけ決めてある |
| 束ね結果に ❌ が付いたときの再分割 | 自動では行わない。ユーザーが `bundle.gap_minutes` を調整して `kijun ingest` をやり直す |
| 1日のレビュー所要時間の測定 | `discord_posts.posted_at` と `review_log.reviewed_at` の差から概算できるが、ユーザーが画面を見ていた時間とは一致しない。ビューの追加は運用してから決める |
