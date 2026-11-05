"""設定ファイル（config.toml）の読み込みと既定値。

読み込み順は次のとおり。
1. config.toml に書かれた値
2. 書かれていない項目は、このファイルの既定値

既定値は config.example.toml と一致させる（tests/test_config.py が一致を検査する）。
秘密情報（pyannote のモデル取得に使う HF_TOKEN、Discord bot の DISCORD_BOT_TOKEN）は
設定ファイルに書かず、環境変数で渡す。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG_PATH = Path("config.toml")


class _Section(BaseModel):
    """設定の1区画。未知のキーは綴り間違いの可能性が高いので拒否する。"""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    inbox: Path = Path("./data/inbox")
    processed: Path = Path("./data/processed")
    transcripts: Path = Path("./data/transcripts")
    db: Path = Path("./data/kijun.duckdb")


class SettingsConfig(_Section):
    formats: list[str] = Field(default_factory=lambda: ["f2f", "call"])
    kinds: list[str] = Field(default_factory=lambda: ["teirei", "sagyou", "zatsudan"])


class BundleConfig(_Section):
    gap_minutes: int = 30
    require_same_setting: bool = True


class TranscribeConfig(_Section):
    backend: str = "faster_whisper"
    model: str = "large-v3"
    compute_type: str = "float16"
    device: str = "cuda"
    language: str = "ja"
    vad_filter: bool = True
    min_utterance_sec: float = 0.4
    # 結合した発話の長さの上限（秒）。0 は「上限なし」。
    max_utterance_sec: float = 30.0
    # faster-whisper の hotwords に渡す語。空でなければ、空白でつないだ1つの文字列にして
    # WhisperModel.transcribe(..., hotwords=...) に渡す。誤認識された頻出語を正しく取らせるための手段。
    # 空のリストなら渡さない。効果は未検証で、その語を言っていない箇所に当てはめる副作用がある
    # （CLAUDE.md 7章）。
    hotwords: list[str] = Field(default_factory=list)


class DiarizeConfig(_Section):
    backend: str = "pyannote"
    model: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    # TOML には null が無いので、0 を「指定なし（pyannote が推定する）」として扱う。
    num_speakers: int = 0


# Strata が答え用の取り置きを計算する下限（control/serve/server.py の ANSWER_ROOM_MIN_CAP）。
# max_tokens がこれ未満だと逆算が働かず、推論が本当に途中で切られて答えが1文字も出ない。
ANSWER_ROOM_MIN_CAP = 1024


class StrataConfig(_Section):
    base_url: str = "http://localhost:8080"
    chat_path: str = "/v1/chat/completions"
    model: str = "qwen3.8-flash-next"
    timeout_sec: int = 3600
    enforces_json_schema: bool = False
    # 0 は「未確認」を表す（TOML に null が無いため。diarize.num_speakers と同じ扱い）。
    # 未確認のときは chunk_minutes の妥当性を検査しない。
    max_input_tokens: int = 0
    # 1リクエストの出力トークン数の上限（推論トークンを含む）。0 は上限なし（payload に入れない）。
    # この値は2つのことを同時に決める。1つは1チャンクの所要時間の上限、もう1つは推論に使える
    # 予算である。Strata は max_tokens から答え用の分を取り置き、残りを推論の予算にする。
    #   取り置き = min(max(512, max_tokens // 4), max_tokens // 2)
    #   推論の予算 = max_tokens - 取り置き（≒ max_tokens の 4分の3）
    # 予算に達すると推論は切られるのではなく、閉じられて答えが書かれる。壊れないが品質は落ちる。
    #
    # 既定値 20000 は、既定の reasoning_effort（medium、推論 2,641 トークン）には十分である。
    # **ただし xhigh には足りない。** 2026-11-04 の実測で、xhigh はこの入力に
    # 推論 21,201 トークンを要した。20000 の予算は 15,000 なので、推論が閉じられる。実測では
    # その設定（= この既定値のまま）が3つの設定の中で stated_reason と pass_criterion が
    # 最も少なかった（3/6 と 3/6。xhigh を予算内で動かせば 4/6 と 4/6）。
    # したがって reasoning_effort と一緒に決める必要がある。目安は次のとおり。
    #   - reasoning_effort = "medium" にする場合: 20000 で足りる（実測の推論は 2,641）。
    #     1チャンクの最悪値は 20000 ÷ 14.0（観測された最も遅い速度）≒ 1,429 秒。
    #   - reasoning_effort = "xhigh" のまま品質を取る場合: 32000 以上にする（予算 24,000）。
    #     1チャンクの最悪値は 32000 ÷ 14.0 ≒ 2,286 秒。2時間の会議（約6チャンク）で約3.8時間。
    # 以前ここに書いていた「実測 13,050 トークンに余裕を持たせた値」という根拠は取り消す。
    # 13,050 は max_tokens を送らずに測った1点で、同じ入力でも 48,764 以上まで振れていた。
    max_tokens: int = Field(default=20000, ge=0)
    # 1リクエストで推論に使ってよいトークン数の上限。OpenAI 互換 API の reasoning_budget_tokens に
    # 渡す。0 なら送らず、Strata が max_tokens から逆算した予算（下記）に任せる。
    # **明示しても、max_tokens からの取り置きで切り下げられる**（2026-11-04 実機で確認）。
    # 実効の予算 = min(reasoning_budget_tokens, max_tokens - 取り置き)。
    # 取り置き = min(max(512, max_tokens // 4), max_tokens // 2) で、額は指定できない。
    # 例: max_tokens = 48000・reasoning_budget_tokens = 40000 なら、取り置き 12,000 で予算は
    # 36,000 に切り下がる。ログに「thinking capped at 36000 of max_tokens 48000 ...」と出る。
    # したがって推論の予算を上げたいときは、max_tokens も一緒に上げる必要がある。
    # max_tokens との関係（2026-11-03、実機で Strata の control/serve/server.py と
    # サーバーのログで確認）:
    # - Strata は max_tokens から推論の予算を逆算し、残りを答え用に取り置く。
    #   取り置き = min(max(512, max_tokens // 4), max_tokens // 2)。
    #   max_tokens = 20000 なら取り置き 5,000・推論の予算 15,000。ログの
    #   「thinking: 1736 of budget 15000 (max 20000) tokens」で裏が取れている。
    # - 予算に達しても推論は途中で切られない。推論を閉じて答えを書かせる。
    # - max_tokens を送らないと、推論の予算は文脈長から逆算されて 151,851 になる。
    #   実機では 2,604 秒の時点で 48,764 トークン推論してまだ終わらず、予算いっぱいなら
    #   18.8 tokens/s で 8,077 秒（2.24 時間）の見込みだった。2分5秒の音声の1チャンクでこれである。
    # この設定を使うのは、max_tokens（答えの長さの上限）と推論の量を別々に決めたいときである。
    # 抽出にかかる時間はほぼ推論の長さで決まる（実測では出力 13,050 トークンのうち約 12,000 が推論）
    # ので、時間を詰めるならここが直接の手段になる。
    reasoning_budget_tokens: int = Field(default=0, ge=0)
    # 推論の量。OpenAI 互換 API の reasoning_effort に渡す。空文字なら送らない。
    # 有効な値は none / low / medium / high / xhigh（2026-11-04 実機で Strata のソースを確認）。
    # - 不正な値に対する 400 のメッセージは「use none, low, medium or high」と列挙するが、
    #   検証を行う frontend.py の EFFORT 表には "xhigh" がある。メッセージの文字列だけが古い。
    # - **high と xhigh は同一である**（どちらも内部の "xhigh" に写る）。
    # - **送らないことは xhigh を送ることと同じである。** chat_template.jinja が
    #   reasoning_effort|default('xhigh') としており、テンプレートの既定が xhigh だからである。
    #   プロファイル JSON と共有設定ファイルのどちらにも上書きは無いことを実機で確認した。
    # **既定は "medium"（2026-11-04 ユーザーが決定）。** 下の実測を見て、所要時間を取る判断をした。
    # 明示して送るのは、送らないとテンプレートの既定（xhigh）になり、挙動が設定から読めなくなるため。
    # 実測（2026-11-04、2分5秒の音声1チャンク。プロンプトは同一で、この値だけを変えた）:
    # - "xhigh": 推論 21,201 トークン・出力 22,408 トークン・409 秒。抽出6件。
    #   stated_reason 4/6、applies_when 2/6、pass_criterion 4/6。
    # - "medium": 推論 2,641 トークン・出力 3,622 トークン・70 秒。抽出6件。
    #   stated_reason 4/6、applies_when **1/6**、pass_criterion 4/6。約5.9倍速い。
    # medium が落とした applies_when は「参加者が募りたい場合」で、文字起こしに実在する。
    # CLAUDE.md 1章が暗黙知の中身として挙げる2点のうちの1つなので、軽い損失ではない。
    # なお、改訂プロンプトでは medium でも推測の混入（発話に無い語の当てはめ）は起きていない。
    # **max_tokens と一緒に決めること。** xhigh はこの入力に約21,200トークンの推論を要するので、
    # xhigh に戻す場合は max_tokens を 32000 以上にしないと推論が閉じられて品質が落ちる。
    # medium は推論 2,641 トークンなので、max_tokens = 20000（予算 15,000）で足りる。
    # 実測（2026-11-03、2分5秒の音声1チャンク。プロンプトは同一で、この値だけを変えた）:
    # - "xhigh"（このときのサーバーの既定でもあった）: 6件。文字起こしが壊れた発話（「感嘆符」が
    #   「簡単フォーイス」に化けた箇所）に対し stated_reason / applies_when / pass_criterion を null、
    #   reusable を false にした。所要 950 秒、出力 13,050 トークン（うち約 12,000 が推論）。
    # - "medium": 発話に無い「フォントサイズ」という語を当てはめた項目を作り、pass_criterion も作って
    #   reusable を true にした。CLAUDE.md 1章が禁じている推測の混入。所要は "xhigh" の約 1/8。
    # - "low": 同じ捏造に加え、反対の意味の発言を理由として誤って結び付けた。
    # - 思考なし（chat_template_kwargs の enable_thinking = false）: その項目を出さなかった（5件）。
    # 速さでは "medium" が有利だが、上記のとおり推測が混入したので既定にはしない。
    # プロンプト側で全項目に根拠制約を掛け、文字起こしが壊れた発話の扱いを明記した改訂の後に測り直す。
    reasoning_effort: str = "medium"
    # 文字起こしに VRAM を明け渡すための起動・停止コマンド。夜間バッチ（kijun batch run）が
    # Strata を使う手順の前後で実行する。空文字なら何もしない（Strata を手で起動・停止する運用）。
    start_command: str = ""
    stop_command: str = ""

    @field_validator("max_tokens")
    @classmethod
    def _答えの分が取り置かれる大きさであること(cls, v: int) -> int:
        if 0 < v < ANSWER_ROOM_MIN_CAP:
            raise ValueError(
                f"max_tokens は 0（上限なし）か {ANSWER_ROOM_MIN_CAP} 以上にしてください。"
                f"{ANSWER_ROOM_MIN_CAP} 未満だと Strata が答え用の取り置きを計算せず、"
                "推論が途中で切られて JSON が1文字も出ません（2026-11-03 実機で Strata のソースを確認）"
            )
        return v


class ExtractConfig(_Section):
    # 値は1つだが Literal のままにする（将来背面を足すときの型の受け皿）。
    provider: Literal["strata"] = "strata"
    chunk_minutes: int = 25
    overlap_minutes: int = 2
    # 入力トークン数の概算に使う係数。既定値は実測1点からの推測であり、`uv run kijun doctor` の
    # 実測値で置き換えるべきもの。
    # tokens_per_character: 実測 1,646 トークン ÷ 約3,141文字 ≒ 0.52 に少し余裕を持たせた値。
    # tokens_per_meeting_minute: 従来の 500（係数 1.5 に対応）を 0.55/1.5 の比で換算して切り下げた値。
    tokens_per_character: float = Field(default=0.55, gt=0)
    tokens_per_meeting_minute: int = Field(default=180, gt=0)
    # extracted_requests.model_name に付ける印。設定を変えた2回の抽出を、同じ会議の中で
    # 並べて残すために使う（CLAUDE.md 4-4 の「同じ Strata の2回の実行の比較」）。
    # これが無いと、2回目の kijun extract が1回目の行を削除してしまい、kijun compare が
    # 比較できない（extract/run.py の delete_requests は (会議ID, model_name) で消すため）。
    # 例: reasoning_effort = "medium" で流すときに run_label = "medium" を入れる。
    # 空文字なら model_name は strata.model のまま。request_id にも入るので英数字・._- に限る。
    run_label: str = Field(default="", pattern=r"^[A-Za-z0-9._-]*$")
    strata: StrataConfig = Field(default_factory=StrataConfig)

    def model_name(self) -> str:
        """extracted_requests.model_name に入れる名前。run_label があれば + でつなぐ。"""
        return f"{self.strata.model}+{self.run_label}" if self.run_label else self.strata.model


class MatchConfig(_Section):
    backend: str = "e5"
    model: str = "intfloat/multilingual-e5-large"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    similarity_threshold: float = 0.85
    device: str = "cuda"


class RetentionConfig(_Section):
    audio_days: int = 30


class DiscordConfig(_Section):
    # Discord のチャンネルID（数字の文字列）。空文字なら post と read は何もせず、設定が必要であることを出して終わる。
    review_channel_id: str = ""
    contradictions_channel_id: str = ""
    # 1回の post で投稿する新規候補の上限（CLAUDE.md 6章のレビュー所要時間の調整用）。
    max_candidates_per_run: int = 20
    # 1回の post で投稿する矛盾の可能性がある組の上限。
    max_contradictions_per_run: int = 5
    # 矛盾の候補とみなす類似度の下限。上限は match.similarity_threshold で、それ以上は重複として扱う。
    # この規則は設計者（Claude）が考案したもので、運用して見直す。
    contradiction_min_similarity: float = 0.70
    # 話者の対応付けで投稿する発言サンプルの件数（CLAUDE.md 4-6 は3件）。
    speaker_sample_count: int = 3
    # 1回の post で投稿する「文字起こしが読めなかった抽出結果」の上限（CLAUDE.md 7章の案A）。
    # 新規候補と同じ #review に出るので、レビューの所要時間（6章）を押し上げないよう上限を置く。
    # 新規候補より小さくしてあるのは、こちらはチェック項目にならないものであり、
    # レビューの本筋ではないためである。
    max_transcription_unclear_per_run: int = 5


class BatchConfig(_Section):
    # 各手順の記録（<日時>.json）を置く場所。
    log_dir: Path = Path("./data/batch_logs")
    # extract.strata.start_command を実行してから、Strata が応答するようになるまで待つ秒数。
    # 125B の MoE の読み込みには時間がかかる。待っても応答しなければ、その手順を失敗として扱う。
    strata_ready_timeout_sec: int = 1800


class Config(_Section):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    bundle: BundleConfig = Field(default_factory=BundleConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    diarize: DiarizeConfig = Field(default_factory=DiarizeConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)


class ConfigError(Exception):
    """設定ファイルが読めない、または値が不正なときの例外。"""


def load_config(path: Path | str | None = None) -> Config:
    """設定を読み込む。

    path を指定した場合は、そのファイルが無ければ ConfigError にする。
    path を省略した場合は、カレントディレクトリの config.toml を読み、
    無ければ既定値だけの設定を返す。
    """
    if path is None:
        target = DEFAULT_CONFIG_PATH
        if not target.exists():
            return Config()
    else:
        target = Path(path)
        if not target.exists():
            raise ConfigError(f"設定ファイルが見つかりません: {target}")
    try:
        with target.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"設定ファイルの TOML 構文が不正です: {target}: {e}") from e
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"設定ファイルの値が不正です: {target}\n{e}") from e
