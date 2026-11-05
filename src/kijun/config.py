"""設定ファイル（config.toml）の読み込みと既定値。

読み込み順は次のとおり。
1. config.toml に書かれた値
2. 書かれていない項目は、このファイルの既定値

既定値は config.example.toml と一致させる（tests/test_config.py が一致を検査する）。
秘密情報（pyannote のモデル取得に使う HF_TOKEN、Discord bot の DISCORD_BOT_TOKEN）は
設定ファイルに書かず、環境変数で渡す。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("config.toml")


class _Section(BaseModel):
    """設定の1区画。未知のキーは綴り間違いの可能性が高いので拒否する。"""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    inbox: Path = Path("./data/inbox")
    processed: Path = Path("./data/processed")
    transcripts: Path = Path("./data/transcripts")
    db: Path = Path("./data/kijun.duckdb")
    # kijun discord post / read の実行記録（<種類>_<日時>.json）を置く場所。
    # DB には「投稿したこと」と「処理済みになった時刻」しか残らず、どのリアクションが
    # 付いたかは残らない。6章の指標（レビュー所要時間、修正せず承認した割合）と、
    # 7章の「文字起こしが読めなかった投稿に ✅ と ❌ のどちらが付いたか」を後から数えるために、
    # 実行ごとの報告をそのまま書き出す。
    review_logs: Path = Path("./data/review_logs")


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
    # faster-whisper の Silero VAD。既定は False（2026-11-05 に True から変更した。設計者の初期値）。
    # 理由：iPhone で録音した Discord の通話（36分）で、True だと発話の約 2/3 を非音声と判定して捨てた
    # （同じ音声・同じモデルで 2,967 文字 = 82 文字/分。False では 9,348 文字 = 260 文字/分）。
    # 1本の通話での観測であり、対面の録音で False が良いかは再測定していない（CLAUDE.md 7章）。
    vad_filter: bool = False
    # 直前の区間の本文を次の区間の文脈に使うか。既定 False。VAD を切ると無音部分で幻聴が出やすく、
    # True だと幻聴が次の区間へ連鎖する。設計者の初期値で、1本の通話での観測に基づく。
    condition_on_previous_text: bool = False
    # 単語の時刻から、この秒数以上の無音を挟む区間を幻聴とみなして飛ばす（faster-whisper 1.2.1 の
    # hallucination_silence_threshold。word_timestamps=True が必要なので、> 0 のとき自動で有効にする）。
    # 0 なら使わない。2.0 は設計者の初期値：36分の通話で、VAD なしだと 鎧/猫/スイカ の無意味な
    # 約12区間などが出たが、2.0 でそれらは消えた（末尾の無音の 3 区間だけ残った）。
    hallucination_silence_threshold: float = Field(default=2.0, ge=0)
    # 幻聴の除去規則。区間の「話者区間との重なりの割合」が drop_max_speaker_overlap 未満、かつ
    # avg_logprob が drop_max_avg_logprob 未満なら除く。除いた区間は文字起こしの JSON の
    # dropped_segments に残す（黙って消さない）。drop_max_speaker_overlap = 0 なら除去しない。
    # 0.3 / -0.5 は設計者の初期値：上の 36分の通話で、末尾の幻聴 3 区間（重なり 0.00、
    # avg_logprob は -0.5 未満）と意味の無い断片 2 件だけを除き、pyannote が拾えなかったが Whisper が
    # 確信していた実発話（重なり 0.00、avg_logprob -0.32 や -0.08）は残した。広い測定ではない。
    # no_speech_prob は実発話でも 0.5〜0.8 になり使えなかったので使わない。
    drop_max_speaker_overlap: float = Field(default=0.3, ge=0, le=1)
    drop_max_avg_logprob: float = -0.5
    # faster-whisper に渡す Silero VAD のパラメータ。vad_filter が true のときだけ使う
    # （2026-11-05 以降の既定は vad_filter = false なので、既定では使われない）。
    # **これを設定値にしたのは、既定のままでは区間が切れないことが実機で分かったためである**
    # （2026-11-04。34.2分の通話で 564.8 秒の区間が1つ出て、中身は3文字だった）。
    # faster-whisper は max_speech_duration_s を既定で無制限にする。そのうえ
    # min_silence_duration_ms（既定2000）より短い無音では区間を切らない。PC のスピーカー出力を
    # iPhone で録る方法では室内の音が常に入るので、2秒続く無音が現れず、区間が延々とつながる。
    #
    # vad_max_speech_duration_sec は、Silero が1つの発話区間として許す長さの上限。
    # 0 は「上限なし」（faster-whisper の既定と同じ）。既定の 30.0 は max_utterance_sec に合わせた値で、
    # 実測に基づくものではない。Silero は上限に達する前の最後の無音で切るので、文の途中で切るとは限らない。
    # **区間の長さを抑えるのはこの設定であり、max_utterance_sec ではない**（下のコメント参照）。
    vad_max_speech_duration_sec: float = 30.0
    # 音声だと判定する確率の下限。上げると、小さい音や雑音を音声と見なさなくなる。
    vad_threshold: float = 0.5
    # 区間を切る無音の長さ（ミリ秒）。faster-whisper の既定は 2000。
    vad_min_silence_duration_ms: int = 2000
    # 区間の前後に付ける余白（ミリ秒）。faster-whisper の既定は 400。
    vad_speech_pad_ms: int = 400
    # 同一話者の連続する区間を1つの発話に結合する間隔の上限（秒）。
    # 2.0 は設計者が 2 本の録音で測った値で、ユーザーが 2026-11-05 に承認した。word_timestamps=True
    # （hallucination_silence_threshold > 0 で有効になる）にすると、Whisper の各区間が最後の単語で
    # 終わるので、1人の話の中の間（ま）が 0.4〜2.6 秒の間隔として現れ、0.4 では細かく割れる。
    # 実測: 2分の独話が 0.4 で 23 発話、2.0 で 7 発話。36分の通話は 255 発話から 136 発話。
    min_utterance_sec: float = 2.0
    # 同一話者の連続する区間を**結合する**ときの、結合後の長さの上限（秒）。0 は「上限なし」。
    # **1つの文字起こし区間そのものがこれより長い場合、分割はしない**（merge.py の説明を参照。
    # 分割の手段は merge.py に無い）。vad_filter = true のときは、区間自体の長さを
    # vad_max_speech_duration_sec で抑える。vad_filter = false のときは、Whisper が音声を 30 秒の窓で
    # 処理するので区間は 30 秒を超えない（2026-11-05 の 36分の通話で最長 26.5 秒）。
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


class OllamaConfig(_Section):
    """抽出と、抽出以外の LLM の用途（返信の振り分け・重要度の判定・話者の解釈）が使う Ollama の設定。

    すべて Ollama のネイティブ API（POST /api/chat）のリクエストごとに送る。OpenAI 互換の
    /v1/chat/completions を使わないのは、num_ctx と keep_alive をリクエストごとに渡せないためである。
    既定値は 2026-11-05 に設計者（Claude）が置いた初期値で、実測に基づかない（推奨・未確定）。
    `uv run kijun doctor` と `scripts/measure_extraction.py` で測って置き換える。
    """

    # localhost ではなく数字で書く。Windows では localhost が IPv6 の ::1 に解決されることがあり、
    # サーバーが IPv4 だけで待ち受けていると接続拒否になる（CLAUDE.md 4-4 で踏んだ問題）。
    base_url: str = "http://127.0.0.1:11434"
    # 推奨・未確定。12GB の VRAM に収まる 9B〜14B から測定で決める。モデルが存在するかは
    # `ollama list` で確認する。名前に ':' が入る（モデル名:タグ）。
    model: str = "qwen3.5:9b"
    # 文脈長（入力 + 出力）。リクエストごとに options.num_ctx で送る。送らないと Ollama の既定
    # （版により 4096 など）になり、入力が黙って先頭から切り捨てられる。
    # 大きくすると KV キャッシュが増えて VRAM を使う。GPU に収まらなくなったかは doctor の検査3で見る。
    num_ctx: int = Field(default=16384, ge=2048)
    # 出力の上限（推論を含む）。options.num_predict で送る。1024 未満は拒否する。
    # 推論（think）を有効にしたモデルは、答えの前に推論を書く。それも数えるので、小さすぎると
    # 推論の途中で切れて JSON が1文字も出ない。
    # 1チャンクの所要時間の最悪値は、num_predict ÷ 観測された最も遅い生成速度で決まる。
    num_predict: int = Field(default=8192, ge=1024)
    # Qwen の推論モード。None なら think を送らない（モデルの既定に任せる）。
    # TOML には null が無いので、config.toml では「書かない」ことで None を表す。
    think: bool | None = None
    # 応答のたびにモデルを VRAM に置いておく時間。"0" なら応答のたびに下ろす。
    # "0" にする理由は、faster-whisper・pyannote と LLM を同時に VRAM に載せないためである
    # （docs/implementation-plan.md 原則2-2）。"5m" のような値にすると、文字起こしの手順の時点で
    # LLM が VRAM に残り得る。
    keep_alive: str = Field(default="0", min_length=1)
    timeout_sec: int = Field(default=1800, gt=0)
    # format にスキーマを渡すか。渡すと Ollama が出力をスキーマに適合させる。
    # 出力は必ず repair.py を通してから検証するので、真でも修復は働く。
    # 適合するかは doctor の検査2で確認する。適合しなければ false にする。
    enforces_json_schema: bool = True

    @model_validator(mode="after")
    def _出力の上限が文脈長に収まること(self) -> "OllamaConfig":
        if self.num_predict >= self.num_ctx:
            raise ValueError(
                f"num_predict（{self.num_predict}）は num_ctx（{self.num_ctx}）より小さくしてください。"
                "出力の上限が文脈長以上だと、入力を入れる余地がありません"
            )
        return self


class ExtractConfig(_Section):
    # 値は1つだが Literal のままにする（将来背面を足すときの型の受け皿）。
    provider: Literal["ollama"] = "ollama"
    # 1リクエストに入れる会議の長さ（分）。推奨・未確定（2026-11-05 設計者の初期値）。
    # Ollama は num_ctx（既定 16384）の中に入力と出力を入れるので、Strata のときの 25 分より小さくした。
    # 目安: 12 分 × tokens_per_meeting_minute 180 ≒ 2,160 トークン + システムプロンプト + num_predict 8192
    # が num_ctx に収まる。収まらない設定は読み込みで拒否する。
    chunk_minutes: int = Field(default=12, gt=0)
    overlap_minutes: int = Field(default=1, ge=0)
    # 入力トークン数の概算に使う係数。既定値は Strata（Qwen3.8-Flash-Next）での実測1点からの推測であり、
    # Ollama で使うモデルのトークナイザには当てはまらない。`uv run kijun doctor` の検査4・5の実測値で
    # 置き換えるべきもの。
    # tokens_per_character: Strata での実測 1,646 トークン ÷ 約3,141文字 ≒ 0.52 に少し余裕を持たせた値。
    # tokens_per_meeting_minute: 従来の 500（係数 1.5 に対応）を 0.55/1.5 の比で換算して切り下げた値。
    tokens_per_character: float = Field(default=0.55, gt=0)
    tokens_per_meeting_minute: int = Field(default=180, gt=0)
    # extracted_requests.model_name に付ける印。設定を変えた2回の抽出を、同じ会議の中で
    # 並べて残すために使う（CLAUDE.md 4-4 の「同じ背面の設定を変えた2回の実行の比較」）。
    # これが無いと、2回目の kijun extract が1回目の行を削除してしまい、kijun compare が
    # 比較できない（extract/run.py の delete_requests は (会議ID, model_name) で消すため）。
    # 例: think = false で流すときに run_label = "nothink" を入れる。
    # 空文字なら model_name は ollama.model のまま。request_id にも入るので英数字・._- に限る。
    run_label: str = Field(default="", pattern=r"^[A-Za-z0-9._-]*$")
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)

    @model_validator(mode="after")
    def _分割と文脈長の整合(self) -> "ExtractConfig":
        if self.chunk_minutes <= self.overlap_minutes:
            raise ValueError(
                f"chunk_minutes（{self.chunk_minutes}）は overlap_minutes（{self.overlap_minutes}）より"
                "大きくしてください。そうでないとチャンクが進みません"
            )
        # 入力が num_ctx を超えると、Ollama はエラーを出さずに先頭を捨てる。捨てられた発話が抽出結果から
        # 静かに消えるので、走らせる前に拒否する（docs/implementation-plan.md 原則2-4「失うより止める」）。
        from kijun.extract.prompt import SYSTEM_PROMPT  # 循環 import を避けるため、使うときに読み込む

        o = self.ollama
        system = int(len(SYSTEM_PROMPT) * self.tokens_per_character)
        user = self.chunk_minutes * self.tokens_per_meeting_minute
        need = user + system + o.num_predict
        if need > o.num_ctx:
            raise ValueError(
                f"入力が文脈長に収まりません。chunk_minutes {self.chunk_minutes} 分 × "
                f"tokens_per_meeting_minute {self.tokens_per_meeting_minute} = 約{user} トークン、"
                f"システムプロンプトの概算 約{system} トークン、出力の上限 num_predict {o.num_predict} の合計は"
                f"{need} で、ollama.num_ctx（{o.num_ctx}）を超えます。"
                "chunk_minutes か num_predict を下げるか、num_ctx を上げてください"
            )
        return self

    def model_name(self) -> str:
        """extracted_requests.model_name に入れる名前。run_label があれば + でつなぐ。

        Ollama のモデル名には ':' が入る（qwen3.5:9b）。model_name は ':' を残したまま DB に入れる。
        request_id には使える文字が決まっているので、そちらでは extract/run.py の request_id_model が
        置き換える。
        """
        model = self.ollama.model
        return f"{model}+{self.run_label}" if self.run_label else model


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


_TASK_TIME = re.compile(r"(?:[01]?\d|2[0-3]):[0-5]\d")


class BatchConfig(_Section):
    # 各手順の記録（<日時>.json）を置く場所。
    log_dir: Path = Path("./data/batch_logs")
    # 夜間バッチの開始時刻（HH:MM）。`kijun batch install-task` が --time を省略されたときに使う。
    task_time: str = "23:30"
    # この中のどれかのプロセスが動いている間は、GPU を使う手順を始めない。
    # 名前はタスクマネージャーの「詳細」タブに出る実行ファイル名（大文字小文字は区別しない）。
    # 推奨・未確定：ユーザーが自分の環境で実行ファイル名を確認して書く。空なら検出を行わない。
    defer_if_running: list[str] = Field(default_factory=list)
    # 延期中に再確認する間隔（分）。
    defer_check_interval_min: int = Field(default=30, gt=0)
    # 延期する時間の上限（時間。バッチ全体の合計）。超えたら、その夜のバッチは GPU を使う手順を
    # 飛ばして終わる。飛ばした会議は翌晩に未処理として拾われる。0 なら待たずに飛ばす。
    defer_max_hours: float = Field(default=4, ge=0)

    @field_validator("task_time")
    @classmethod
    def _時刻の形式(cls, v: str) -> str:
        if _TASK_TIME.fullmatch(v) is None:
            raise ValueError(f"task_time は HH:MM の形式（00:00 から 23:59）で書いてください: {v}")
        return v

    @field_validator("defer_if_running")
    @classmethod
    def _空の名前を拒否(cls, v: list[str]) -> list[str]:
        # 空文字は、どのプロセスにも一致しない名前として黙って残るより、書き間違いとして拒否する。
        if any(not name.strip() for name in v):
            raise ValueError("defer_if_running に空の名前があります")
        return v


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
