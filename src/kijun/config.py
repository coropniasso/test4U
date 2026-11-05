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
    # 議事録の Markdown（<会議ID>.md）を置く場所。
    minutes: Path = Path("./data/minutes")


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
    """議事録の生成が使う Ollama の設定（設定ファイルの [llm]）。

    すべて Ollama のネイティブ API（POST /api/chat）のリクエストごとに送る。OpenAI 互換の
    /v1/chat/completions を使わないのは、num_ctx と keep_alive をリクエストごとに渡せないためである。
    既定値は 2026-11-05 に設計者（Claude）が置いた初期値で、実測に基づかない（推奨・未確定）。
    `uv run kijun doctor` で測って置き換える。
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
    # （docs/history/implementation-plan.md 原則2-2）。"5m" のような値にすると、文字起こしの手順の時点で
    # LLM が VRAM に残り得る。
    keep_alive: str = Field(default="0", min_length=1)
    timeout_sec: int = Field(default=1800, gt=0)
    # format にスキーマを渡すか。渡すと Ollama が出力をスキーマに適合させる。
    # 出力は必ず llm/repair.py を通してから検証するので、真でも修復は働く。
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


class MinutesConfig(_Section):
    """議事録の生成（minutes/）の設定。

    発話を chunk_minutes 分ごとのチャンク（重なり 0）に分け、チャンクごとに Ollama へ投げる。
    既定値は 2026-11-06 に設計者（Claude）が置いた初期値で、実測に基づかない（推奨・未確定）。
    """

    # 1リクエストに入れる会議の長さ（分）。Ollama は llm.num_ctx の中に入力と出力を入れるので、
    # 入力（chunk_minutes × tokens_per_meeting_minute + システムプロンプト）と num_predict の合計が
    # num_ctx に収まらない設定は、読み込みで拒否する（Config の検証）。
    chunk_minutes: int = Field(default=15, gt=0)
    # 入力トークン数の概算に使う係数。抽出のときに Strata で測った1点からの推測で、Ollama のモデルの
    # トークナイザには当てはまらない。`uv run kijun doctor` の検査4・5が実測値を出すので置き換える。
    tokens_per_character: float = Field(default=0.55, gt=0)
    tokens_per_meeting_minute: int = Field(default=180, gt=0)


class RetentionConfig(_Section):
    audio_days: int = 30


class DiscordConfig(_Section):
    # 議事録を投稿するチャンネルのID（数字の文字列）。空文字なら `kijun discord post` は何もせず、
    # 設定が必要であることを出して正常終了する。
    # チャンネルは、プロジェクトのサーバーではなくユーザー個人のサーバーに作る（CLAUDE.md 2章）。
    minutes_channel_id: str = ""


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
    llm: OllamaConfig = Field(default_factory=OllamaConfig)
    minutes: MinutesConfig = Field(default_factory=MinutesConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)

    @model_validator(mode="after")
    def _チャンクが文脈長に収まること(self) -> "Config":
        # 入力が num_ctx を超えると、Ollama はエラーを出さずに先頭を捨てる。捨てられた発話が議事録から
        # 静かに消えるので、走らせる前に拒否する（docs/history/implementation-plan.md 原則2-4「失うより止める」）。
        from kijun.minutes.prompt import CHUNK_SYSTEM_PROMPT  # 循環 import を避けるため、使うときに読み込む

        m, o = self.minutes, self.llm
        system = int(len(CHUNK_SYSTEM_PROMPT) * m.tokens_per_character)
        user = m.chunk_minutes * m.tokens_per_meeting_minute
        need = user + system + o.num_predict
        if need > o.num_ctx:
            raise ValueError(
                f"入力が文脈長に収まりません。minutes.chunk_minutes {m.chunk_minutes} 分 × "
                f"tokens_per_meeting_minute {m.tokens_per_meeting_minute} = 約{user} トークン、"
                f"システムプロンプトの概算 約{system} トークン、出力の上限 llm.num_predict {o.num_predict} の合計は"
                f"{need} で、llm.num_ctx（{o.num_ctx}）を超えます。"
                "chunk_minutes か num_predict を下げるか、num_ctx を上げてください"
            )
        return self


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
