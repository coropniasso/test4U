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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class DiarizeConfig(_Section):
    backend: str = "pyannote"
    model: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    # TOML には null が無いので、0 を「指定なし（pyannote が推定する）」として扱う。
    num_speakers: int = 0


class StrataConfig(_Section):
    base_url: str = "http://localhost:8080"
    chat_path: str = "/v1/chat/completions"
    model: str = "qwen3.8-flash-next"
    timeout_sec: int = 3600
    enforces_json_schema: bool = False
    # 0 は「未確認」を表す（TOML に null が無いため。diarize.num_speakers と同じ扱い）。
    # 未確認のときは chunk_minutes の妥当性を検査しない。
    max_input_tokens: int = 0
    # 文字起こしに VRAM を明け渡すための起動・停止コマンド。夜間バッチ（kijun batch run）が
    # Strata を使う手順の前後で実行する。空文字なら何もしない（Strata を手で起動・停止する運用）。
    start_command: str = ""
    stop_command: str = ""


class ExtractConfig(_Section):
    # 値は1つだが Literal のままにする（将来背面を足すときの型の受け皿）。
    provider: Literal["strata"] = "strata"
    chunk_minutes: int = 25
    overlap_minutes: int = 2
    strata: StrataConfig = Field(default_factory=StrataConfig)


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
