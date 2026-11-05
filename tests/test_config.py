from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from pydantic import ValidationError

from kijun.config import Config, ConfigError, load_config

ROOT = Path(__file__).resolve().parent.parent


def test_既定値は_config_example_toml_と一致する():
    """config.py の既定値と config.example.toml がずれていないことを検査する。"""
    with (ROOT / "config.example.toml").open("rb") as f:
        raw = tomllib.load(f)
    from_example = Config.model_validate(raw)
    assert from_example == Config()


def test_計画書の主な既定値():
    c = Config()
    assert c.bundle.gap_minutes == 30
    assert c.bundle.require_same_setting is True
    assert c.retention.audio_days == 30
    assert c.settings.formats == ["f2f", "call"]


def test_Ollama_の既定値():
    # 2026-11-05 に設計者が置いた初期値(推奨・未確定)。実測で置き換える。
    o = Config().llm
    # localhost ではなく数字(Windows で ::1 に解決される問題を避ける)
    assert o.base_url == "http://127.0.0.1:11434"
    assert o.model == "qwen3.5:9b"
    assert o.num_ctx == 16384 and o.num_predict == 8192
    assert o.think is None
    assert o.keep_alive == "0"
    assert o.timeout_sec == 1800
    assert o.enforces_json_schema is True


def test_議事録の既定値():
    # 2026-11-06 に設計者が置いた初期値(推奨・未確定)
    c = Config()
    assert c.minutes.chunk_minutes == 15
    assert c.paths.minutes == Path("./data/minutes")
    assert c.discord.minutes_channel_id == ""
    assert c.batch.log_dir == Path("./data/batch_logs")


def test_チェックリスト時代の設定は拒否する():
    # 2026-11-06 に削除した。古い設定が残った config.toml は、綴り間違いと同じく読み込みで拒否する。
    for raw in [
        {"extract": {"chunk_minutes": 15}},
        {"extract": {"ollama": {"model": "x"}}},
        {"match": {"model": "x"}},
        {"discord": {"review_channel_id": "1"}},
        {"discord": {"contradictions_channel_id": "1"}},
        {"paths": {"review_logs": "./x"}},
    ]:
        with pytest.raises(ValidationError):
            Config.model_validate(raw)
    c = Config()
    assert not hasattr(c, "extract") and not hasattr(c, "match")


def test_config_toml_が無ければ既定値(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config() == Config()


def test_指定したファイルが無ければエラー(tmp_path):
    with pytest.raises(ConfigError, match="見つかりません"):
        load_config(tmp_path / "none.toml")


def test_書いた項目だけが上書きされる(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[bundle]\ngap_minutes = 10\n[paths]\ninbox = "C:/x/inbox"\n', encoding="utf-8")
    c = load_config(p)
    assert c.bundle.gap_minutes == 10
    assert c.bundle.require_same_setting is True  # 書かなかった項目は既定値
    assert c.paths.inbox == Path("C:/x/inbox")
    assert c.minutes.chunk_minutes == 15


def test_綴り間違いのキーは拒否する(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[bundle]\ngap_minute = 10\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_不正な値は拒否する(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[llm]\nnum_ctx = "大きい"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_toml_の構文エラー(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[bundle\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="構文"):
        load_config(p)


def _path_settings(model, prefix=""):
    """Config の中の Path 型の値を (設定名, 値) で列挙する。入れ子のセクションも辿る。"""
    from pathlib import Path

    from pydantic import BaseModel

    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, Path):
            yield f"{prefix}{name}", value
        elif isinstance(value, BaseModel):
            yield from _path_settings(value, f"{prefix}{name}.")


def test_cfg_フィクスチャの_Path_設定はすべて_tmp_path_の下(cfg, tmp_path):
    found = dict(_path_settings(cfg))
    # 列挙が空だと検査が素通りするので、既知の設定が見えていることを確かめる
    assert {"paths.inbox", "paths.db", "batch.log_dir"} <= set(found)
    outside = {k: str(v) for k, v in found.items() if not v.resolve().is_relative_to(tmp_path.resolve())}
    assert outside == {}, f"一時ディレクトリの外を指す設定がある（テストが利用者のデータを汚す）: {outside}"


def test_num_predict_は_1024_以上でなければ拒否する():
    # 推論(think)も数えるので、小さすぎると推論の途中で切れて JSON が1文字も出ない。
    for ok in [1024, 8192]:
        assert Config.model_validate({"llm": {"num_predict": ok}})
    for bad in [1, 512, 1023]:
        with pytest.raises(ValidationError, match="1024"):
            Config.model_validate({"llm": {"num_predict": bad}})


def test_num_predict_は_num_ctx_より小さくなければ拒否する():
    with pytest.raises(ValidationError, match="num_predict"):
        Config.model_validate({"llm": {"num_ctx": 4096, "num_predict": 4096}})


def test_chunk_minutes_は_1_以上():
    for bad in [0, -1]:
        with pytest.raises(ValidationError):
            Config.model_validate({"minutes": {"chunk_minutes": bad}})


def test_入力が文脈長に収まらない設定は拒否する():
    # Ollama は入力が num_ctx を超えると、エラーを出さずに先頭を捨てる。走らせる前に拒否する。
    with pytest.raises(ValidationError, match="収まりません") as e:
        Config.model_validate({"minutes": {"chunk_minutes": 60}})  # 60 × 180 + system + 8192 > 16384
    assert "num_ctx" in str(e.value)
    # num_ctx を上げれば通る
    assert Config.model_validate({"minutes": {"chunk_minutes": 60}, "llm": {"num_ctx": 32768}})
    # num_predict を下げても通る
    assert Config.model_validate({"minutes": {"chunk_minutes": 60}, "llm": {"num_predict": 2048}})


def test_既定の設定は文脈長にある程度の余裕を持って収まる():
    from kijun.minutes.prompt import CHUNK_SYSTEM_PROMPT

    c = Config()
    need = (
        c.minutes.chunk_minutes * c.minutes.tokens_per_meeting_minute
        + int(len(CHUNK_SYSTEM_PROMPT) * c.minutes.tokens_per_character)
        + c.llm.num_predict
    )
    assert need <= c.llm.num_ctx


def test_batch_の既定値とゲーム検出の設定():
    b = Config().batch
    assert b.task_time == "23:30"
    assert b.defer_if_running == []  # 空なら検出しない
    assert b.defer_check_interval_min == 30 and b.defer_max_hours == 4


def test_task_time_は_HH_MM_の形式だけ許す():
    assert Config.model_validate({"batch": {"task_time": "03:05"}})
    assert Config.model_validate({"batch": {"task_time": "9:30"}})
    for bad in ["25:00", "23:60", "2330", "", "23:30:00"]:
        with pytest.raises(ValidationError, match="task_time"):
            Config.model_validate({"batch": {"task_time": bad}})


def test_ゲーム検出の設定の範囲():
    with pytest.raises(ValidationError):
        Config.model_validate({"batch": {"defer_check_interval_min": 0}})
    with pytest.raises(ValidationError):
        Config.model_validate({"batch": {"defer_max_hours": -1}})
    with pytest.raises(ValidationError, match="空の名前"):
        Config.model_validate({"batch": {"defer_if_running": ["Game.exe", " "]}})
    assert Config.model_validate({"batch": {"defer_max_hours": 0}})  # 0 は待たずに飛ばす


def test_文字起こしの幻聴対策の既定値():
    t = Config().transcribe
    assert t.vad_filter is False
    assert t.condition_on_previous_text is False
    assert t.hallucination_silence_threshold == 2.0
    assert (t.drop_max_speaker_overlap, t.drop_max_avg_logprob) == (0.3, -0.5)


@pytest.mark.parametrize(
    "key,value",
    [
        ("drop_max_speaker_overlap", -0.1),
        ("drop_max_speaker_overlap", 1.1),
        ("hallucination_silence_threshold", -1),
    ],
)
def test_文字起こしの幻聴対策の範囲外の値は拒否する(key, value):
    from pydantic import ValidationError

    from kijun.config import TranscribeConfig

    with pytest.raises(ValidationError):
        TranscribeConfig(**{key: value})


def test_幻聴対策の境界値は受け付ける():
    from kijun.config import TranscribeConfig

    t = TranscribeConfig(drop_max_speaker_overlap=0, hallucination_silence_threshold=0)
    assert t.drop_max_speaker_overlap == 0 and t.hallucination_silence_threshold == 0
