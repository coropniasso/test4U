from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

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
    assert c.extract.chunk_minutes == 25
    assert c.extract.overlap_minutes == 2
    assert c.match.similarity_threshold == 0.85
    assert c.retention.audio_days == 30
    assert c.extract.provider == "strata"
    assert c.extract.strata.base_url == "http://localhost:8080"
    assert c.extract.strata.chat_path == "/v1/chat/completions"
    assert c.extract.strata.model == "qwen3.8-flash-next"
    assert c.extract.strata.timeout_sec == 3600
    # 既定は「保証しない」「未確認」の安全側。kijun doctor の実測で書き換える。
    assert c.extract.strata.enforces_json_schema is False
    assert c.extract.strata.max_input_tokens == 0
    assert c.extract.strata.start_command == "" and c.extract.strata.stop_command == ""
    assert c.settings.formats == ["f2f", "call"]


def test_第2弾の設定の既定値():
    c = Config()
    assert c.discord.review_channel_id == "" and c.discord.contradictions_channel_id == ""
    assert c.discord.max_candidates_per_run == 20
    assert c.discord.max_contradictions_per_run == 5
    assert c.discord.contradiction_min_similarity == 0.70
    assert c.discord.speaker_sample_count == 3
    assert c.batch.log_dir == Path("./data/batch_logs")
    # 矛盾の下限は、重複として統合される閾値より小さくなければ範囲が空になる
    assert c.discord.contradiction_min_similarity < c.match.similarity_threshold


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
    assert c.extract.chunk_minutes == 25


def test_綴り間違いのキーは拒否する(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[bundle]\ngap_minute = 10\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_不正な値は拒否する(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[extract]\nprovider = "gpt"\n', encoding="utf-8")
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


def test_run_label_は英数字と記号3種だけ許す(tmp_path):
    # request_id（"<会議ID>#<model_name>#<番号>"）とログのファイル名に入るため。
    import pytest
    from pydantic import ValidationError

    from kijun.config import Config

    assert Config.model_validate({"extract": {"run_label": "iq3_xxs-medium.1"}})
    for bad in ["medium 1", "日本語", "a/b", "a#b", "a+b"]:
        with pytest.raises(ValidationError):
            Config.model_validate({"extract": {"run_label": bad}})
