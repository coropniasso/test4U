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
    assert c.extract.ollama.keep_alive == 0
    assert c.extract.claude.model == "claude-sonnet-5-5"
    assert c.extract.claude.effort == "low"
    assert c.settings.formats == ["f2f", "call"]


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
