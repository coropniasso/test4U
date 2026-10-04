from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from kijun.config import SettingsConfig
from kijun.ingest.filename import FilenameError, parse_filename, scan_inbox

SETTINGS = SettingsConfig()


def test_正常なファイル名():
    p = parse_filename("20260920_1930_f2f_teirei.m4a", SETTINGS)
    assert p.recorded_at == datetime(2026, 9, 20, 19, 30)
    assert p.setting_format == "f2f"
    assert p.setting_kind == "teirei"
    assert p.part is None
    assert p.ext == "m4a"
    assert p.kind_known is True
    assert p.date.isoformat() == "2026-09-20"
    assert p.time.isoformat() == "19:30:00"
    assert p.path == Path("20260920_1930_f2f_teirei.m4a")


def test_通話の形式():
    assert parse_filename("20260921_0800_call_sagyou.m4a", SETTINGS).setting_format == "call"


def test_未知の種類は警告を出して受け入れる():
    warnings: list[str] = []
    p = parse_filename("20260920_1930_f2f_kenshu.m4a", SETTINGS, warn=warnings.append)
    assert p.setting_kind == "kenshu"  # そのまま記録する
    assert p.kind_known is False
    assert len(warnings) == 1
    assert "kenshu" in warnings[0]


def test_未知の種類の警告は既定で標準エラーに出る(capsys):
    parse_filename("20260920_1930_f2f_kenshu.m4a", SETTINGS)
    captured = capsys.readouterr()
    assert "kenshu" in captured.err
    assert captured.out == ""


def test_既知の種類では警告を出さない():
    warnings: list[str] = []
    parse_filename("20260920_1930_f2f_teirei.m4a", SETTINGS, warn=warnings.append)
    assert warnings == []


def test_未知の形式は失敗させる():
    with pytest.raises(FilenameError, match="zoom"):
        parse_filename("20260920_1930_zoom_teirei.m4a", SETTINGS)


def test_連番付き():
    p = parse_filename("20260920_1930_f2f_teirei_2.m4a", SETTINGS)
    assert p.part == 2
    assert p.setting_kind == "teirei"


@pytest.mark.parametrize(
    "name",
    [
        "voice memo.m4a",  # 書式に合わない
        "20260920_1930_f2f.m4a",  # 種類が無い
        "20260920_1930_f2f_teirei.txt",  # 拡張子が違う
        "2026920_1930_f2f_teirei.m4a",  # 日付が7桁
        "20260920_193_f2f_teirei.m4a",  # 時刻が3桁
        "20260920_1930_F2F_teirei.m4a",  # 大文字は書式に含めない
        "20260920_1930_f2f_定例.m4a",  # 日本語
        "20261340_1930_f2f_teirei.m4a",  # 存在しない日付
        "20260920_2560_f2f_teirei.m4a",  # 存在しない時刻
        "",
    ],
)
def test_解析できないファイル名(name):
    with pytest.raises(FilenameError):
        parse_filename(name, SETTINGS)


def test_scan_inbox_は解析できないファイルに何もしない(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    good = inbox / "20260920_1930_f2f_teirei.m4a"
    bad = inbox / "Recording 12.m4a"
    bad_format = inbox / "20260920_2000_zoom_teirei.m4a"
    hidden = inbox / ".DS_Store"
    for p in (good, bad, bad_format, hidden):
        p.write_bytes(b"x")
    (inbox / "subdir").mkdir()

    parsed, failed = scan_inbox(inbox, SETTINGS, warn=lambda m: None)

    assert [p.path.name for p in parsed] == ["20260920_1930_f2f_teirei.m4a"]
    assert {p.name for p, _ in failed} == {"Recording 12.m4a", "20260920_2000_zoom_teirei.m4a"}
    # 削除も移動もしていない
    assert good.exists() and bad.exists() and bad_format.exists()
    assert sorted(p.name for p in inbox.iterdir()) == sorted(
        [".DS_Store", "20260920_1930_f2f_teirei.m4a", "20260920_2000_zoom_teirei.m4a", "Recording 12.m4a", "subdir"]
    )


def test_scan_inbox_は_inbox_が無くても落ちない(tmp_path: Path):
    assert scan_inbox(tmp_path / "none", SETTINGS) == ([], [])
