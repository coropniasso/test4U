"""タスクスケジューラの XML(仕様 3-3、テスト 10)。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

import pytest
from typer.testing import CliRunner

from kijun.batch.task_xml import build_task_xml, parse_time, schtasks_command
from kijun.cli import app

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def parse(xml: bytes) -> ET.Element:
    return ET.fromstring(xml)  # 整形式でなければ ParseError


def text(root: ET.Element, path: str) -> str:
    node = root.find(path, NS)
    assert node is not None, path
    return node.text


def test_整形式の_XML_になる():
    xml = build_task_xml("23:30", "uv", "run kijun batch run", "C:/work/test4U", date(2026, 11, 3))
    root = parse(xml)
    assert root.tag == "{http://schemas.microsoft.com/windows/2004/02/mit/task}Task"
    assert xml.startswith(b"\xff\xfe") or xml.startswith(b"\xfe\xff")  # UTF-16(BOM 付き)。schtasks が読める形


def test_WakeToRun_と_StartWhenAvailable_が_true():
    root = parse(build_task_xml("23:30", "uv", "run kijun batch run", "C:/work"))
    assert text(root, ".//t:Settings/t:WakeToRun") == "true"
    assert text(root, ".//t:Settings/t:StartWhenAvailable") == "true"


def test_毎日の指定時刻で実行する():
    root = parse(build_task_xml("07:05", "uv", "run kijun batch run", "C:/work", date(2026, 11, 3)))
    assert text(root, ".//t:CalendarTrigger/t:StartBoundary") == "2026-11-03T07:05:00"
    assert text(root, ".//t:ScheduleByDay/t:DaysInterval") == "1"


def test_実行するコマンドと作業ディレクトリ():
    root = parse(build_task_xml("23:30", "C:/Tools/uv.exe", 'run kijun --config "C:/a b/c.toml" batch run', "C:/work dir"))
    assert text(root, ".//t:Exec/t:Command") == "C:/Tools/uv.exe"
    assert text(root, ".//t:Exec/t:Arguments") == 'run kijun --config "C:/a b/c.toml" batch run'
    assert text(root, ".//t:Exec/t:WorkingDirectory") == "C:/work dir"  # config.toml と ./data の基準になる


def test_特殊文字は_XML_として正しくエスケープされる():
    root = parse(build_task_xml("23:30", "uv", 'run kijun --config "a&b<c>.toml" batch run', "C:/x&y"))
    assert text(root, ".//t:Exec/t:Arguments") == 'run kijun --config "a&b<c>.toml" batch run'


def test_前夜のバッチが動いている間は重ねて起動しない():
    root = parse(build_task_xml("23:30", "uv", "run kijun batch run", "C:/work"))
    assert text(root, ".//t:MultipleInstancesPolicy") == "IgnoreNew"
    assert text(root, ".//t:ExecutionTimeLimit") == "PT0S"


@pytest.mark.parametrize("value", ["24:00", "9", "23:60", "abc", "", "23:5"])
def test_不正な時刻は拒否する(value):
    with pytest.raises(ValueError):
        build_task_xml(value, "uv", "x", "C:/")
    with pytest.raises(ValueError):
        parse_time(value)


def test_時刻の解釈():
    assert parse_time("23:30") == (23, 30) and parse_time("0:05") == (0, 5)


def test_登録コマンドは_schtasks_create_xml():
    assert schtasks_command("C:/a/task.xml") == 'schtasks /create /tn "kijun nightly batch" /xml "C:/a/task.xml"'


# --- コマンド ---------------------------------------------------------------------------------------


def test_install_task_は_XML_を保存し_登録コマンドを案内する(tmp_path):
    out = tmp_path / "task.xml"
    result = CliRunner().invoke(app, ["batch", "install-task", "--time", "22:15", "--out", str(out)])
    assert result.exit_code == 0, result.output
    root = parse(out.read_bytes())
    assert text(root, ".//t:Settings/t:WakeToRun") == "true"
    assert text(root, ".//t:Settings/t:StartWhenAvailable") == "true"
    assert text(root, ".//t:CalendarTrigger/t:StartBoundary").endswith("T22:15:00")
    assert "batch run" in text(root, ".//t:Exec/t:Arguments")
    assert "schtasks /create" in result.output and str(out) in result.output


def test_install_task_は_config_を引き継ぐ(tmp_path):
    cfg = tmp_path / "my config.toml"
    cfg.write_text("", encoding="utf-8")
    out = tmp_path / "task.xml"
    result = CliRunner().invoke(app, ["--config", str(cfg), "batch", "install-task", "--out", str(out)])
    assert result.exit_code == 0
    assert f'--config "{cfg.resolve()}"' in text(parse(out.read_bytes()), ".//t:Exec/t:Arguments")


def test_install_task_は不正な時刻なら終了コード_1(tmp_path):
    result = CliRunner().invoke(app, ["batch", "install-task", "--time", "25:00", "--out", str(tmp_path / "x.xml")])
    assert result.exit_code == 1 and "HH:MM" in result.output
    assert not (tmp_path / "x.xml").exists()
