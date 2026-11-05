"""Windows のタスクスケジューラに登録するための XML を作る(仕様 3-3)。

登録そのものは `schtasks /create /xml` でユーザーが実行する(管理者権限が必要になることがあるので、
コマンドを生成して案内する形にする)。この XML の生成は純粋な文字列処理で、外部に触らない。

XML に必ず入れる設定は次の2つ。
- WakeToRun = true: CLAUDE.md 4-8 の「タスクを実行するためにスリープを解除する」。これが無いと PC が
  スリープしていると実行されない。
- StartWhenAvailable = true: PC が電源オフだった場合に、次に起動したときに実行する。

ほかの設定の理由:
- ExecutionTimeLimit = PT0S(無制限): ゲームによる延期の待ち時間(最大 batch.defer_max_hours)を含めると
  所要時間が読めないため。既定の72時間で打ち切るより、batch の記録で遅さを見て判断する。
- MultipleInstancesPolicy = IgnoreNew: 前夜のバッチがまだ動いているときに、重ねて起動しない
  (DuckDB は同時に複数のプロセスから書き込めない)。
- DisallowStartIfOnBatteries / StopIfGoingOnBatteries = false: ノート PC でなくデスクトップだが、
  既定がバッテリー駆動のとき実行しない設定なので明示して外す。
- LogonType = InteractiveToken: ユーザーがログオンしている間(画面ロックを含む)に実行する。パスワードを
  保存しないで済む。ログオフしている間は実行されない。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

TASK_NAME = "kijun nightly batch"
DEFAULT_TASK_TIME = "23:30"
TASK_NAMESPACE = "http://schemas.microsoft.com/windows/2004/02/mit/task"

_TIME = re.compile(r"(\d{1,2}):(\d{2})")


def parse_time(text: str) -> tuple[int, int]:
    """`23:30` を (時, 分) にする。形式や範囲が不正なら ValueError。"""
    m = _TIME.fullmatch(text.strip())
    if m is None or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"時刻は HH:MM の形式(00:00 から 23:59)で指定してください: {text}")
    return int(m.group(1)), int(m.group(2))


def build_task_xml(
    time_of_day: str,
    command: str,
    arguments: str,
    working_directory: Path | str,
    start_date: date | None = None,
) -> bytes:
    """タスクスケジューラの XML(UTF-16)を作る。毎日 time_of_day に command arguments を working_directory で実行する。"""
    hour, minute = parse_time(time_of_day)
    start = datetime.combine(start_date or date.today(), datetime.min.time()).replace(hour=hour, minute=minute)

    ET.register_namespace("", TASK_NAMESPACE)

    def el(parent: ET.Element | None, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
        node = ET.Element(f"{{{TASK_NAMESPACE}}}{tag}", attrs) if parent is None else ET.SubElement(
            parent, f"{{{TASK_NAMESPACE}}}{tag}", attrs
        )
        if text is not None:
            node.text = text
        return node

    task = el(None, "Task", version="1.4")
    info = el(task, "RegistrationInfo")
    el(info, "Description", "kijun: 会話録音から判断基準を抽出する夜間バッチ(uv run kijun batch run)")

    trigger = el(el(task, "Triggers"), "CalendarTrigger")
    el(trigger, "StartBoundary", start.strftime("%Y-%m-%dT%H:%M:%S"))
    el(trigger, "Enabled", "true")
    el(el(trigger, "ScheduleByDay"), "DaysInterval", "1")

    principal = el(el(task, "Principals"), "Principal", id="Author")
    el(principal, "LogonType", "InteractiveToken")
    el(principal, "RunLevel", "LeastPrivilege")

    settings = el(task, "Settings")
    el(settings, "MultipleInstancesPolicy", "IgnoreNew")
    el(settings, "DisallowStartIfOnBatteries", "false")
    el(settings, "StopIfGoingOnBatteries", "false")
    el(settings, "AllowHardTerminate", "true")
    el(settings, "StartWhenAvailable", "true")
    el(settings, "RunOnlyIfNetworkAvailable", "false")
    idle = el(settings, "IdleSettings")
    el(idle, "StopOnIdleEnd", "false")
    el(idle, "RestartOnIdle", "false")
    el(settings, "AllowStartOnDemand", "true")
    el(settings, "Enabled", "true")
    el(settings, "Hidden", "false")
    el(settings, "RunOnlyIfIdle", "false")
    el(settings, "WakeToRun", "true")
    el(settings, "ExecutionTimeLimit", "PT0S")
    el(settings, "Priority", "7")

    exec_ = el(el(task, "Actions", Context="Author"), "Exec")
    el(exec_, "Command", command)
    el(exec_, "Arguments", arguments)
    el(exec_, "WorkingDirectory", str(working_directory))

    ET.indent(task)
    return ET.tostring(task, encoding="UTF-16", xml_declaration=True)


def schtasks_command(xml_path: Path | str) -> str:
    """ユーザーが実行する登録コマンド。同名のタスクが既にあるときは、末尾に /f を付けると置き換える(案内は cli.py が出す)。"""
    return f'schtasks /create /tn "{TASK_NAME}" /xml "{xml_path}"'
