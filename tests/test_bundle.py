from __future__ import annotations

from datetime import timedelta

from kijun.config import BundleConfig, SettingsConfig
from kijun.ingest.bundle import AudioFile, bundle_files
from kijun.ingest.filename import parse_filename

SETTINGS = SettingsConfig()


def f(name: str, minutes: float | None) -> AudioFile:
    """ファイル名と長さ（分）から AudioFile を作る。長さ None は取得失敗を表す。"""
    return AudioFile(
        parsed=parse_filename(name, SETTINGS, warn=lambda m: None),
        duration_sec=None if minutes is None else minutes * 60,
    )


CFG = BundleConfig()  # gap_minutes = 30, require_same_setting = True


def test_30分以内なら束ねる():
    # 19:30 開始、60分 → 20:30 終了。次は 20:55 開始（間隔 25分）
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", 60), f("20260920_2055_f2f_teirei.m4a", 20)], CFG)
    assert len(bundles) == 1
    b = bundles[0]
    assert b.conversation_id == "20260920_1930_f2f_teirei"  # 先頭ファイルから作る
    assert b.file_names == ["20260920_1930_f2f_teirei.m4a", "20260920_2055_f2f_teirei.m4a"]
    assert b.total_duration_sec == 80 * 60
    assert b.has_unknown_duration is False


def test_ちょうど30分は束ねる():
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", 60), f("20260920_2100_f2f_teirei.m4a", 10)], CFG)
    assert len(bundles) == 1


def test_30分を超えたら束ねない():
    # 間隔は 31分
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", 60), f("20260920_2101_f2f_teirei.m4a", 10)], CFG)
    assert len(bundles) == 2
    assert [b.conversation_id for b in bundles] == ["20260920_1930_f2f_teirei", "20260920_2101_f2f_teirei"]


def test_gap_minutes_の設定が効く():
    cfg = BundleConfig(gap_minutes=5)
    files = [f("20260920_1930_f2f_teirei.m4a", 60), f("20260920_2040_f2f_teirei.m4a", 10)]  # 間隔 10分
    assert len(bundle_files(files, cfg)) == 2
    assert len(bundle_files(files, BundleConfig(gap_minutes=10))) == 1


def test_日付が変わったら束ねない():
    # 23:50 開始、10分 → 0:00 終了。翌日 0:10 開始（間隔 10分）でも、日付が違うので別の会議
    bundles = bundle_files([f("20260920_2350_f2f_teirei.m4a", 10), f("20260921_0010_f2f_teirei.m4a", 10)], CFG)
    assert len(bundles) == 2


def test_形式が違えば束ねない():
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", 60), f("20260920_2035_call_teirei.m4a", 10)], CFG)
    assert len(bundles) == 2


def test_種類が違えば束ねない():
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", 60), f("20260920_2035_f2f_zatsudan.m4a", 10)], CFG)
    assert len(bundles) == 2


def test_require_same_setting_が_false_なら形式と種類を無視して束ねる():
    cfg = BundleConfig(require_same_setting=False)
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", 60), f("20260920_2035_call_zatsudan.m4a", 10)], cfg)
    assert len(bundles) == 1
    assert bundles[0].conversation_id == "20260920_1930_f2f_teirei"
    assert bundles[0].setting_format == "f2f"  # 束の設定は先頭ファイルのもの


def test_長さが取得できなかった場合は開始時刻を終了時刻とみなす():
    # 先頭の長さが不明 → 終了時刻 = 19:30。次が 19:50（間隔 20分）なら束ねる。
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", None), f("20260920_1950_f2f_teirei.m4a", 10)], CFG)
    assert len(bundles) == 1
    assert bundles[0].has_unknown_duration is True
    assert bundles[0].total_duration_sec == 10 * 60  # 取れた分だけの合計


def test_長さが取得できず間隔が30分を超えれば束ねない():
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", None), f("20260920_2010_f2f_teirei.m4a", 10)], CFG)
    assert len(bundles) == 2
    assert bundles[0].has_unknown_duration is True
    assert bundles[1].has_unknown_duration is False


def test_すべて長さ不明なら合計は_None():
    bundles = bundle_files([f("20260920_1930_f2f_teirei.m4a", None)], CFG)
    assert bundles[0].total_duration_sec is None


def test_入力の順序に関わらず開始時刻の昇順で束ねる():
    files = [
        f("20260920_2055_f2f_teirei.m4a", 20),
        f("20260920_1930_f2f_teirei.m4a", 60),
        f("20260922_1000_call_sagyou.m4a", 30),
    ]
    bundles = bundle_files(files, CFG)
    assert [b.conversation_id for b in bundles] == ["20260920_1930_f2f_teirei", "20260922_1000_call_sagyou"]
    assert bundles[0].file_names[0] == "20260920_1930_f2f_teirei.m4a"


def test_3本が連鎖して1つの会議になる():
    files = [
        f("20260920_1930_f2f_teirei.m4a", 30),  # 20:00 まで
        f("20260920_2015_f2f_teirei.m4a", 30),  # 間隔 15分。20:45 まで
        f("20260920_2100_f2f_teirei.m4a", 10),  # 間隔 15分
    ]
    bundles = bundle_files(files, CFG)
    assert len(bundles) == 1
    assert len(bundles[0].files) == 3


def test_間隔は直前のファイルの終了時刻から測る():
    # 先頭の終了は 20:30。2本目は 21:00 開始・60分（22:00 終了）。
    # 3本目は 22:20 開始。先頭から見れば 110分後だが、直前（2本目）の終了からは 20分なので束ねる。
    files = [
        f("20260920_1930_f2f_teirei.m4a", 60),
        f("20260920_2100_f2f_teirei.m4a", 60),
        f("20260920_2220_f2f_teirei.m4a", 10),
    ]
    assert len(bundle_files(files, CFG)) == 1


def test_連番付きのファイルも束ねられ_会議IDに連番は入らない():
    files = [f("20260920_1930_f2f_teirei.m4a", 30), f("20260920_2010_f2f_teirei_2.m4a", 30)]
    bundles = bundle_files(files, CFG)
    assert len(bundles) == 1
    assert bundles[0].conversation_id == "20260920_1930_f2f_teirei"


def test_重なっているファイルも束ねる():
    # 先頭が 20:30 まで続いているのに、2本目が 20:00 に始まっている（間隔が負）
    files = [f("20260920_1930_f2f_teirei.m4a", 60), f("20260920_2000_f2f_teirei.m4a", 10)]
    assert len(bundle_files(files, CFG)) == 1


def test_ファイルが無ければ空():
    assert bundle_files([], CFG) == []


def test_recorded_at_は先頭ファイルの開始時刻():
    b = bundle_files([f("20260920_1930_f2f_teirei.m4a", 10)], CFG)[0]
    assert b.recorded_at.strftime("%H:%M") == "19:30"
    assert b.recorded_at + timedelta(minutes=10) == b.recorded_at + timedelta(seconds=b.total_duration_sec)
