"""議事録の生成（minutes/）。LLM は tests/fakes.py の FakeJsonChat で差し替える。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from kijun.db import repo
from kijun.llm.chat import LlmError
from kijun.minutes.generate import (
    NO_CONTENT_OVERVIEW,
    NO_UTTERANCE_OVERVIEW,
    MinutesError,
    build_minutes,
    check_chunk_input_size,
    generate_minutes,
    merge_chunk_minutes,
)
from kijun.minutes.chunk import split_into_chunks
from kijun.minutes.models import ChunkMinutes, Overview, Todo, Topic
from kijun.minutes.prompt import (
    CHUNK_SYSTEM_PROMPT,
    OVERVIEW_SYSTEM_PROMPT,
    build_chunk_messages,
    build_overview_messages,
    format_timestamp,
)
from kijun.minutes.render import (
    EMPTY,
    NOT_STATED,
    minutes_path,
    parse_sections,
    render_markdown,
    todo_rows,
)
from kijun.models import Utterance
from tests.conftest import make_conversation, make_utterances
from tests.fakes import FakeJsonChat

NOW = datetime(2026, 11, 6, 9, 0)
quiet = lambda m: None  # noqa: E731


def chunk_result(i: int) -> ChunkMinutes:
    return ChunkMinutes(
        topics=[Topic(title=f"議題{i}", points=[f"要点{i}"])],
        decisions=[f"決定{i}"],
        todos=[Todo(task=f"作業{i}", owner="SPEAKER_00" if i == 0 else None, due=None)],
        open_issues=[f"未決{i}"],
    )


def handler(system, user, model_cls, name):
    if model_cls is Overview:
        return Overview(overview="会議の概要です。")
    return chunk_result(int(name.split()[1].split("/")[0]) - 1)  # 「チャンク 2/4」-> 1


# --- プロンプト --------------------------------------------------------------------------------


def test_時刻は分_秒で_60分を超えても分のまま():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65.9) == "01:05"
    assert format_timestamp(125 * 60 + 3) == "125:03"


def test_チャンクのプロンプトは発話を時刻と話者つきの行に並べる():
    chunk = split_into_chunks(make_utterances(3), 15, 0)[0]
    system, user = build_chunk_messages(chunk)
    assert system == CHUNK_SYSTEM_PROMPT
    assert "[00:00] SPEAKER_00: 発話0" in user and "[01:00] SPEAKER_01: 発話1" in user
    assert "発話 3 件" in user


def test_話者ラベルが無い発話は_UNKNOWN_と書く():
    chunk = split_into_chunks([Utterance(0, None, 0.0, 5.0, "こんにちは")], 15, 0)[0]
    assert "[00:00] UNKNOWN: こんにちは" in build_chunk_messages(chunk)[1]


def test_システムプロンプトに規則が全部入っている():
    for word in ["書かないでください", "null", "推測しないでください", "SPEAKER_00", "聞き取り誤り", "かぎ括弧", "相づち", "空の配列"]:
        assert word in CHUNK_SYSTEM_PROMPT, word
    for word in ["書かれていない内容を書かない", "3文から5文", "overview"]:
        assert word in OVERVIEW_SYSTEM_PROMPT, word


def test_概要のプロンプトは各チャンクの結果の_JSON_を渡し_文字起こしは渡さない():
    merged = merge_chunk_minutes([chunk_result(0), chunk_result(1)])
    system, user = build_overview_messages(
        {
            "topics": [t.model_dump() for t in merged.topics],
            "decisions": merged.decisions,
            "todos": [t.model_dump() for t in merged.todos],
            "open_issues": merged.open_issues,
        }
    )
    assert system == OVERVIEW_SYSTEM_PROMPT
    assert "議題0" in user and "決定1" in user and "作業1" in user
    assert "決定事項は無い" not in user  # 決定事項があるときは注意書きを付けない


def test_決定事項が空なら概要のプロンプトに_決定事項は無い_と明示する():
    _, user = build_overview_messages({"topics": [], "decisions": [], "todos": [{"task": "資料を送る"}], "open_issues": []})
    assert "決定事項は無い" in user and "「決定」" in user


def test_プロンプトは決定_聞き取れない語_担当_表記の規則を持つ():
    for word in ["合意・決定", "迷ったら決定事項に入れない", "意味の説明を足さない", "呼びかけられただけ", "表記をそのまま"]:
        assert word in CHUNK_SYSTEM_PROMPT, word
    assert "13禁" not in CHUNK_SYSTEM_PROMPT  # 会議の内容を例に使わない(例の写しを防ぐ)
    for word in ["decisions に書かれていること", "決定事項は無かった"]:
        assert word in OVERVIEW_SYSTEM_PROMPT, word


def test_空白と句読点を除いて一致する文は1つにする():
    parts = [
        ChunkMinutes(decisions=["水曜日に、実施する。"], open_issues=["場所は未定"]),
        ChunkMinutes(decisions=["水曜日に実施する", "場所は 未定。"], todos=[Todo(task="資料を、送る"), Todo(task="資料を送る。")]),
    ]
    m = merge_chunk_minutes(parts)
    assert m.decisions == ["水曜日に、実施する。"]  # 「場所は未定」は未決事項にも出るので決定事項から除く
    assert m.open_issues == ["場所は未定"]
    assert len(m.todos) == 1


def test_概要が決定と書いたのに決定事項が無ければ警告し_書き換えない(cfg):
    messages = []

    def h(system, user, model_cls, name):
        if model_cls is Overview:
            return Overview(overview="対応が決定しました。")
        return ChunkMinutes(todos=[Todo(task="資料を送る")])

    doc, _ = build_minutes(FakeJsonChat(h), cfg, make_utterances(5), messages.append)
    assert doc.overview == "対応が決定しました。"
    assert len(doc.warnings) == 1 and "決定事項が無い" in doc.warnings[0]
    assert any(m.startswith("警告:") for m in messages)
    conv = make_conversation("20260920_1930_f2f_teirei", duration_sec=600.0)
    assert "- 注意: 決定事項が無い" in render_markdown(conv, make_utterances(5), doc, "m", NOW)


def test_決定事項があれば概要に決定と書いても警告しない(cfg):
    def h(system, user, model_cls, name):
        if model_cls is Overview:
            return Overview(overview="水曜日に決まった。")
        return ChunkMinutes(decisions=["水曜日にする"])

    doc, _ = build_minutes(FakeJsonChat(h), cfg, make_utterances(5), quiet)
    assert doc.warnings == []


# --- つなぎ合わせ ------------------------------------------------------------------------------


def test_同じ議題名は要点をまとめ_同じ文は1つにする():
    parts = [
        ChunkMinutes(
            topics=[Topic(title="日程", points=["水曜日", "午後"])],
            decisions=["水曜日にする", " 水曜日にする "],
            todos=[Todo(task="資料を作る", owner="SPEAKER_00", due="金曜日")],
        ),
        ChunkMinutes(
            topics=[Topic(title="日程", points=["午後", "場所は未定"]), Topic(title="予算", points=[])],
            decisions=["水曜日にする"],
            todos=[Todo(task="資料を作る", owner="SPEAKER_00", due="金曜日"), Todo(task="  ", owner="x")],
            open_issues=["場所", ""],
        ),
    ]
    m = merge_chunk_minutes(parts)
    assert [(t.title, t.points) for t in m.topics] == [("日程", ["水曜日", "午後", "場所は未定"]), ("予算", [])]
    assert m.decisions == ["水曜日にする"]
    assert m.todos == [Todo(task="資料を作る", owner="SPEAKER_00", due="金曜日")]  # 内容が空の ToDo は捨てる
    assert m.open_issues == ["場所"]


def test_ToDo_は担当か期限が違えば別の作業として残す():
    parts = [ChunkMinutes(todos=[Todo(task="連絡する", owner="A"), Todo(task="連絡する", owner="B"), Todo(task="連絡する")])]
    assert len(merge_chunk_minutes(parts).todos) == 3


def test_空の担当と期限は_None_にそろえる():
    m = merge_chunk_minutes([ChunkMinutes(todos=[Todo(task="作業", owner="  ", due="")])])
    assert m.todos[0].owner is None and m.todos[0].due is None


# --- build_minutes -----------------------------------------------------------------------------


def test_60分の会議は15分ごと4チャンクに分けて_最後に概要を1回だけ作る(cfg):
    chat = FakeJsonChat(handler)
    doc, count = build_minutes(chat, cfg, make_utterances(60), quiet)

    assert count == 4
    kinds = [c[2] for c in chat.calls]
    assert kinds == [ChunkMinutes] * 4 + [Overview]
    assert [t.title for t in doc.merged.topics] == ["議題0", "議題1", "議題2", "議題3"]
    assert doc.overview == "会議の概要です。"
    # 発話は重ならず、欠けない
    seen = [line for c in chat.calls[:4] for line in c[1].splitlines() if line.startswith("[")]
    assert len(seen) == 60 and len(set(seen)) == 60


def test_概要の入力はチャンクの結果の_JSON_で_文字起こし全文ではない(cfg):
    chat = FakeJsonChat(handler)
    build_minutes(chat, cfg, make_utterances(30), quiet)
    system, user, model_cls, _ = chat.calls[-1]
    assert model_cls is Overview and system == OVERVIEW_SYSTEM_PROMPT
    assert "SPEAKER_01: 発話" not in user  # 発話の行は含まない
    body = json.loads(user.split("\n\n", 1)[1])
    assert body["decisions"] == ["決定0", "決定1"] and set(body) == {"topics", "decisions", "todos", "open_issues"}


def test_発話が無ければ_LLM_を呼ばない(cfg):
    chat = FakeJsonChat(handler)
    doc, count = build_minutes(chat, cfg, [], quiet)
    assert chat.calls == [] and count == 0 and doc.overview == NO_UTTERANCE_OVERVIEW


def test_全チャンクが空なら概要も_LLM_に作らせない(cfg):
    chat = FakeJsonChat()  # 既定値(空)を返す
    doc, _ = build_minutes(chat, cfg, make_utterances(5), quiet)
    assert [c[2] for c in chat.calls] == [ChunkMinutes]  # 概要は呼ばない(入力が空だと作り話になる)
    assert doc.overview == NO_CONTENT_OVERVIEW and doc.merged.is_empty


def test_チャンクが1つ失敗したら会議全体を失敗にする(cfg):
    def failing(system, user, model_cls, name):
        return LlmError("接続拒否") if name == "チャンク 2/4" else handler(system, user, model_cls, name)

    chat = FakeJsonChat(failing)
    with pytest.raises(MinutesError) as e:
        build_minutes(chat, cfg, make_utterances(60), quiet)
    assert "チャンク 2/4" in str(e.value) and "発話 15〜29" in str(e.value) and "接続拒否" in str(e.value)
    assert Overview not in [c[2] for c in chat.calls]


def test_概要の失敗も会議全体の失敗にする(cfg):
    def failing(system, user, model_cls, name):
        return LlmError("読めない") if model_cls is Overview else handler(system, user, model_cls, name)

    with pytest.raises(MinutesError, match="概要"):
        build_minutes(FakeJsonChat(failing), cfg, make_utterances(10), quiet)


def test_概要が空文字列なら失敗にする(cfg):
    def blank(system, user, model_cls, name):
        return Overview(overview="  ") if model_cls is Overview else chunk_result(0)

    with pytest.raises(MinutesError, match="空"):
        build_minutes(FakeJsonChat(blank), cfg, make_utterances(10), quiet)


def test_入力が上限を超える見込みなら警告する(cfg):
    cfg.minutes.tokens_per_character = 50.0  # 係数を大きくして、概算が上限を超える状況を作る
    logs: list[str] = []
    build_minutes(FakeJsonChat(handler), cfg, make_utterances(10), logs.append)
    assert any("上限" in m and "minutes.chunk_minutes" in m for m in logs)
    chunk = split_into_chunks(make_utterances(10), 15, 0)[0]
    assert check_chunk_input_size(chunk, cfg) is not None
    cfg.minutes.tokens_per_character = 0.55
    assert check_chunk_input_size(chunk, cfg) is None


# --- generate_minutes(DB とファイル) -----------------------------------------------------------


def seed(con, cid="20260920_1930_f2f_teirei", count=20):
    repo.insert_conversation(con, make_conversation(cid, transcribed_at=datetime(2026, 9, 21), duration_sec=1200.0))
    repo.replace_utterances(con, cid, make_utterances(count))
    return cid


def test_議事録を_Markdown_に書き_minutes_に記録する(con, cfg):
    cid = seed(con)
    run = generate_minutes(con, cfg, cid, FakeJsonChat(handler), NOW, quiet)

    path = minutes_path(cfg, cid)
    assert path.exists() and run.path == path.as_posix() and run.chunk_count == 2
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"# 議事録 {cid}\n")
    row = repo.get_minutes(con, cid)
    assert row.path == path.as_posix() and row.model_name == "qwen3.5:9b" and row.created_at == NOW
    assert row.posted_at is None


def test_失敗したらファイルも_DB_も書かない_既存の議事録を壊さない(con, cfg):
    cid = seed(con)
    generate_minutes(con, cfg, cid, FakeJsonChat(handler), NOW, quiet)
    before = minutes_path(cfg, cid).read_text(encoding="utf-8")
    with pytest.raises(MinutesError):
        generate_minutes(con, cfg, cid, FakeJsonChat(lambda *a: LlmError("x")), datetime(2026, 11, 7), quiet)
    assert minutes_path(cfg, cid).read_text(encoding="utf-8") == before
    assert repo.get_minutes(con, cid).created_at == NOW


def test_最初の失敗ではファイルも行も作らない(con, cfg):
    cid = seed(con)
    with pytest.raises(MinutesError):
        generate_minutes(con, cfg, cid, FakeJsonChat(lambda *a: LlmError("x")), NOW, quiet)
    assert not minutes_path(cfg, cid).exists() and repo.get_minutes(con, cid) is None


def test_作り直しは上書きし_投稿済みの記録を残す(con, cfg):
    cid = seed(con)
    generate_minutes(con, cfg, cid, FakeJsonChat(handler), NOW, quiet)
    repo.mark_minutes_posted(con, cid, "m1", NOW)
    generate_minutes(con, cfg, cid, FakeJsonChat(handler), datetime(2026, 11, 7), quiet)
    row = repo.get_minutes(con, cid)
    assert row.created_at == datetime(2026, 11, 7) and row.posted_at == NOW


def test_文字起こし前の会議と未登録の会議は_LookupError(con, cfg):
    repo.insert_conversation(con, make_conversation("c1"))
    with pytest.raises(LookupError, match="文字起こしが済んでいません"):
        generate_minutes(con, cfg, "c1", FakeJsonChat(handler), NOW, quiet)
    with pytest.raises(LookupError, match="conversations にありません"):
        generate_minutes(con, cfg, "none", FakeJsonChat(handler), NOW, quiet)


def test_発話が0件の会議も議事録を作る(con, cfg):
    cid = seed(con, count=0)
    chat = FakeJsonChat(handler)
    generate_minutes(con, cfg, cid, chat, NOW, quiet)
    assert chat.calls == []
    assert NO_UTTERANCE_OVERVIEW in minutes_path(cfg, cid).read_text(encoding="utf-8")
    assert repo.get_minutes(con, cid) is not None  # 行が無いと毎晩やり直すため


# --- Markdown ----------------------------------------------------------------------------------


def make_doc(chat_handler=handler, count=20, cfg=None):
    from kijun.config import Config

    doc, _ = build_minutes(FakeJsonChat(chat_handler), cfg or Config(), make_utterances(count), quiet)
    return doc


def test_Markdown_の節は決まった順序で並ぶ():
    conv = make_conversation("20260920_1930_f2f_teirei", duration_sec=1200.0)
    text = render_markdown(conv, make_utterances(20), make_doc(), "qwen3.5:9b", NOW)
    heads = [l for l in text.splitlines() if l.startswith("## ")]
    assert heads == ["## 概要", "## 議題と要点", "## 決定事項", "## ToDo", "## 未決事項", "## 付録: 文字起こし"]
    assert "- 日時: 2026-09-20 19:30" in text
    assert "- 形式・種類: f2f / teirei" in text
    assert "- 長さ: 20分" in text
    assert "- 話者: SPEAKER_00, SPEAKER_01" in text
    assert "- 作成: qwen3.5:9b、2026-11-06 09:00（機械の文字起こしと要約。誤りを含む）" in text
    assert "### 議題0\n\n- 要点0" in text
    assert "[00:00] SPEAKER_00: 発話0" in text and "[19:00] SPEAKER_01: 発話19" in text


def test_ToDo_は表で_null_は記載なしと書く():
    conv = make_conversation(duration_sec=1200.0)
    text = render_markdown(conv, make_utterances(20), make_doc(), "m", NOW)
    assert "| 内容 | 担当 | 期限 |" in text
    assert f"| 作業0 | SPEAKER_00 | {NOT_STATED} |" in text
    assert f"| 作業1 | {NOT_STATED} | {NOT_STATED} |" in text


def test_表のセルの縦棒と改行は壊さない():
    def pipes(system, user, model_cls, name):
        if model_cls is Overview:
            return Overview(overview="概要")
        return ChunkMinutes(todos=[Todo(task="A|B\nC", owner="x|y")])

    conv = make_conversation(duration_sec=60.0)
    text = render_markdown(conv, make_utterances(2), make_doc(pipes, 2), "m", NOW)
    _, sections = parse_sections(text)
    assert todo_rows(sections["ToDo"]) == [("A|B C", "x|y", NOT_STATED)]


def test_空の節は_なし_と書く():
    conv = make_conversation(duration_sec=60.0)
    doc = make_doc(lambda *a: Overview(overview="概要") if a[2] is Overview else ChunkMinutes(), 2)
    # 全部空だと概要を LLM に作らせないので、固定の文になる
    text = render_markdown(conv, make_utterances(2), doc, "m", NOW)
    _, sections = parse_sections(text)
    assert sections["概要"] == NO_CONTENT_OVERVIEW
    for name in ["議題と要点", "決定事項", "ToDo", "未決事項"]:
        assert sections[name] == EMPTY


def test_長さが不明なら最後の発話の終了時刻から出し_発話も無ければ不明():
    conv = make_conversation(duration_sec=None)
    text = render_markdown(conv, make_utterances(10), make_doc(count=10), "m", NOW)
    assert "- 長さ: 10分" in text
    text0 = render_markdown(conv, [], make_doc(count=0), "m", NOW)
    assert "- 長さ: 不明" in text0 and "- 話者: なし" in text0


def test_節の読み出しは付録で打ち切る():
    conv = make_conversation(duration_sec=60.0)
    utterances = [Utterance(0, "SPEAKER_00", 0.0, 5.0, "## 決定事項 と言った")]
    text = render_markdown(conv, utterances, make_doc(count=1), "m", NOW)
    head, sections = parse_sections(text)
    assert head.startswith("# 議事録") and "付録: 文字起こし" not in sections
    assert "と言った" not in sections["決定事項"]
