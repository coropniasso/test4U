"""議事録の生成に使うプロンプト。

system はチャンクによらず固定にする。固定にしておくと、サーバー側にプロンプトのキャッシュがある場合に効く。

規則（2026-11-06 にユーザーが目的を議事録に変えたときの要件。CLAUDE.md の「議事録の生成」）:
- 発言の中に無い内容を書かない。
- owner と due は、発言の中で明示されたときだけ書く。無ければ null。人名を推測しない。
- 話者は SPEAKER_00 のようなラベルのまま扱う。
- 入力は機械の文字起こしで、聞き取り誤りを含む。意味が取れない語は推測で別の語に置き換えず、
  原文のままかぎ括弧に入れる。
- 相づち・挨拶・雑談は要点に入れない。
- 該当が無い項目は空の配列にする。
"""

from __future__ import annotations

import json

from kijun.minutes.chunk import Chunk

CHUNK_SYSTEM_PROMPT = """\
あなたは、会議の文字起こしから議事録の材料を作る担当です。
入力は、会議の一部分の文字起こしです。1行が1つの発話で、「[分:秒] 話者ラベル: 本文」の形をしています。
次の4つを、指定の JSON の形で出力してください。JSON 以外の文章は出力しないでください。

- topics: 話し合われた議題の一覧。各議題は title（議題名。短い名詞句）と points（その議題で話された要点の文の一覧）を持つ。
- decisions: この部分で決まったこと（決定事項）の文の一覧。
- todos: 誰かがやると決まった、またはやるよう頼まれた作業の一覧。各 ToDo は task（作業の内容）、owner（担当者）、due（期限）を持つ。
- open_issues: 話題に出たが、この部分では結論が出ていないこと（未決事項）の文の一覧。

# 守る規則
- 発言の中に無い内容を書かないでください。推測や一般論で補わないでください。
- owner と due は、発言の中で担当者や期限がはっきり言われているときだけ書いてください。言われていなければ null にします。
- 人名や役割を推測しないでください。話者は SPEAKER_00 のようなラベルのまま扱います。担当者を書くときも、発言に出てきた呼び方か、話者ラベルをそのまま使います。
- 入力は機械が作った文字起こしで、聞き取り誤りを含みます。意味が取れない語や、前後と合わない語を、推測で別の語に置き換えないでください。扱う必要があるときは、原文のままかぎ括弧「」に入れて書いてください。
- 相づち、挨拶、雑談、言い淀みは要点に入れないでください。
- 決定事項は、決まったと言える発言があるものだけを書いてください。提案や意見の段階のものは、points か open_issues に入れます。
- 該当するものが無い項目は、空の配列 [] にしてください。無理に埋めないでください。
- 文は、発言の内容を短く言い直した文にします。長い引用はしないでください。
"""

OVERVIEW_SYSTEM_PROMPT = """\
あなたは、会議の議事録の概要を書く担当です。
入力は、1回の会議を時間順に区切った各部分の議事録の材料（JSON）です。各部分は topics（議題と要点）、
decisions（決定事項）、todos（ToDo）、open_issues（未決事項）を持ちます。
会議全体の概要を、3文から5文で書き、{"overview": "..."} の形の JSON だけを出力してください。

# 守る規則
- 入力に書かれていない内容を書かないでください。推測や一般論で補わないでください。
- 人名や役割を推測しないでください。話者は SPEAKER_00 のようなラベルのまま扱います。
- 入力に「」で囲まれた語があれば、それは意味が取れなかった語です。別の語に置き換えず、そのまま「」に入れるか、触れないでください。
- 何が話し合われ、何が決まり、何が残っているかの順に書いてください。
"""


def format_timestamp(sec: float) -> str:
    """秒を「分:秒」にする。60 分を超えても分のまま（125:03）。"""
    total = int(sec)
    return f"{total // 60:02d}:{total % 60:02d}"


def format_utterances(chunk: Chunk) -> str:
    """チャンクの発話を `[分:秒] 話者ラベル: 本文` の行に並べる。話者ラベルが無い発話は UNKNOWN と書く。"""
    return "\n".join(
        f"[{format_timestamp(u.start_sec)}] {u.speaker_label or 'UNKNOWN'}: {u.text}"
        for u in chunk.utterances
    )


def build_chunk_messages(chunk: Chunk) -> tuple[str, str]:
    """(system, user) のプロンプトを返す。"""
    user = (
        f"次の文字起こし（会議の {format_timestamp(chunk.start_sec)} から "
        f"{format_timestamp(chunk.end_sec)} まで、発話 {len(chunk.utterances)} 件）から、"
        "議事録の材料を作ってください。\n\n"
        f"{format_utterances(chunk)}"
    )
    return CHUNK_SYSTEM_PROMPT, user


def build_overview_messages(parts: list[dict]) -> tuple[str, str]:
    """各チャンクの結果（ChunkMinutes を dict にしたもの）から、概要を書かせるプロンプトを返す。

    文字起こしの全文は渡さない（入力が num_ctx に収まらなくなるため）。
    """
    body = json.dumps(
        [{"part": i + 1, **part} for i, part in enumerate(parts)], ensure_ascii=False, indent=1
    )
    user = f"次の議事録の材料から、会議全体の概要を書いてください。\n\n{body}"
    return OVERVIEW_SYSTEM_PROMPT, user
