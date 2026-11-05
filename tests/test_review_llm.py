"""重要度の判定・返信の振り分け・話者の解釈の Ollama 実装(仕様 1-3、2-5、2-6、原則2-3)。

実際の Ollama には接続せず、偽の HttpTransport に壊れた JSON を返させて、修復を通ることを確認する。
"""

from __future__ import annotations

import json

import pytest

from kijun.db import repo
from kijun.review.importance import (
    ImportanceVerdict, LlmImportanceJudge, build_user_message, propose_importance,
)
from kijun.review.llm import OllamaJsonChat, ReviewLlmError
from kijun.review.reply import LlmReplyRouter, RoutedEntry, build_user_message as reply_message, process_reply
from kijun.review.speaker import LlmSpeakerInterpreter
from kijun.transport import TransportError
from tests.fakes import FakeOllamaTransport, ollama_body
from tests.review_helpers import CID, NOW, add_item, candidate, reply, setup_conversation

GOOD = json.dumps({"importance": 3, "rationale": "読み手が行動を取れなくなるため"}, ensure_ascii=False)


def judge(cfg, contents):
    transport = FakeOllamaTransport(contents)
    return LlmImportanceJudge(OllamaJsonChat(cfg, transport)), transport


def test_重要度の判定はそのまま読める出力を受け取る(cfg):
    j, transport = judge(cfg, [GOOD])
    verdict = j.judge("冒頭に行動を書くか", ["最初に書いて"])
    assert verdict == ImportanceVerdict(importance=3, rationale="読み手が行動を取れなくなるため")
    sent = transport.posts[0]
    # num_ctx と keep_alive をリクエストごとに送れる /api/chat を使う(OpenAI 互換の /v1 は使わない)
    assert sent["url"] == "http://127.0.0.1:11434/api/chat"
    payload = sent["payload"]
    assert payload["model"] == cfg.extract.ollama.model
    assert payload["stream"] is False and payload["keep_alive"] == "0"
    assert payload["options"] == {"temperature": 0, "num_ctx": 16384, "num_predict": 8192}
    assert "最初に書いて" in payload["messages"][1]["content"]
    assert "think" not in payload  # think = None が既定なので送らない
    assert "response_format" not in payload


def test_コードフェンスと前置きの付いた出力も修復して読める(cfg):
    j, _ = judge(cfg, [f"はい、判定しました。\n```json\n{GOOD}\n```"])
    assert j.judge("x", []).importance == 3


def test_format_にスキーマを渡す_既定は渡す(cfg):
    j, transport = judge(cfg, [GOOD])
    j.judge("x", [])
    assert transport.posts[0]["payload"]["format"] == ImportanceVerdict.model_json_schema()


def test_enforces_json_schema_が偽なら_format_を送らない(cfg):
    cfg.extract.ollama.enforces_json_schema = False
    j, transport = judge(cfg, [GOOD])
    j.judge("x", [])
    assert "format" not in transport.posts[0]["payload"]


def test_think_は設定したときだけ送る(cfg):
    for value in (True, False):
        cfg.extract.ollama.think = value
        j, transport = judge(cfg, [GOOD])
        j.judge("x", [])
        assert transport.posts[0]["payload"]["think"] is value


def test_入力が上限に達したら_投げ直さずに_ReviewLlmError(cfg):
    # Ollama は入力が num_ctx を超えると、エラーを出さずに先頭を捨てる。num_ctx - num_predict に達したら疑う。
    limit = cfg.extract.ollama.num_ctx - cfg.extract.ollama.num_predict
    j, transport = judge(cfg, [(200, ollama_body(GOOD, prompt_eval_count=limit))])
    with pytest.raises(ReviewLlmError, match="切り捨て|捨てる"):
        j.judge("x", [])
    assert len(transport.posts) == 1


def test_入力が上限に達していなければ読める(cfg):
    limit = cfg.extract.ollama.num_ctx - cfg.extract.ollama.num_predict
    j, _ = judge(cfg, [(200, ollama_body(GOOD, prompt_eval_count=limit - 1))])
    assert j.judge("x", []).importance == 3


def test_読めない出力は1回だけ投げ直し_それでも読めなければ_ReviewLlmError(cfg):
    j, transport = judge(cfg, ["判定できません"])
    with pytest.raises(ReviewLlmError, match="2 回とも"):
        j.judge("x", [])
    assert len(transport.posts) == 2


def test_1回目が読めなくても2回目で読めれば成功(cfg):
    j, transport = judge(cfg, ["判定できません", GOOD])
    assert j.judge("x", []).importance == 3 and len(transport.posts) == 2


def test_範囲外の重要度は読めない出力として扱う(cfg):
    j, _ = judge(cfg, [json.dumps({"importance": 5, "rationale": "x"})])
    with pytest.raises(ReviewLlmError):
        j.judge("x", [])


def test_接続できなければ_ReviewLlmError(cfg):
    j, _ = judge(cfg, [TransportError("接続拒否")])
    with pytest.raises(ReviewLlmError, match="送れませんでした") as e:
        j.judge("x", [])
    assert "ollama list" in str(e.value)


def test_HTTP_エラーと形の違う応答も_ReviewLlmError(cfg):
    with pytest.raises(ReviewLlmError, match="HTTP 500"):
        judge(cfg, [(500, "boom")])[0].judge("x", [])
    with pytest.raises(ReviewLlmError, match="/api/chat の形"):
        judge(cfg, [(200, '{"result": 1}')])[0].judge("x", [])


def test_根拠の発言は最大3件まで渡す(con, cfg):
    setup_conversation(con)
    from tests.review_helpers import add_request

    add_item(con, "item_0001")
    for n in range(5):
        add_request(con, f"r{n}", quote=f"発言{n}")
        repo.add_evidence(con, "item_0001", f"r{n}", None, NOW.replace(minute=n))
    j, transport = judge(cfg, [GOOD])
    propose_importance(con, j, repo.get_knowledge_item(con, "item_0001"), NOW)
    user = transport.posts[0]["payload"]["messages"][1]["content"]
    assert "発言0" in user and "発言2" in user and "発言3" not in user


def test_判定に失敗しても_importance_は_NULL_のまま_review_log_に何も残さない(con, cfg):
    candidate(con, 1, importance=None)
    j, _ = judge(cfg, ["読めない"])
    assert propose_importance(con, j, repo.get_knowledge_item(con, "item_0001"), NOW) is None
    assert repo.get_knowledge_item(con, "item_0001").importance is None
    assert repo.list_review_log(con) == []


def test_プロンプトは推測で補わないことを指示している():
    from kijun.review import reply as reply_module, speaker as speaker_module

    assert "推測" in reply_module.SYSTEM_PROMPT and "推測" in speaker_module.SYSTEM_PROMPT
    assert "(記録なし)" in build_user_message("x", [])


# --- 返信の振り分け ----------------------------------------------------------------------------


def router(cfg, contents):
    transport = FakeOllamaTransport(contents)
    return LlmReplyRouter(OllamaJsonChat(cfg, transport)), transport


def test_返信の振り分けは_items_を持つオブジェクトも素の配列も受け取れる(cfg):
    entries = [{"column": "reason", "text": "忙しいから", "target": None}]
    for body in (json.dumps({"items": entries}, ensure_ascii=False), json.dumps(entries, ensure_ascii=False)):
        r, _ = router(cfg, [body])
        item = repo_item()
        assert r.route("忙しいから", item).items == [RoutedEntry(column="reason", text="忙しいから")]


def test_返信の振り分けで末尾が切れた出力は最後の完全な要素までで閉じる(cfg):
    body = '{"items": [{"column": "reason", "text": "忙しい", "target": null}, {"column": "applies_whe'
    r, _ = router(cfg, [body])
    assert [e.column for e in r.route("x", repo_item()).items] == ["reason"]


def test_欄の名前が違う出力は読めない出力として扱う(cfg):
    r, _ = router(cfg, [json.dumps({"items": [{"column": "note", "text": "x", "target": None}]})])
    with pytest.raises(ReviewLlmError):
        r.route("x", repo_item())


def test_矛盾の投稿では_A_と_B_の両方の項目をプロンプトに入れる():
    a, b = repo_item("item_0001", "文言A"), repo_item("item_0002", "文言B")
    text = reply_message("返信", a, b)
    assert "項目A" in text and "文言A" in text and "項目B" in text and "文言B" in text


def test_Ollama_の振り分けを通した返信が_review_log_に原文付きで残る(con, cfg):
    candidate(con, 1)
    body = json.dumps({"items": [{"column": "applies_when", "text": "社外向け", "target": None}]}, ensure_ascii=False)
    r, _ = router(cfg, [f"結果です\n{body}"])
    outcome = process_reply(con, r, "item_0001", reply("社外向けのとき"), NOW)
    assert outcome.routed and repo.get_knowledge_item(con, "item_0001").applies_when == "社外向け"
    assert repo.list_review_log(con, "item_0001", "reply_received")[0].after == "社外向けのとき"


# --- 話者の解釈 --------------------------------------------------------------------------------


def test_話者の解釈は番号を正規化し_人物ラベルを小文字にする(cfg):
    transport = FakeOllamaTransport([json.dumps({"items": [{"speaker": "00", "person": "self"}, {"speaker": "SPEAKER_01", "person": "Kouhai_A"}]})])
    interp = LlmSpeakerInterpreter(OllamaJsonChat(cfg, transport))
    assert interp.interpret("00は自分、01は後輩A", ["SPEAKER_00", "SPEAKER_01"]) == {
        "SPEAKER_00": "self", "SPEAKER_01": "kouhai_a",
    }
    assert "SPEAKER_00, SPEAKER_01" in transport.posts[0]["payload"]["messages"][1]["content"]


def repo_item(item_id: str = "item_0001", check_text: str = "文言") -> repo.KnowledgeItemRow:
    from datetime import datetime

    return repo.KnowledgeItemRow(item_id, None, check_text, None, None, None, None, None, "candidate", 1, datetime(2026, 11, 3))
