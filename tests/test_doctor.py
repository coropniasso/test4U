"""`kijun doctor` の判定ロジックと推奨値の算出。

偽の HTTP 層を差し込むので、推論サーバーが無いこのクラウド環境で全部テストできる。
実サーバーへの接続は tests/test_external_services.py の requires_network 側で行う。
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from kijun.config import Config
from kijun.doctor import (
    MAX_INPUT_MULTIPLIERS,
    OUTPUT_DIRECT_JSON,
    OUTPUT_REPAIRED,
    OUTPUT_UNREADABLE,
    SCHEMA_ACCEPTED_NOT_ENFORCED,
    SCHEMA_ENFORCED,
    SCHEMA_UNSUPPORTED,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARN,
    TIMEOUT_FLOOR_SEC,
    WHOLE_MEETING_CHUNK_MINUTES,
    build_recommendations,
    check_max_input,
    check_minimal_request,
    check_models,
    check_schema_enforcement,
    check_speed,
    classify_output,
    format_report,
    report_path,
    run_doctor,
    sample_chunk,
    stretched_chunk,
)
from kijun.extract.tokens import estimate_tokens, tokens_for_minutes
from kijun.transport import HttpResponse, TransportError

NOW = datetime(2026, 11, 3, 21, 30, 15)

ITEM = {
    "seq_from": 0,
    "seq_to": 0,
    "speaker_from_label": "SPEAKER_00",
    "speaker_to_label": "SPEAKER_01",
    "counterpart_role": None,
    "artifact_type": None,
    "quote": "募集要項の最初に、何をしてほしいかを書いてください。",
    "request_summary": "告知文の冒頭に、読み手にしてほしい行動を書くことを求めた",
    "stated_reason": None,
    "applies_when": None,
    "reusable": True,
    "pass_criterion": "冒頭2行以内に、読み手にしてほしい行動が書かれているか",
}
GOOD_JSON = json.dumps({"items": [ITEM]}, ensure_ascii=False)
MODELS_BODY = json.dumps({"data": [{"id": "qwen3.8-flash-next"}, {"id": "other-model"}]})


def chat(content: str, prompt_tokens: int | None = None, completion_tokens: int | None = None) -> str:
    body: dict = {"choices": [{"message": {"content": content}}]}
    if prompt_tokens is not None or completion_tokens is not None:
        body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return json.dumps(body, ensure_ascii=False)


class FakeTransport:
    """決めた応答を返す偽の HTTP 層。

    get_response は /v1/models 用。post_responses は POST の応答を順に返す。
    足りなくなったら最後の応答を繰り返す（手順4・5が何回投げても止まらないように）。
    """

    def __init__(self, get_response=None, post_responses=None, elapsed_sec: float = 2.0) -> None:
        self._get = get_response
        self._posts = list(post_responses or [])
        self._elapsed = elapsed_sec
        self.gets: list[str] = []
        self.posts: list[dict] = []

    def _make(self, item) -> HttpResponse:
        if isinstance(item, Exception):
            raise item
        if isinstance(item, HttpResponse):
            return item
        status, text = item
        return HttpResponse(status_code=status, text=text, elapsed_sec=self._elapsed)

    def get(self, url: str, timeout_sec: float) -> HttpResponse:
        self.gets.append(url)
        if self._get is None:
            raise TransportError("get の応答が設定されていません")
        return self._make(self._get)

    def post(self, url: str, payload: dict, timeout_sec: float) -> HttpResponse:
        self.posts.append({"url": url, "payload": payload, "timeout_sec": timeout_sec})
        if not self._posts:
            raise TransportError("post の応答が設定されていません")
        item = self._posts[0] if len(self._posts) == 1 else self._posts.pop(0)
        return self._make(item)


# --- 入力の作り方 -----------------------------------------------------------------------------


def test_固定の発話で入力を作るので音声が無くても実行できる():
    chunk = sample_chunk()
    assert len(chunk.utterances) == 3
    assert chunk.seq_from == 0 and chunk.seq_to == 2
    assert all(u.text for u in chunk.utterances)


def test_入力を目標のトークン数に引き伸ばせる():
    small = stretched_chunk(1000)
    big = stretched_chunk(50000)
    assert len(big.utterances) > len(small.utterances)
    # 文字起こし部分の概算が目標に近い（係数は概算なので 2 割の幅を許す）
    text_tokens = sum(estimate_tokens(u.text) for u in big.utterances)
    assert 0.8 * 50000 <= text_tokens <= 1.2 * 50000
    # seq が 0 から連続している（プロンプトの番号づけが壊れない）
    assert [u.seq for u in big.utterances] == list(range(len(big.utterances)))


def test_引き伸ばしても最低1発話は作る():
    assert len(stretched_chunk(1).utterances) == 1


# --- 出力の分類 -------------------------------------------------------------------------------


def test_出力の分類_そのまま読める():
    kind, actions, _note = classify_output(GOOD_JSON)
    assert kind == OUTPUT_DIRECT_JSON and actions == []


def test_出力の分類_修復して読める():
    kind, actions, note = classify_output("はい、結果です。\n" + GOOD_JSON)
    assert kind == OUTPUT_REPAIRED and actions and "散文" in note


def test_出力の分類_読めない():
    kind, _actions, note = classify_output("抽出できませんでした")
    assert kind == OUTPUT_UNREADABLE and note


# --- 手順1: 接続確認 ---------------------------------------------------------------------------


def test_手順1_接続できてモデルが一覧にある():
    transport = FakeTransport(get_response=(200, MODELS_BODY))
    r = check_models(transport, Config())
    assert r.status == STATUS_OK
    assert transport.gets == ["http://localhost:8080/v1/models"]
    assert r.detail["model_ids"] == ["qwen3.8-flash-next", "other-model"]


def test_手順1_設定のモデルが一覧に無ければ注意():
    transport = FakeTransport(get_response=(200, json.dumps({"data": [{"id": "別のモデル"}]})))
    r = check_models(transport, Config())
    assert r.status == STATUS_WARN and "一覧にありません" in r.summary


def test_手順1_接続できなければ失敗し_base_url_を疑うよう案内する():
    r = check_models(FakeTransport(get_response=TransportError("接続拒否")), Config())
    assert r.status == STATUS_ERROR and "base_url" in r.summary


def test_手順1_HTTP_エラーは失敗():
    r = check_models(FakeTransport(get_response=(404, "not found")), Config())
    assert r.status == STATUS_ERROR and "404" in r.summary


@pytest.mark.parametrize(
    "body",
    [
        json.dumps(["qwen3.8-flash-next"]),  # 素の配列
        json.dumps({"models": [{"name": "qwen3.8-flash-next"}]}),  # models/name の形
    ],
)
def test_手順1_モデル一覧の形が違っても読み取る(body):
    r = check_models(FakeTransport(get_response=(200, body)), Config())
    assert r.status == STATUS_OK and "qwen3.8-flash-next" in r.detail["model_ids"]


def test_手順1_一覧を読み取れなければ注意にとどめる():
    r = check_models(FakeTransport(get_response=(200, json.dumps({"ok": True}))), Config())
    assert r.status == STATUS_WARN and "読み取れません" in r.summary


# --- 手順2: 最小リクエスト（3分類） ------------------------------------------------------------


def test_手順2a_そのまま読めた():
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON))])
    r = check_minimal_request(transport, Config())
    assert r.status == STATUS_OK and "(a)" in r.summary
    assert r.detail["classification"] == OUTPUT_DIRECT_JSON
    # response_format を付けずに投げている
    assert "response_format" not in transport.posts[0]["payload"]


def test_手順2b_修復すれば読めた():
    body = chat("```json\n" + GOOD_JSON + "\n```")
    r = check_minimal_request(FakeTransport(post_responses=[(200, body)]), Config())
    assert r.status == STATUS_WARN and "(b)" in r.summary
    assert r.detail["classification"] == OUTPUT_REPAIRED
    assert r.detail["repair_actions"]


def test_手順2c_修復しても読めなかった():
    r = check_minimal_request(FakeTransport(post_responses=[(200, chat("無理です"))]), Config())
    assert r.status == STATUS_ERROR and "(c)" in r.summary
    assert r.detail["classification"] == OUTPUT_UNREADABLE


def test_手順2_応答が_OpenAI_互換でなければ_chat_path_を疑う():
    r = check_minimal_request(FakeTransport(post_responses=[(200, json.dumps({"x": 1}))]), Config())
    assert r.status == STATUS_ERROR and "chat_path" in r.summary


def test_手順2_接続できなければ失敗():
    r = check_minimal_request(FakeTransport(post_responses=[TransportError("切断")]), Config())
    assert r.status == STATUS_ERROR and "送れませんでした" in r.summary


# --- 手順3: スキーマ適合の実測（3分類） --------------------------------------------------------


def test_手順3a_400_ならサーバーは対応していない():
    body = '{"error": "response_format is not supported"}'
    r = check_schema_enforcement(FakeTransport(post_responses=[(400, body)]), Config())
    assert r.status == STATUS_WARN
    assert r.detail["classification"] == SCHEMA_UNSUPPORTED
    assert "(a)" in r.summary and "false のまま" in r.summary


def test_手順3b_200_で適合すれば_true_を推奨():
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON))])
    r = check_schema_enforcement(transport, Config())
    assert r.status == STATUS_OK
    assert r.detail["classification"] == SCHEMA_ENFORCED
    assert "enforces_json_schema = true" in r.summary
    # response_format を付けて投げている
    fmt = transport.posts[0]["payload"]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True


def test_手順3c_200_だが不適合なら受け付けるが保証しない():
    body = chat("結果:\n" + GOOD_JSON)  # 修復が要る = 保証していない
    r = check_schema_enforcement(FakeTransport(post_responses=[(200, body)]), Config())
    assert r.status == STATUS_WARN
    assert r.detail["classification"] == SCHEMA_ACCEPTED_NOT_ENFORCED
    assert "(c)" in r.summary and "保証しない" in r.summary


def test_手順3c_200_だが読めない場合も保証しない扱い():
    r = check_schema_enforcement(FakeTransport(post_responses=[(200, chat("だめ"))]), Config())
    assert r.detail["classification"] == SCHEMA_ACCEPTED_NOT_ENFORCED


# --- 手順4: 速度の実測 -------------------------------------------------------------------------


def test_手順4_2つの長さで投げて_tokens_per_sec_を出す():
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON, 12000, 400))], elapsed_sec=8.0)
    r = check_speed(transport, Config())
    assert r.status == STATUS_OK
    measurements = r.detail["measurements"]
    assert len(measurements) == 2 and len(transport.posts) == 2
    assert measurements[0]["output_tokens_per_sec"] == 50.0  # 400 / 8 秒
    assert measurements[0]["input_tokens"] == 12000
    # 2本目は chunk_minutes 相当の長さ
    assert measurements[1]["utterances"] > measurements[0]["utterances"]
    assert "chunk_minutes=25" in measurements[1]["label"]


def test_手順4_usage_が無ければ概算しか分からないと出す():
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON))], elapsed_sec=5.0)
    r = check_speed(transport, Config())
    assert r.status == STATUS_WARN
    assert "usage" in r.summary
    assert r.detail["measurements"][0]["estimated_input_tokens"] > 0


def test_手順4_片方が失敗すれば注意にして続ける():
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON, 100, 10)), (500, "boom")])
    r = check_speed(transport, Config())
    assert r.status == STATUS_WARN
    assert "error" in r.detail["measurements"][1]


def test_手順4_両方失敗すれば失敗():
    r = check_speed(FakeTransport(post_responses=[(500, "boom")]), Config())
    assert r.status == STATUS_ERROR


# --- 手順5: 最大入力長の実測 --------------------------------------------------------------------


def test_手順5_3段階すべて通れば_4倍でも通ったと出す():
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON))])
    r = check_max_input(transport, Config())
    assert r.status == STATUS_OK
    assert len(transport.posts) == len(MAX_INPUT_MULTIPLIERS) == 3
    assert r.detail["all_multipliers_passed"] is True
    assert "4倍でも通りました" in r.summary
    assert r.detail["largest_passed_user_tokens"] == tokens_for_minutes(25) * 4
    # 既定の 25 分では 4 倍が 100 分相当なので、2時間はまだ確かめていないと言う
    assert "2時間" in r.summary and "doctor をもう一度" in r.summary


def test_手順5_2段階目で失敗すれば1段階目までを報告して打ち切る():
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON)), (413, "too large"), (200, chat(GOOD_JSON))])
    r = check_max_input(transport, Config())
    assert r.status == STATUS_WARN
    assert len(transport.posts) == 2  # 3段階目は投げない
    assert r.detail["largest_passed_user_tokens"] == tokens_for_minutes(25)
    assert r.detail["all_multipliers_passed"] is False
    assert "2倍" in r.summary


def test_手順5_3段階目で失敗すれば2段階目までを報告():
    transport = FakeTransport(
        post_responses=[(200, chat(GOOD_JSON)), (200, chat(GOOD_JSON)), (400, "context length exceeded")]
    )
    r = check_max_input(transport, Config())
    assert r.detail["largest_passed_user_tokens"] == tokens_for_minutes(25) * 2
    assert "4倍" in r.summary


def test_手順5_1段階目で失敗すれば失敗():
    r = check_max_input(FakeTransport(post_responses=[(400, "too large")]), Config())
    assert r.status == STATUS_ERROR
    assert r.detail["largest_passed_user_tokens"] is None
    assert "最小の段階" in r.summary


def test_手順5_時間切れも通らなかった扱いにする():
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON)), TransportError("時間切れ")])
    r = check_max_input(transport, Config())
    assert r.status == STATUS_WARN
    assert r.detail["largest_passed_user_tokens"] == tokens_for_minutes(25)


def test_手順5_200_なら出力が壊れていても長さは通った扱いにする():
    # 長さの問題と出力の壊れ方は別の問題なので、分けて報告する
    transport = FakeTransport(post_responses=[(200, chat("壊れた出力"))])
    r = check_max_input(transport, Config())
    assert r.status == STATUS_OK
    assert r.detail["attempts"][0]["classification"] == OUTPUT_UNREADABLE


def test_手順5_chunk_minutes_を大きくすれば2時間を確かめられる():
    cfg = Config()
    cfg.extract.chunk_minutes = 30  # 4 倍で 120 分相当
    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON))])
    r = check_max_input(transport, cfg)
    assert r.status == STATUS_OK
    assert "2時間" not in r.summary  # 120 分相当が通ったので、この注意は出ない


# --- 推奨値 -----------------------------------------------------------------------------------


def full_report(cfg: Config, transport: FakeTransport):
    return run_doctor(transport, cfg, now=NOW)


def test_推奨値_すべて通った場合():
    cfg = Config()
    cfg.extract.chunk_minutes = 30  # 4 倍で 120 分相当 = 2時間を1リクエストで渡せる
    transport = FakeTransport(
        get_response=(200, MODELS_BODY),
        post_responses=[(200, chat(GOOD_JSON, 12000, 400))],
        elapsed_sec=10.0,
    )
    rec = full_report(cfg, transport).recommendations

    assert rec["enforces_json_schema"]["value"] is True  # 200 で適合した
    assert rec["max_input_tokens"]["value"] > tokens_for_minutes(120)
    assert rec["timeout_sec"]["value"] == TIMEOUT_FLOOR_SEC  # 10 秒 × 3 は下限未満
    assert rec["chunk_minutes"]["value"] == WHOLE_MEETING_CHUNK_MINUTES
    assert rec["overlap_minutes"]["value"] == 0
    assert "1リクエストで通った" in rec["chunk_minutes"]["reason"]


def test_推奨値_2時間が通らなければ実測から_chunk_minutes_を算出する():
    cfg = Config()  # chunk_minutes = 25。4 倍でも 100 分相当で 120 分に届かない
    transport = FakeTransport(
        get_response=(200, MODELS_BODY),
        post_responses=[(200, chat(GOOD_JSON, 12000, 400))],
        elapsed_sec=10.0,
    )
    rec = full_report(cfg, transport).recommendations
    # 100 分通った → 8 割の 80 分、5 分単位
    assert rec["chunk_minutes"]["value"] == 80
    assert rec["overlap_minutes"]["value"] == 2  # 分割するので重複は残す
    assert "2時間" in rec["chunk_minutes"]["reason"]


def test_推奨値_スキーマを保証しなければ_false_のまま():
    transport = FakeTransport(
        get_response=(200, MODELS_BODY),
        post_responses=[(200, chat("前置き\n" + GOOD_JSON))],
    )
    rec = full_report(Config(), transport).recommendations
    assert rec["enforces_json_schema"]["value"] is False
    assert "保証しない" in rec["enforces_json_schema"]["reason"]


def test_推奨値_timeout_は実測の3倍を分単位に切り上げる():
    transport = FakeTransport(
        get_response=(200, MODELS_BODY),
        post_responses=[(200, chat(GOOD_JSON, 1000, 100))],
        elapsed_sec=400.0,  # 400 × 3 = 1200 秒
    )
    rec = full_report(Config(), transport).recommendations
    assert rec["timeout_sec"]["value"] == 1200


def test_推奨値_何も測れなければ現在の値のままにして推測値を書かない():
    cfg = Config()
    transport = FakeTransport(get_response=TransportError("切断"), post_responses=[TransportError("切断")])
    report = full_report(cfg, transport)
    rec = report.recommendations

    assert rec["max_input_tokens"]["value"] == 0  # 未確認のまま
    assert rec["timeout_sec"]["value"] == cfg.extract.strata.timeout_sec
    assert rec["chunk_minutes"]["value"] == cfg.extract.chunk_minutes
    assert rec["overlap_minutes"]["value"] == cfg.extract.overlap_minutes
    assert all(c.status == STATUS_ERROR for c in report.checks)


# --- 全体 -------------------------------------------------------------------------------------


def test_run_doctor_は5つの検査を順に行い_途中で失敗しても続ける():
    transport = FakeTransport(
        get_response=TransportError("接続拒否"),  # 手順1 は失敗
        post_responses=[(200, chat(GOOD_JSON))],  # 手順2 以降は成功
    )
    report = full_report(Config(), transport)
    assert [c.key for c in report.checks] == ["models", "minimal", "schema", "speed", "max_input"]
    assert report.get("models").status == STATUS_ERROR
    assert report.get("minimal").status == STATUS_OK  # 手順1 が失敗しても続ける
    assert report.get("max_input").status == STATUS_OK


def test_表と_JSON_に同じ内容が出る():
    transport = FakeTransport(
        get_response=(200, MODELS_BODY), post_responses=[(200, chat(GOOD_JSON, 100, 10))]
    )
    report = full_report(Config(), transport)

    table = format_report(report)
    assert "1. 接続確認" in table and "5. 最大入力長の実測" in table
    assert "config.toml に書くべき推奨値" in table
    for key in ("enforces_json_schema", "max_input_tokens", "timeout_sec", "chunk_minutes", "overlap_minutes"):
        assert key in table

    data = json.loads(report.to_json())
    assert data["model"] == "qwen3.8-flash-next"
    assert data["started_at"] == "2026-11-03T21:30:15"
    assert [c["key"] for c in data["checks"]] == ["models", "minimal", "schema", "speed", "max_input"]
    assert set(data["recommendations"]) == {
        "enforces_json_schema", "max_input_tokens", "timeout_sec", "chunk_minutes", "overlap_minutes",
    }


def test_保存先は_DB_と同じディレクトリの_doctor_配下(cfg):
    path = report_path(cfg, NOW)
    assert path.parent == cfg.paths.db.parent / "doctor"
    assert path.name == "strata_20261103_213015.json"


def test_doctor_は抽出が実際に送るのと同じ形のリクエストを送る():
    """doctor が測るものと extract が送るものが食い違わないことを確かめる。"""
    from kijun.extract.strata_impl import build_payload

    transport = FakeTransport(post_responses=[(200, chat(GOOD_JSON))])
    cfg = Config()
    check_minimal_request(transport, cfg)
    assert transport.posts[0]["payload"] == build_payload(cfg, sample_chunk(), False)
