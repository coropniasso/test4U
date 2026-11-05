"""重い依存を使う実装クラスが、依存が無くても import でき、入れ方を示すエラーを出すことを検査する。

また、偽の HTTP 層を差し込んで、Ollama の実装がリクエストの形と再試行・修復を正しく行うことを
検査する（実際の通信はしない）。
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types

import pytest

from kijun.config import Config
from kijun.deps import MissingDependencyError, import_optional
from kijun.extract.base import ExtractionError
from kijun.extract.chunk import split_into_chunks
from kijun.extract.models import ExtractionResult
from kijun.extract.prompt import build_messages
from kijun.extract.repair import ACTION_REMOVED_PROSE_BEFORE, ACTION_WRAPPED_ARRAY_IN_ITEMS
from kijun.extract.ollama_impl import OllamaExtractor, build_payload
from kijun.transport import TransportError
from kijun.ollama_api import require_all_properties
from tests.conftest import make_utterances
from tests.fakes import FakeOllamaTransport, ollama_body

IMPL_MODULES = [
    "kijun.transcribe.faster_whisper_impl",
    "kijun.transcribe.pyannote_impl",
    "kijun.extract.ollama_impl",
    "kijun.review.llm",
    "kijun.batch.defer",
    "kijun.match.e5_impl",
    "kijun.transport",
    "kijun.doctor",
]
HEAVY = ["torch", "faster_whisper", "pyannote", "sentence_transformers", "httpx"]


def test_実装モジュールは重い依存を_import_せずに_import_できる():
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in IMPL_MODULES)
        + f"heavy = {HEAVY!r}\n"
        + "loaded = [h for h in heavy if h in sys.modules]\n"
        + "assert not loaded, loaded\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_anthropic_に依存しない():
    """抽出から Claude API が消えたので、anthropic は import されない。"""
    code = "import sys\nimport kijun.cli\nimport kijun.extract.ollama_impl\nassert 'anthropic' not in sys.modules\n"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("module", IMPL_MODULES)
def test_import_だけでは_ImportError_にならない(module):
    importlib.import_module(module)


def block(monkeypatch, *names: str) -> None:
    """名前の import を失敗させる（インストールされていない状態を作る）。"""
    for n in names:
        monkeypatch.setitem(sys.modules, n, None)


def test_import_optional_は入れ方を示す(monkeypatch):
    block(monkeypatch, "faster_whisper")
    with pytest.raises(MissingDependencyError) as e:
        import_optional("faster_whisper", "transcribe", "文字起こし")
    assert "uv sync --extra transcribe" in str(e.value)
    assert isinstance(e.value, ImportError)


@pytest.mark.parametrize(
    "factory,extra,heavy",
    [
        (
            lambda c: __import__(
                "kijun.transcribe.faster_whisper_impl", fromlist=["x"]
            ).FasterWhisperTranscriber(c.transcribe),
            "transcribe",
            "faster_whisper",
        ),
        (
            lambda c: __import__("kijun.transcribe.pyannote_impl", fromlist=["x"]).PyannoteDiarizer(
                c.diarize
            ),
            "transcribe",
            "pyannote.audio",
        ),
        (lambda c: OllamaExtractor(c), "extract", "httpx"),
        (
            lambda c: __import__("kijun.match.e5_impl", fromlist=["x"]).E5Embedder(c.match),
            "match",
            "sentence_transformers",
        ),
    ],
)
def test_依存が無ければ_extra_名を含むエラーになる(monkeypatch, factory, extra, heavy):
    block(monkeypatch, heavy)
    with pytest.raises(MissingDependencyError) as e:
        factory(Config())
    assert f"uv sync --extra {extra}" in str(e.value)


def test_HF_TOKEN_が無ければ3つのモデルページへの同意を案内する(monkeypatch):
    fake_pyannote = types.ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = object
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from kijun.transcribe.pyannote_impl import PyannoteDiarizer

    with pytest.raises(RuntimeError) as e:
        PyannoteDiarizer(Config().diarize)
    msg = str(e.value)
    assert "HF_TOKEN" in msg
    for url in PYANNOTE_PAGE_URLS:
        assert url in msg
    assert "4.x" in msg


PYANNOTE_PAGE_URLS = [
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
    "https://huggingface.co/pyannote/speaker-diarization-community-1",
]


def test_モデル読み込みが失敗したときも3つのURLを案内する(monkeypatch):
    class FakePipeline:
        @staticmethod
        def from_pretrained(name, token=None):
            raise OSError("Cannot access gated repo (403)")

    fake_pyannote = types.ModuleType("pyannote.audio")
    fake_pyannote.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    from kijun.transcribe.pyannote_impl import PyannoteDiarizer

    with pytest.raises(RuntimeError) as e:
        PyannoteDiarizer(Config().diarize)
    msg = str(e.value)
    for url in PYANNOTE_PAGE_URLS:
        assert url in msg
    assert "4.x" in msg


# --- Ollama ---------------------------------------------------------------------------------

ITEM = {
    "seq_from": 0,
    "seq_to": 0,
    "speaker_from_label": "SPEAKER_00",
    "speaker_to_label": None,
    "counterpart_role": None,
    "artifact_type": None,
    "quote": "結論を先に書いて",
    "request_summary": "結論を先に書くことを求めた",
    "stated_reason": None,
    "applies_when": None,
    "reusable": True,
    "pass_criterion": None,
}
EMPTY = json.dumps({"items": []})
ONE_ITEM = json.dumps({"items": [ITEM]}, ensure_ascii=False)

LIMIT = Config().extract.ollama.num_ctx - Config().extract.ollama.num_predict  # 8192


def a_chunk():
    return split_into_chunks(make_utterances(5), 12, 1)[0]


def ok(content: str, **kw) -> tuple[int, str]:
    return 200, ollama_body(content, **kw)


def test_Ollama_のリクエストの形():
    transport = FakeOllamaTransport([ok(EMPTY)])
    cfg = Config()
    ex = OllamaExtractor(cfg, transport)
    assert ex.model_name == "qwen3.5:9b"
    assert ex.enforces_json_schema is True
    assert ex.max_input_tokens == LIMIT  # 入力に使える上限は num_ctx - num_predict
    assert ex.extract(a_chunk()).items == []

    post = transport.posts[0]
    # OpenAI 互換の /v1/chat/completions ではなく、num_ctx と keep_alive を送れるネイティブ API を使う
    assert post["url"] == "http://127.0.0.1:11434/api/chat"
    assert post["timeout_sec"] == 1800
    payload = post["payload"]
    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "0"
    assert payload["options"] == {"temperature": 0, "num_ctx": 16384, "num_predict": 8192}
    # 既定値のある項目もモデルに必ず出させるため、全項目を required にしたスキーマを送る
    assert payload["format"] == require_all_properties(ExtractionResult.model_json_schema())
    assert "transcription_unclear" in payload["format"]["$defs"]["ExtractedRequest"]["required"]
    assert "think" not in payload  # None は送らない(モデルの既定に任せる)
    assert set(payload) == {"model", "messages", "stream", "keep_alive", "options", "format"}
    system, user = build_messages(a_chunk(), cfg)
    assert payload["messages"] == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def test_build_payload_は抽出が送るものと同じ形を返す():
    # doctor.py と scripts/measure_extraction.py が同じ関数を使う。
    transport = FakeOllamaTransport([ok(EMPTY)])
    cfg = Config()
    OllamaExtractor(cfg, transport).extract(a_chunk())
    assert build_payload(cfg, a_chunk()) == transport.posts[0]["payload"]


def test_プロンプトの文面は_prompt_py_のものをそのまま使う():
    transport = FakeOllamaTransport([ok(EMPTY)])
    cfg = Config()
    OllamaExtractor(cfg, transport).extract(a_chunk())
    system = transport.posts[0]["payload"]["messages"][0]["content"]
    assert system == build_messages(a_chunk(), cfg)[0]
    assert "推測" in system and "Yes か No で判定できる" in system


def test_スキーマ適合を保証しないと設定したときは_format_を付けない():
    cfg = Config()
    cfg.extract.ollama.enforces_json_schema = False
    transport = FakeOllamaTransport([ok(EMPTY)])
    ex = OllamaExtractor(cfg, transport)
    assert ex.enforces_json_schema is False
    ex.extract(a_chunk())
    assert "format" not in transport.posts[0]["payload"]


@pytest.mark.parametrize("value", [True, False])
def test_think_は設定したときだけ送る(value):
    cfg = Config()
    cfg.extract.ollama.think = value
    transport = FakeOllamaTransport([ok(EMPTY)])
    OllamaExtractor(cfg, transport).extract(a_chunk())
    assert transport.posts[0]["payload"]["think"] is value


def test_推論の文字数を標準エラーに出す_本文は保存しない(capsys):
    transport = FakeOllamaTransport([ok(EMPTY, thinking="あ" * 321)])
    cfg = Config()
    cfg.extract.ollama.think = True
    ex = OllamaExtractor(cfg, transport)
    ex.extract(a_chunk())
    err = capsys.readouterr().err
    assert "推論 321 文字" in err
    assert "あああ" not in err
    assert ex.last_usage == {"prompt_tokens": 100, "completion_tokens": 20}


def test_num_ctx_num_predict_keep_alive_は設定の値を送る():
    cfg = Config()
    cfg.extract.ollama.num_ctx = 12000
    cfg.extract.ollama.num_predict = 2048
    cfg.extract.ollama.keep_alive = "5m"
    transport = FakeOllamaTransport([ok(EMPTY)])
    ex = OllamaExtractor(cfg, transport)
    assert ex.max_input_tokens == 12000 - 2048
    ex.extract(a_chunk())
    payload = transport.posts[0]["payload"]
    assert payload["options"]["num_ctx"] == 12000 and payload["options"]["num_predict"] == 2048
    assert payload["keep_alive"] == "5m"


def test_base_url_は設定で変えられる():
    cfg = Config()
    cfg.extract.ollama.base_url = "http://192.168.1.5:11434/"
    transport = FakeOllamaTransport([ok(EMPTY)])
    OllamaExtractor(cfg, transport).extract(a_chunk())
    assert transport.posts[0]["url"] == "http://192.168.1.5:11434/api/chat"


# --- 出力が切れた・入力が切り捨てられた場合 --------------------------------------------------------


def test_done_reason_が_length_なら標準エラーに出し_読めれば例外にはしない(capsys):
    transport = FakeOllamaTransport([ok('{"items": []}', done_reason="length")])
    assert OllamaExtractor(Config(), transport).extract(a_chunk()).items == []
    err = capsys.readouterr().err
    assert "num_predict に達して切れた" in err and "extract.ollama.num_predict" in err


def test_done_reason_が_stop_なら警告しない(capsys):
    OllamaExtractor(Config(), FakeOllamaTransport([ok('{"items": []}')])).extract(a_chunk())
    assert "num_predict に達して" not in capsys.readouterr().err


def test_num_predict_で切れて1件も読めなければ_投げ直さずに失敗する():
    # think を有効にしたモデルでは、推論も num_predict に数えられる。推論の途中で切れると content が空になる。
    transport = FakeOllamaTransport([ok("", done_reason="length", thinking="考え中")])
    with pytest.raises(ExtractionError, match="num_predict") as e:
        OllamaExtractor(Config(), transport).extract(a_chunk())
    assert "推論も num_predict に数えられ" in str(e.value) and "していません" in str(e.value)
    assert len(transport.posts) == 1  # 2回目は投げない


def test_num_predict_で切れても1件でも読めれば_修復して返す():
    broken = '{"items": [' + json.dumps(ITEM, ensure_ascii=False) + ', {"seq_from": 1, "quo'
    transport = FakeOllamaTransport([ok(broken, done_reason="length")])
    result = OllamaExtractor(Config(), transport).extract(a_chunk())
    assert len(result.items) == 1
    assert len(transport.posts) == 1


def test_content_が_null_なら読めない出力として扱う():
    body = ollama_body(None, done_reason="stop")
    transport = FakeOllamaTransport([(200, body), (200, body)])
    with pytest.raises(ExtractionError, match="2 回とも"):
        OllamaExtractor(Config(), transport).extract(a_chunk())


def test_入力が上限に達したら_ExtractionError_で_投げ直さない():
    # Ollama は入力が文脈長を超えると、エラーを出さずに先頭を捨てる。捨てられた入力から抽出すると
    # 捨てられた発話が静かに消える(原則2-4「失うより止める」)。
    transport = FakeOllamaTransport([ok(EMPTY, prompt_eval_count=LIMIT)])
    with pytest.raises(ExtractionError) as e:
        OllamaExtractor(Config(), transport).extract(a_chunk())
    msg = str(e.value)
    assert str(LIMIT) in msg and "先頭を捨てる" in msg and "chunk_minutes" in msg and "num_ctx" in msg
    assert len(transport.posts) == 1


def test_入力が上限に達する直前なら読める():
    transport = FakeOllamaTransport([ok(EMPTY, prompt_eval_count=LIMIT - 1)])
    assert OllamaExtractor(Config(), transport).extract(a_chunk()).items == []


def test_prompt_eval_count_が返らなければ確認できないことを出して続ける(capsys):
    transport = FakeOllamaTransport([ok(EMPTY, prompt_eval_count=None, eval_count=None)])
    assert OllamaExtractor(Config(), transport).extract(a_chunk()).items == []
    err = capsys.readouterr().err
    assert "確認できません" in err and "トークン数は記録できません" in err


# --- 修復と再試行 -----------------------------------------------------------------------------


def test_前置きつきの出力を修復して読み_修復の記録を残す(capsys):
    transport = FakeOllamaTransport(["はい、結果です。\n" + ONE_ITEM])
    ex = OllamaExtractor(Config(), transport)
    result = ex.extract(a_chunk())

    assert len(result.items) == 1
    assert len(ex.repair_events) == 1
    event = ex.repair_events[0]
    assert event.actions == [ACTION_REMOVED_PROSE_BEFORE]
    assert event.attempt == 1 and event.truncated_items == 0
    assert event.seq_from == 0
    assert "修復しました" in capsys.readouterr().err


def test_format_を付けていても_repair_を通す():
    # enforces_json_schema が真でも、出力は必ず修復を通してから検証する。
    cfg = Config()
    assert cfg.extract.ollama.enforces_json_schema is True
    ex = OllamaExtractor(cfg, FakeOllamaTransport(["```json\n" + ONE_ITEM + "\n```"]))
    assert len(ex.extract(a_chunk()).items) == 1
    assert ex.repair_events


def test_トップレベルが配列の出力も読める():
    transport = FakeOllamaTransport([json.dumps([ITEM], ensure_ascii=False)])
    ex = OllamaExtractor(Config(), transport)
    assert len(ex.extract(a_chunk()).items) == 1
    assert ex.repair_events[0].actions == [ACTION_WRAPPED_ARRAY_IN_ITEMS]


def test_修復が要らなかったときは記録を残さない():
    ex = OllamaExtractor(Config(), FakeOllamaTransport([EMPTY]))
    ex.extract(a_chunk())
    assert ex.repair_events == []


def test_末尾が切れた出力は捨てた件数を記録する():
    full = json.dumps({"items": [ITEM, ITEM]}, ensure_ascii=False)
    cut = full[: full.rindex("{") + 30]
    ex = OllamaExtractor(Config(), FakeOllamaTransport([cut]))
    assert len(ex.extract(a_chunk()).items) == 1
    assert ex.repair_events[0].truncated_items == 1


def test_読めない出力は1回だけ投げ直す(capsys):
    transport = FakeOllamaTransport(["抽出できませんでした", EMPTY])
    ex = OllamaExtractor(Config(), transport)
    assert ex.extract(a_chunk()).items == []
    assert len(transport.posts) == 2
    assert "警告" in capsys.readouterr().err


def test_2回とも読めなければ_ExtractionError():
    transport = FakeOllamaTransport(["だめ1", "だめ2", EMPTY])
    ex = OllamaExtractor(Config(), transport)
    with pytest.raises(ExtractionError, match="2 回とも"):
        ex.extract(a_chunk())
    assert len(transport.posts) == 2  # 3回目は投げない


def test_投げ直しても同じリクエストを送る():
    transport = FakeOllamaTransport(["だめ", EMPTY])
    OllamaExtractor(Config(), transport).extract(a_chunk())
    assert transport.posts[0]["payload"] == transport.posts[1]["payload"]


# --- 失敗の扱い -------------------------------------------------------------------------------


def test_接続できなければ_起動とモデルと_doctor_を案内する():
    transport = FakeOllamaTransport([TransportError("接続拒否")])
    with pytest.raises(ExtractionError) as e:
        OllamaExtractor(Config(), transport).extract(a_chunk())
    msg = str(e.value)
    assert "送れませんでした" in msg and "タスクトレイ" in msg and "ollama list" in msg and "kijun doctor" in msg


def test_HTTP_404_はモデルの取得を案内する():
    transport = FakeOllamaTransport([(404, json.dumps({"error": "model 'qwen3.5:9b' not found"}))])
    with pytest.raises(ExtractionError) as e:
        OllamaExtractor(Config(), transport).extract(a_chunk())
    assert "HTTP 404" in str(e.value) and "ollama pull qwen3.5:9b" in str(e.value)


def test_HTTP_400_で_think_が理由なら_think_を外すよう案内する():
    cfg = Config()
    cfg.extract.ollama.think = True
    transport = FakeOllamaTransport([(400, json.dumps({"error": "qwen3.5:9b does not support thinking"}))])
    with pytest.raises(ExtractionError) as e:
        OllamaExtractor(cfg, transport).extract(a_chunk())
    assert "HTTP 400" in str(e.value) and "extract.ollama.think を書かない" in str(e.value)


def test_think_を送っていなければ_400_でも_think_を外す案内は出さない():
    transport = FakeOllamaTransport([(400, json.dumps({"error": "bad request: think"}))])
    with pytest.raises(ExtractionError) as e:
        OllamaExtractor(Config(), transport).extract(a_chunk())
    assert "HTTP 400" in str(e.value) and "extract.ollama.think を書かない" not in str(e.value)


def test_HTTP_エラーは状態コードと本文を出す():
    transport = FakeOllamaTransport([(500, json.dumps({"error": "boom"}))])
    with pytest.raises(ExtractionError) as e:
        OllamaExtractor(Config(), transport).extract(a_chunk())
    assert "HTTP 500" in str(e.value) and "boom" in str(e.value)


def test_応答の形が_Ollama_の_api_chat_でなければ_base_url_を疑うよう案内する():
    # OpenAI 互換の形(choices)が返ってきた場合も、Ollama の形ではない。
    transport = FakeOllamaTransport([(200, json.dumps({"choices": [{"message": {"content": "x"}}]}))])
    with pytest.raises(ExtractionError) as e:
        OllamaExtractor(Config(), transport).extract(a_chunk())
    assert "base_url" in str(e.value) and "message" in str(e.value)


def test_応答が_JSON_でなければ_ExtractionError():
    transport = FakeOllamaTransport([(200, "<html>404</html>")])
    with pytest.raises(ExtractionError):
        OllamaExtractor(Config(), transport).extract(a_chunk())


def test_使用量があればトークン数と速度を出す(capsys):
    transport = FakeOllamaTransport([ok(EMPTY, prompt_eval_count=1234, eval_count=56, eval_sec=2.0)])
    OllamaExtractor(Config(), transport).extract(a_chunk())
    err = capsys.readouterr().err
    assert "1234" in err and "56" in err and "28.0 tokens/s" in err


def test_last_usage_に直近の使用量を残す():
    transport = FakeOllamaTransport([ok(EMPTY, prompt_eval_count=2325, eval_count=1305)])
    ex = OllamaExtractor(Config(), transport)
    assert ex.last_usage is None
    ex.extract(a_chunk())
    assert ex.last_usage == {"prompt_tokens": 2325, "completion_tokens": 1305}


def test_run_label_が_model_name_に付く():
    # 設定を変えた2回の抽出を同じ会議の中に並べて残すための印。これが無いと2回目の extract が
    # 1回目の行を削除し、kijun compare が比較できない。
    cfg = Config()
    cfg.extract.run_label = "think"
    transport = FakeOllamaTransport([ok(EMPTY)])
    ex = OllamaExtractor(cfg, transport)
    assert ex.model_name == "qwen3.5:9b+think"
    ex.extract(a_chunk())
    # API に送るモデルIDには印を付けない(Ollama が知らない名前になる)。
    assert transport.posts[0]["payload"]["model"] == "qwen3.5:9b"


def test_run_label_が空なら_model_name_はモデル名のまま():
    assert OllamaExtractor(Config(), FakeOllamaTransport([])).model_name == "qwen3.5:9b"


def test_match_が使う_model_name_は_extract_が書くものと一致する():
    # 食い違うと kijun match が「照合の対象がありません」になり、原因が分かりにくい。
    from kijun.cli import Provider, _provider_model

    cfg = Config()
    cfg.extract.run_label = "think"
    _p, model_name = _provider_model(cfg, Provider.ollama)
    assert model_name == OllamaExtractor(cfg, FakeOllamaTransport([])).model_name


# --- faster-whisper の VAD パラメータ ---------------------------------------------------------
#
# 既定のままでは区間が切れないことが実機で分かった（2026-11-04。34.2分の通話で 564.8 秒の区間が
# 1つ出て、中身は3文字だった）。max_utterance_sec は結合を抑えるだけで分割はしないので、
# 区間の長さを抑えるのは VAD 側である。


def test_VAD_パラメータは設定から作られる():
    from kijun.transcribe.faster_whisper_impl import build_vad_parameters

    cfg = Config().transcribe
    got = build_vad_parameters(cfg)
    assert got["threshold"] == cfg.vad_threshold
    assert got["min_silence_duration_ms"] == cfg.vad_min_silence_duration_ms
    assert got["speech_pad_ms"] == cfg.vad_speech_pad_ms
    # 既定で上限が入ること。これが入らないと 564.8 秒の区間が再発する
    assert got["max_speech_duration_s"] == 30.0


def test_上限を_0_にすると_max_speech_duration_s_を渡さない():
    from kijun.transcribe.faster_whisper_impl import build_vad_parameters

    cfg = Config().transcribe
    cfg.vad_max_speech_duration_sec = 0
    assert "max_speech_duration_s" not in build_vad_parameters(cfg)


def test_その版が受け付けない項目は渡さない():
    # faster-whisper の VadOptions は版によって項目が増減する。知らない名前を渡すと TypeError。
    from kijun.transcribe.faster_whisper_impl import build_vad_parameters

    cfg = Config().transcribe
    got = build_vad_parameters(cfg, accepted={"threshold", "speech_pad_ms"})
    assert set(got) == {"threshold", "speech_pad_ms"}


def test_VadOptions_の項目名を取り出す():
    import dataclasses
    import types

    from kijun.transcribe.faster_whisper_impl import vad_option_names

    @dataclasses.dataclass
    class VadOptions:
        threshold: float = 0.5
        max_speech_duration_s: float = float("inf")

    # faster_whisper.vad.VadOptions にある場合
    fw = types.SimpleNamespace(vad=types.SimpleNamespace(VadOptions=VadOptions))
    assert vad_option_names(fw) == {"threshold", "max_speech_duration_s"}
    # faster_whisper.VadOptions にある場合
    assert vad_option_names(types.SimpleNamespace(VadOptions=VadOptions)) == {
        "threshold", "max_speech_duration_s"
    }
    # 見つからない場合は None（絞り込まない）
    assert vad_option_names(types.SimpleNamespace()) is None
    # データクラスでない場合も None
    assert vad_option_names(types.SimpleNamespace(VadOptions=object)) is None


def test_require_all_properties_は既定値のある項目も必須にし_元のスキーマを変えない():
    """2026-11-06 の実測: transcription_unclear が required でないと、qwen3.5:9b はこの項目を出さなかった。"""
    original = ExtractionResult.model_json_schema()
    assert "transcription_unclear" not in original["$defs"]["ExtractedRequest"]["required"]
    strict = require_all_properties(original)
    item = strict["$defs"]["ExtractedRequest"]
    assert set(item["required"]) == set(item["properties"])
    assert strict["required"] == ["items"]
    # 写しを返す(pydantic が返した辞書を書き換えない)
    assert "transcription_unclear" not in original["$defs"]["ExtractedRequest"]["required"]
