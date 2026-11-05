"""テスト用の偽の実装。GPU、ネットワーク、外部プロセスに触らずに、本物と同じインターフェースを満たす。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kijun.discord.base import DiscordError
from kijun.transcribe.base import SpeakerTurn, TranscribedSegment


class FakeTranscriber:
    """渡された区間をそのまま返す。呼ばれた audio_paths を記録する。"""

    def __init__(self, segments: list[TranscribedSegment]) -> None:
        self._segments = segments
        self.calls: list[list[Path]] = []

    def transcribe(self, audio_paths: list[Path]) -> list[TranscribedSegment]:
        self.calls.append(list(audio_paths))
        return list(self._segments)


class FakeDiarizer:
    """渡された話者区間をそのまま返す。"""

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self._turns = turns
        self.calls: list[list[Path]] = []

    def diarize(self, audio_paths: list[Path]) -> list[SpeakerTurn]:
        self.calls.append(list(audio_paths))
        return list(self._turns)


class FakeJsonChat:
    """LLM を呼ばずに、決めた結果を返す偽の JsonChat。

    handler(system, user, model_cls, name) -> model_cls のインスタンス、または Exception。
    handler を渡さないときは、model_cls() の既定値(空)を返す。呼ばれた問い合わせを calls に記録する。
    """

    def __init__(self, handler: Callable | None = None) -> None:
        self._handler = handler
        self.calls: list[tuple[str, str, type, str]] = []
        self.last_reply = None

    def ask(self, system, user, model_cls, name):
        self.calls.append((system, user, model_cls, name))
        result = self._handler(system, user, model_cls, name) if self._handler else model_cls()
        if isinstance(result, Exception):
            raise result
        return result


class FakeDiscordClient:
    """実際の Discord に接続しない偽の DiscordClient。

    - post_file は message_id を "m1", "m2", ... と採番し、posted に (channel_id, content, message_id) を、
      files に message_id -> (filename, data) を記録する。
    - fail_post_when に関数を渡すと、content がその条件を満たす投稿は DiscordError にする。
    """

    def __init__(self, fail_post_when=None) -> None:
        self.posted: list[tuple[str, str, str]] = []
        self.files: dict[str, tuple[str, bytes]] = {}
        self._fail_post_when = fail_post_when

    def post_file(self, channel_id: str, content: str, filename: str, data: bytes) -> str:
        if self._fail_post_when is not None and self._fail_post_when(content):
            raise DiscordError("偽の投稿失敗")
        message_id = f"m{len(self.posted) + 1}"
        self.posted.append((channel_id, content, message_id))
        self.files[message_id] = (filename, data)
        return message_id

    def contents(self) -> list[str]:
        return [c for _, c, _ in self.posted]


def ollama_body(
    content: str | None,
    *,
    thinking: str | None = None,
    done_reason: str | None = "stop",
    prompt_eval_count: int | None = 100,
    eval_count: int | None = 20,
    eval_sec: float = 1.0,
    load_sec: float = 0.5,
    prompt_eval_sec: float = 0.2,
    total_sec: float = 2.0,
) -> str:
    """Ollama の POST /api/chat(stream: false)の応答の本文。None の項目は含めない。"""
    import json

    message: dict = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    body: dict = {"model": "m", "message": message, "done": True}
    if done_reason is not None:
        body["done_reason"] = done_reason
    if prompt_eval_count is not None:
        body["prompt_eval_count"] = prompt_eval_count
    if eval_count is not None:
        body["eval_count"] = eval_count
    body["eval_duration"] = int(eval_sec * 1e9)
    body["load_duration"] = int(load_sec * 1e9)
    body["prompt_eval_duration"] = int(prompt_eval_sec * 1e9)
    body["total_duration"] = int(total_sec * 1e9)
    return json.dumps(body, ensure_ascii=False)


class FakeOllamaTransport:
    """Ollama の /api/chat の応答を順に返す偽の HttpTransport。

    contents の要素が文字列なら、その本文を ollama_body で包んで 200 で返す。Exception なら送出する。
    (status, text) ならそのまま返す。足りなくなったら最後の応答を繰り返す。投げられた payload を記録する。
    """

    def __init__(self, contents: list) -> None:
        from kijun.transport import HttpResponse

        self._response = HttpResponse
        self._contents = list(contents)
        self.posts: list[dict] = []

    def post(self, url: str, payload: dict, timeout_sec: float):
        self.posts.append({"url": url, "payload": payload, "timeout_sec": timeout_sec})
        item = self._contents[0] if len(self._contents) == 1 else self._contents.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return self._response(item[0], item[1], 0.1)
        return self._response(200, ollama_body(item), 0.1)

    def get(self, url: str, timeout_sec: float):  # pragma: no cover - 使われたら誤り
        raise AssertionError("このテストでは GET を使わない")


class FakeCommandRunner:
    """外部コマンドを実行せず、決めた結果を返す偽の CommandRunner。

    handler(command) -> int | tuple[int, str] | None。int は終了コード、tuple は (終了コード, 出力)、
    None は成功(0)。command は、list(kijun の起動)なら ["python", "-m", "kijun.cli", ...] の引数列、
    str(start_command など)ならその文字列。calls に実行順に記録する。
    handler の中で DB を書き換えて、子プロセスの副作用を再現できる。
    """

    def __init__(self, handler=None) -> None:
        from datetime import datetime

        self._handler = handler
        self._datetime = datetime
        self.calls: list[list[str] | str] = []
        self.timeouts: list[float | None] = []

    def run(self, command, timeout_sec=None):
        from kijun.batch.runner import CommandResult

        self.calls.append(command)
        self.timeouts.append(timeout_sec)
        outcome = self._handler(command) if self._handler else None
        code, output = (outcome if isinstance(outcome, tuple) else (0 if outcome is None else outcome, ""))
        now = self._datetime(2026, 11, 4, 23, 30)
        return CommandResult(code, output, now, now)

    def kijun_calls(self) -> list[str]:
        """kijun のサブコマンド(--config を除く)を「ingest」「transcribe --conversation X」の形で返す。"""
        out = []
        for c in self.calls:
            if isinstance(c, list):
                args = c[3:]
                if args[:1] == ["--config"]:
                    args = args[2:]
                out.append(" ".join(args))
        return out

    def commands(self) -> list[str]:
        """実行したコマンドを順に、kijun のものは「kijun ingest」、シェルのものはそのままの形で返す。"""
        out = []
        for c in self.calls:
            if isinstance(c, list):
                args = c[3:]
                if args[:1] == ["--config"]:
                    args = args[2:]
                out.append("kijun " + " ".join(args))
            else:
                out.append(c)
        return out


class FakeProcessLister:
    """呼ばれるたびに、決めた順で「動いているプロセス名」を返す偽の ProcessLister。

    sequence の要素は、名前の集合(小文字に直して返す)か、ProcessListError にする例外。
    足りなくなったら最後の要素を繰り返す。呼ばれた回数を calls に数える。
    """

    def __init__(self, sequence: list) -> None:
        self._sequence = list(sequence)
        self.calls = 0

    def running_names(self) -> set[str]:
        self.calls += 1
        item = self._sequence[0] if len(self._sequence) == 1 else self._sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return {n.lower() for n in item}
