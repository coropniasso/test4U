"""テスト用の偽の実装。GPU、ネットワーク、外部プロセスに触らずに、本物と同じインターフェースを満たす。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from kijun.discord.base import DiscordError, Reaction, Reply
from kijun.extract.base import ExtractionError, RepairEvent
from kijun.db.repo import KnowledgeItemRow
from kijun.extract.chunk import Chunk
from kijun.extract.models import ExtractedRequest, ExtractionResult
from kijun.review.importance import ImportanceVerdict
from kijun.review.llm import ReviewLlmError
from kijun.review.reply import RoutedEntry, RoutedReply
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


class FakeExtractor:
    """チャンクごとに決めた抽出結果を返す。

    responses は、チャンクの番号（chunk.index）から「そのチャンク内の番号で書いた抽出結果」への辞書。
    無いチャンクは空の結果を返す。fail_chunks に入れた番号のチャンクは ExtractionError を出す。
    handler を渡すと、それを優先して使う。

    enforces_json_schema と max_input_tokens は、背面の宣言を変えたときの呼び出し側の挙動を
    テストするための引数である（どちらの値でもテストできるようにしてある）。
    """

    def __init__(
        self,
        responses: dict[int, list[ExtractedRequest]] | None = None,
        fail_chunks: set[int] | None = None,
        model_name: str = "fake-model",
        handler: Callable[[Chunk], ExtractionResult] | None = None,
        enforces_json_schema: bool = False,
        max_input_tokens: int | None = None,
        repair_events: list[RepairEvent] | None = None,
    ) -> None:
        self.model_name = model_name
        self.enforces_json_schema = enforces_json_schema
        self.max_input_tokens = max_input_tokens
        self.repair_events = repair_events if repair_events is not None else []
        self._responses = responses or {}
        self._fail_chunks = fail_chunks or set()
        self._handler = handler
        self.chunks_seen: list[Chunk] = []

    def extract(self, chunk: Chunk) -> ExtractionResult:
        self.chunks_seen.append(chunk)
        if chunk.index in self._fail_chunks:
            raise ExtractionError(f"偽の失敗（チャンク {chunk.index}）")
        if self._handler is not None:
            return self._handler(chunk)
        return ExtractionResult(items=list(self._responses.get(chunk.index, [])))


class FakeEmbedder:
    """文字列のハッシュから決定的にベクトルを作る。

    - 同じ文字列には、必ず同じベクトルを返す（hashlib.md5 を使う。Python の hash() は
      実行ごとに値が変わるので使わない）。
    - 似た文字列には、似たベクトルを返す。文字の2-gram（連続する2文字）を、ハッシュで
      決めた次元に符号付きで加算して作るので、共通する2-gram が多いほど内積が大きくなる。
    - queries と passages は同じ計算にする。呼ばれた文字列を記録するので、再計算の有無を検査できる。
    """

    def __init__(self, dim: int = 256, model_name: str = "fake-embedder") -> None:
        self.dim = dim
        self.model_name = model_name
        self.query_texts: list[str] = []
        self.passage_texts: list[str] = []

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        padded = f"^{text}$"
        for i in range(len(padded) - 1):
            digest = hashlib.md5(padded[i : i + 2].encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            v[index] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self.query_texts.extend(texts)
        return self._encode(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        self.passage_texts.extend(texts)
        return self._encode(texts)

    @property
    def encoded_count(self) -> int:
        return len(self.query_texts) + len(self.passage_texts)


def make_item(
    seq_from: int,
    seq_to: int | None = None,
    quote: str = "引用",
    summary: str = "要約",
    reason: str | None = None,
    applies_when: str | None = None,
    reusable: bool = True,
    criterion: str | None = None,
    speaker_from: str = "SPEAKER_00",
    speaker_to: str | None = "SPEAKER_01",
    artifact_type: str | None = None,
) -> ExtractedRequest:
    """テスト用の ExtractedRequest を作る。"""
    return ExtractedRequest(
        seq_from=seq_from,
        seq_to=seq_from if seq_to is None else seq_to,
        speaker_from_label=speaker_from,
        speaker_to_label=speaker_to,
        counterpart_role=None,
        artifact_type=artifact_type,
        quote=quote,
        request_summary=summary,
        stated_reason=reason,
        applies_when=applies_when,
        reusable=reusable,
        pass_criterion=criterion,
    )


class FakeDiscordClient:
    """実際の Discord に接続しない偽の DiscordClient。

    - post は message_id を "m1", "m2", ... と採番し、posted に (channel_id, content, message_id) を記録する。
    - reactions / replies に message_id をキーにして値を入れると、fetch が返す。
      入れていない message_id は「リアクションも返信も無い」として空のリストを返す。
    - missing に入れた message_id は「取得できなかった(削除された)」として、結果のキーに含めない。
    - fail_post_when に関数を渡すと、content がその条件を満たす投稿は DiscordError にする。
    """

    def __init__(self, fail_post_when=None) -> None:
        self.posted: list[tuple[str, str, str]] = []
        self.reactions: dict[str, list[Reaction]] = {}
        self.replies: dict[str, list[Reply]] = {}
        self.missing: set[str] = set()
        self.fetch_calls: list[str] = []
        self._fail_post_when = fail_post_when

    def post(self, channel_id: str, content: str) -> str:
        if self._fail_post_when is not None and self._fail_post_when(content):
            raise DiscordError("偽の投稿失敗")
        message_id = f"m{len(self.posted) + 1}"
        self.posted.append((channel_id, content, message_id))
        return message_id

    def fetch_reactions(self, channel_id: str, message_ids: list[str]) -> dict[str, list[Reaction]]:
        self.fetch_calls.append("reactions")
        return {
            m: list(self.reactions.get(m, [])) for m in message_ids if m not in self.missing
        }

    def fetch_thread_replies(self, channel_id: str, message_ids: list[str]) -> dict[str, list[Reply]]:
        self.fetch_calls.append("replies")
        return {m: list(self.replies.get(m, [])) for m in message_ids if m not in self.missing}

    def contents(self) -> list[str]:
        return [c for _, c, _ in self.posted]


class FakeImportanceJudge:
    """決めた重要度を返す。fail_for に入れた check_text は ReviewLlmError にする。呼ばれた check_text を記録する。"""

    def __init__(self, importance: int = 2, fail_for: set[str] | None = None, by_text: dict[str, int] | None = None) -> None:
        self._importance = importance
        self._fail_for = fail_for or set()
        self._by_text = by_text or {}
        self.calls: list[tuple[str, list[str]]] = []

    def judge(self, check_text: str, quotes: list[str]) -> ImportanceVerdict:
        self.calls.append((check_text, list(quotes)))
        if check_text in self._fail_for:
            raise ReviewLlmError("偽の判定失敗")
        return ImportanceVerdict(
            importance=self._by_text.get(check_text, self._importance), rationale=f"根拠:{check_text}"
        )


class FakeReplyRouter:
    """返信の原文ごとに決めた振り分けを返す。responses に無い原文は、空の結果を返す。

    fail が True なら、すべて ReviewLlmError にする。呼ばれた (原文, 項目, もう一方の項目) を記録する。
    """

    def __init__(self, responses: dict[str, list[RoutedEntry]] | None = None, fail: bool = False) -> None:
        self._responses = responses or {}
        self.fail = fail
        self.calls: list[tuple[str, KnowledgeItemRow, KnowledgeItemRow | None]] = []

    def route(self, reply_text: str, item: KnowledgeItemRow, other: KnowledgeItemRow | None = None) -> RoutedReply:
        self.calls.append((reply_text, item, other))
        if self.fail:
            raise ReviewLlmError("偽の振り分け失敗")
        return RoutedReply(items=list(self._responses.get(reply_text, [])))


class FakeSpeakerInterpreter:
    """自然な文の返信ごとに決めた対応を返す。呼ばれた原文を記録する。"""

    def __init__(self, responses: dict[str, dict[str, str]] | None = None, fail: bool = False) -> None:
        self._responses = responses or {}
        self.fail = fail
        self.calls: list[str] = []

    def interpret(self, text: str, labels: list[str]) -> dict[str, str]:
        self.calls.append(text)
        if self.fail:
            raise ReviewLlmError("偽の解釈失敗")
        return dict(self._responses.get(text, {}))


class FakeChatTransport:
    """OpenAI 互換の応答(choices[0].message.content)を順に返す偽の HttpTransport。

    contents の要素が文字列なら、その本文を 200 で返す。Exception なら送出する。(status, text) なら
    そのまま返す。足りなくなったら最後の応答を繰り返す。投げられた payload を記録する。
    """

    def __init__(self, contents: list) -> None:
        import json

        from kijun.transport import HttpResponse

        self._json, self._response = json, HttpResponse
        self._contents = list(contents)
        self.posts: list[dict] = []
        self.gets: list[str] = []
        self.get_error: Exception | None = None
        self.get_ready_after: int = 0  # この回数だけ get を失敗させてから成功させる

    def post(self, url: str, payload: dict, timeout_sec: float):
        self.posts.append({"url": url, "payload": payload, "timeout_sec": timeout_sec})
        item = self._contents[0] if len(self._contents) == 1 else self._contents.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return self._response(item[0], item[1], 0.1)
        body = self._json.dumps({"choices": [{"message": {"content": item}}]}, ensure_ascii=False)
        return self._response(200, body, 0.1)

    def get(self, url: str, timeout_sec: float):
        from kijun.transport import TransportError

        self.gets.append(url)
        if self.get_error is not None or len(self.gets) <= self.get_ready_after:
            raise self.get_error or TransportError("まだ応答しません")
        return self._response(200, '{"data": []}', 0.1)
