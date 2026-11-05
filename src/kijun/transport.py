"""HTTP の送受信を差し替えられる形にするための薄い層。

推論サーバー（Ollama）に触る処理は、この Protocol の背後に置く。
extract/ollama_impl.py、review/llm.py、doctor.py がこれを使い、テストでは偽の実装を差し込む。

httpx を直接使わずにこの層を挟む理由は2つある。
- Ollama の挙動（スキーマへの適合、入力の切り捨て、モデルの読み込み）は実機でしか確かめられないため、
  判定ロジックを偽の応答で検査できるようにしておく必要がある。
- 所要時間の計測をここに1か所だけ置き、doctor の速度の実測で使えるようにする。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from kijun.deps import import_optional


class TransportError(Exception):
    """応答そのものが得られなかったときの例外（接続できない、時間切れ等）。

    HTTP のエラー応答（400 など）はこの例外にはせず、HttpResponse の status_code で表す。
    サーバーが機能に対応していないこと（400）と、サーバーに届かないこと（接続不可）を
    呼び出し側が区別できるようにするため。
    """


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    text: str
    elapsed_sec: float

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        """本文を JSON として読む。読めなければ TransportError を出す。"""
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as e:
            head = self.text[:200].replace("\n", " ")
            raise TransportError(f"応答が JSON として読めません（{head}）: {e}") from e


class HttpTransport(Protocol):
    def get(self, url: str, timeout_sec: float) -> HttpResponse: ...

    def post(self, url: str, payload: dict, timeout_sec: float) -> HttpResponse: ...


class HttpxTransport:
    """httpx を使う実装。httpx は `uv sync --extra extract` で入る。"""

    def __init__(self) -> None:
        self._httpx = import_optional("httpx", "extract", "推論サーバーへの通信")

    def _wrap(self, call) -> HttpResponse:
        started = time.monotonic()
        try:
            response = call()
        except Exception as e:  # 接続不可、時間切れ、名前解決の失敗をまとめて扱う
            raise TransportError(f"要求を送れませんでした: {e}") from e
        return HttpResponse(
            status_code=response.status_code,
            text=response.text,
            elapsed_sec=time.monotonic() - started,
        )

    def get(self, url: str, timeout_sec: float) -> HttpResponse:
        return self._wrap(lambda: self._httpx.get(url, timeout=timeout_sec))

    def post(self, url: str, payload: dict, timeout_sec: float) -> HttpResponse:
        return self._wrap(lambda: self._httpx.post(url, json=payload, timeout=timeout_sec))
