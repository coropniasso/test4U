"""議事録の LLM 出力の型（pydantic）。

Ollama の format に渡すスキーマは、このモデルから作る（ollama_api.build_chat_payload が
全項目を required にする）。受け取る側の検証は、既定値のある項目が欠けていても通す
（項目の無い古い JSON も読めるように）。ただし欠けたまま黙って空になるのを避けるため、
Ollama に渡すスキーマでは全項目を必須にしてある。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Topic(BaseModel):
    """議題1つと、その要点。"""

    title: str
    points: list[str] = Field(default_factory=list)


class Todo(BaseModel):
    """ToDo 1件。owner と due は、発言の中で明示されたときだけ書く。無ければ None。"""

    task: str
    owner: str | None = None
    due: str | None = None


class ChunkMinutes(BaseModel):
    """会議の一部（1チャンク）から作った議事録の要素。"""

    topics: list[Topic] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    todos: list[Todo] = Field(default_factory=list)
    open_issues: list[str] = Field(default_factory=list)


class Overview(BaseModel):
    """会議全体の概要（3〜5文）。"""

    overview: str
