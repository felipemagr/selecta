"""Conversation records."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from selecta.domain.brief import Brief

Language = Literal["en", "es", "fr"]


class ToolCall(BaseModel):
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    """A message in the history."""

    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    name: str | None = None


class ToolSpec(BaseModel):
    """A tool offered to the model."""

    name: str
    description: str
    parameters: dict[str, Any]


class Reply(BaseModel):
    """A model reply."""

    text: str | None = None
    tool_calls: list[ToolCall] = []
    input_tokens: int = 0
    output_tokens: int = 0


class ToolRun(BaseModel):
    """A tool call and its result."""

    call: ToolCall
    result: str


class Pick(BaseModel):
    """The record chosen for the listener, and why."""

    release_id: int
    title: str
    artists: list[str]
    reason: str


class Turn(BaseModel):
    """The trace of one listener turn."""

    started_at: datetime
    user: str
    reply: str
    said_while_waiting: str | None = None
    contexts: list[str] = []
    tool_runs: list[ToolRun] = []
    brief_before: Brief
    brief_after: Brief
    pick_before: Pick | None = None
    pick_after: Pick | None = None
    model_rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None


class Rejection(BaseModel):
    """A pick the listener turned down, in their words."""

    release_id: int
    reason: str


class Summary(BaseModel):
    """The end-of-session summary: key points from code, a short text from the model."""

    key_points: dict[str, Any]
    text: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class Session(BaseModel):
    """A conversation."""

    id: str
    started_at: datetime
    language: Language = "en"
    brief: Brief = Brief()
    pick: Pick | None = None
    rejections: list[Rejection] = []
    summary: Summary | None = None
    messages: list[Message] = []
    turns: list[Turn] = []

    def rejected_ids(self) -> list[int]:
        """Return the ids of the records the listener rejected."""
        return [r.release_id for r in self.rejections]
