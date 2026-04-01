from __future__ import annotations

from pydantic import BaseModel
from typing import Any


class ChatMessage(BaseModel):
    sender: str
    text: str
    timestamp: float
    system: bool = False
    # Used by the frontend to render bot messages differently.
    isBot: bool = False
    trace: list[dict[str, Any]] | None = None
    traceSource: dict[str, Any] | None = None


class ClientEvent(BaseModel):
    type: str
    username: str | None = None
    text: str | None = None
