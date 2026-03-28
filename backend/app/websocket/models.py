from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    sender: str
    text: str
    timestamp: float
    system: bool = False
    # Used by the frontend to render bot messages differently.
    isBot: bool = False


class ClientEvent(BaseModel):
    type: str
    username: str | None = None
    text: str | None = None
