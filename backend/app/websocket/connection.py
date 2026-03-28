from __future__ import annotations

from dataclasses import dataclass

from fastapi import WebSocket


@dataclass
class ClientConnection:
    websocket: WebSocket
    username: str | None = None
    participant_id: str | None = None
