from __future__ import annotations

import time
from typing import Awaitable, Callable
from typing import Any

from .broadcaster import RoomBroadcaster
from .models import ChatMessage

DeadSocketHandler = Callable[[list[str]], Awaitable[None]]


class ChatService:
    def __init__(
        self,
        *,
        broadcaster: RoomBroadcaster,
    ) -> None:
        self.broadcaster = broadcaster
        self._on_dead_sockets: DeadSocketHandler | None = None

    def set_dead_socket_handler(self, handler: DeadSocketHandler) -> None:
        self._on_dead_sockets = handler

    async def post_message(
        self,
        *,
        sender: str,
        text: str,
        is_bot: bool = False,
        system: bool = False,
        trace: list[dict[str, Any]] | None = None,
        trace_source: dict[str, Any] | None = None,
    ) -> None:
        message = ChatMessage(
            sender=sender,
            text=text,
            timestamp=time.time(),
            system=system,
            isBot=is_bot,
            trace=trace,
            traceSource=trace_source,
        )
        dead = await self.broadcaster.broadcast(
            {"type": "chat_message", "message": message.model_dump()}
        )
        if self._on_dead_sockets is not None:
            await self._on_dead_sockets(dead)

    async def post_system_message(self, text: str) -> None:
        await self.post_message(sender="system", text=text, system=True, is_bot=False)

    async def post_trace(
        self,
        *,
        trace_id: str,
        source: dict[str, Any],
        trace: list[dict[str, Any]],
        generated_reply: str | None = None,
    ) -> None:
        dead = await self.broadcaster.broadcast(
            {
                "type": "chat_trace",
                "trace": {
                    "traceId": trace_id,
                    "source": source,
                    "trace": trace,
                    "generatedReply": generated_reply or "",
                    "timestamp": time.time(),
                },
            }
        )
        if self._on_dead_sockets is not None:
            await self._on_dead_sockets(dead)
