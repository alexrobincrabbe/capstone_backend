from __future__ import annotations

import time
from typing import Awaitable, Callable

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
        self, *, sender: str, text: str, is_bot: bool = False, system: bool = False
    ) -> None:
        message = ChatMessage(
            sender=sender,
            text=text,
            timestamp=time.time(),
            system=system,
            isBot=is_bot,
        )
        dead = await self.broadcaster.broadcast(
            {"type": "chat_message", "message": message.model_dump()}
        )
        if self._on_dead_sockets is not None:
            await self._on_dead_sockets(dead)

    async def post_system_message(self, text: str) -> None:
        await self.post_message(sender="system", text=text, system=True, is_bot=False)
