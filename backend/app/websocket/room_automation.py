from __future__ import annotations

from typing import Protocol

from ..bot import BotManager
from .chat_service import ChatService
from .game_service import GameService


class RoomAutomation(Protocol):
    async def startup(self) -> None: ...
    async def on_player_joined(self, username: str) -> None: ...
    async def on_chat_message(
        self,
        sender: str,
        text: str,
        is_round_active: bool,
        participant_count: int,
    ) -> None: ...
    async def on_round_started(self) -> None: ...
    async def on_round_ended(self, room_state: dict | None = None) -> None: ...
    async def on_player_left(self, username: str) -> None: ...


class NoOpRoomAutomation:
    async def startup(self) -> None:
        return

    async def on_player_joined(self, username: str) -> None:
        return

    async def on_chat_message(
        self, sender: str, text: str, is_round_active: bool, participant_count: int
    ) -> None:
        return

    async def on_round_started(self) -> None:
        return

    async def on_round_ended(self, room_state: dict | None = None) -> None:
        return

    async def on_player_left(self, username: str) -> None:
        return

class BotRoomAutomation:
    def __init__(
        self,
        *,
        bot_manager: BotManager,
        chat_service: ChatService,
        game_service: GameService,
    ) -> None:
        self.bot_manager = bot_manager
        self.chat_service = chat_service
        self.game_service = game_service

    async def startup(self) -> None:
        await self.bot_manager.ensure_bots_in_room(self.game_service.room)

    async def on_player_joined(self, username: str) -> None:
        await self.bot_manager.on_participant_joined(
            username,
            send_chat=self.chat_service.post_message,
        )

    async def on_chat_message(
        self, sender: str, text: str, is_round_active: bool, participant_count: int
    ) -> None:
        await self.bot_manager.on_chat_message(
            sender=sender,
            text=text,
            is_round_active=is_round_active,
            participant_count=participant_count,
            send_chat=self.chat_service.post_message,
        )

    async def on_round_started(self) -> None:
        await self.bot_manager.on_round_started(
            room=self.game_service.room,
            tap_by_participant_id=self.game_service.tap,
            send_chat=self.chat_service.post_message,
        )

    async def on_round_ended(self, room_state: dict | None = None) -> None:
        await self.bot_manager.on_round_ended(
            send_chat=self.chat_service.post_message,
            room_state=room_state,
        )

    async def on_player_left(self, username: str) -> None:
        return
