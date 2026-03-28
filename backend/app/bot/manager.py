from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .chat_engine import BotChatEngine
from .config import BotConfig, ChatSendFn, TapFn
from .gameplay_logic import BotGameplayLogic


@dataclass(frozen=True)
class BotParticipantBinding:
    participant_id: str
    name: str


class BotController:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.chat = BotChatEngine(config)
        self.gameplay = BotGameplayLogic(config)
        self.binding: BotParticipantBinding | None = None
        self._tap_task: asyncio.Task[None] | None = None
        self._tap_stop_event: asyncio.Event | None = None

    async def on_participant_joined(self, username: str, send_chat: ChatSendFn) -> None:
        await self.chat.on_player_joined(username, send_chat)

    async def on_round_started(self, send_chat: ChatSendFn) -> None:
        await self.chat.on_round_started(send_chat)

    async def on_round_ended(self, send_chat: ChatSendFn, room_state: dict | None = None) -> None:
        await self.chat.on_round_ended(send_chat, room_state=room_state)

    async def on_chat_message(
        self,
        *,
        sender: str,
        text: str,
        is_round_active: bool,
        participant_count: int,
        send_chat: ChatSendFn,
    ) -> None:
        await self.chat.on_chat_message(
            sender=sender,
            text=text,
            is_round_active=is_round_active,
            participant_count=participant_count,
            send_chat=send_chat,
        )

    async def start_tapping(self, *, room, tap: TapFn) -> None:
        if self._tap_task is not None and not self._tap_task.done():
            return
        self._tap_stop_event = asyncio.Event()
        self._tap_task = asyncio.create_task(
            self.gameplay.start_tap_loop(room=room, tap=tap, stop_event=self._tap_stop_event)
        )

    async def stop_tapping(self) -> None:
        if self._tap_stop_event is not None:
            self._tap_stop_event.set()
        if self._tap_task is None:
            return
        try:
            await asyncio.wait_for(self._tap_task, timeout=2.0)
        except asyncio.TimeoutError:
            self._tap_task.cancel()
            try:
                await self._tap_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass
        finally:
            self._tap_task = None
            self._tap_stop_event = None


class BotManager:
    def __init__(self, controllers: list[BotController]) -> None:
        self.controllers = controllers

    async def ensure_bots_in_room(self, room) -> None:
        for controller in self.controllers:
            participant = await room.get_participant_by_name(controller.config.name)
            if participant is None:
                participant = await room.add_participant(controller.config.name, is_bot=True)
            if participant is None:
                continue
            controller.binding = BotParticipantBinding(
                participant_id=participant.id,
                name=participant.name,
            )

    async def on_participant_joined(self, username: str, send_chat: ChatSendFn) -> None:
        for controller in self.controllers:
            await controller.on_participant_joined(username, send_chat)

    async def on_round_started(self, *, room, tap_by_participant_id, send_chat: ChatSendFn) -> None:
        for controller in self.controllers:
            await controller.on_round_started(send_chat)
            if controller.binding is None:
                continue

            async def tap_cb(controller=controller) -> None:
                if controller.binding is None:
                    return
                await tap_by_participant_id(controller.binding.participant_id)

            await controller.start_tapping(room=room, tap=tap_cb)

    async def on_round_ended(self, send_chat: ChatSendFn, room_state: dict | None = None) -> None:
        for controller in self.controllers:
            await controller.stop_tapping()
            await controller.on_round_ended(send_chat, room_state=room_state)

    async def on_chat_message(
        self,
        *,
        sender: str,
        text: str,
        is_round_active: bool,
        participant_count: int,
        send_chat: ChatSendFn,
    ) -> None:
        for controller in self.controllers:
            await controller.on_chat_message(
                sender=sender,
                text=text,
                is_round_active=is_round_active,
                participant_count=participant_count,
                send_chat=send_chat,
            )


def bot_config_from_env() -> BotConfig:
    return BotConfig.from_env()
