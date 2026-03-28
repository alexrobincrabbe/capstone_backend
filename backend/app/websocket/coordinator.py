from __future__ import annotations

import asyncio
import time

from .broadcaster import RoomBroadcaster
from .chat_service import ChatService
from .connection import ClientConnection
from .game_service import GameService
from .room_automation import RoomAutomation


class RealtimeRoomCoordinator:
    def __init__(
        self,
        *,
        broadcaster: RoomBroadcaster,
        chat_service: ChatService,
        game_service: GameService,
        automation: RoomAutomation,
    ) -> None:
        self.broadcaster = broadcaster
        self.chat_service = chat_service
        self.game_service = game_service
        self.automation = automation
        self._is_cleaning_dead_sockets = False
        self.chat_service.set_dead_socket_handler(self._cleanup_dead_sockets)

    async def startup(self) -> None:
        await self.automation.startup()
        await self._broadcast_room_state()
        asyncio.create_task(self.round_watcher())

    async def handle_join(self, connection: ClientConnection, username: str | None) -> None:
        candidate = (username or "").strip()
        if not candidate:
            await connection.websocket.send_json({"type": "error", "message": "Username is required"})
            return
        if connection.username is not None:
            await connection.websocket.send_json({"type": "error", "message": "Already joined"})
            return

        participant = await self.game_service.join_player(candidate)
        if participant is None:
            await connection.websocket.send_json({"type": "error", "message": "Username already taken"})
            return

        connection.username = participant.name
        connection.participant_id = participant.id
        self.broadcaster.attach(participant.id, connection.websocket)
        await self._broadcast({"type": "player_joined", "username": participant.name})
        await self._broadcast_room_state()
        await self.chat_service.post_system_message(f"{participant.name} joined the room")
        await self.automation.on_player_joined(participant.name)

    async def handle_chat_message(self, connection: ClientConnection, text: str | None) -> None:
        trimmed = (text or "").strip()
        if not trimmed:
            return
        sender = connection.username or ""
        await self.chat_service.post_message(
            sender=sender,
            text=trimmed,
            is_bot=False,
            system=False,
        )
        await self.automation.on_chat_message(
            sender=sender,
            text=trimmed,
            is_round_active=(await self.game_service.get_status()) == "active",
            participant_count=await self.game_service.get_participant_count(),
        )

    async def handle_start_round(self, connection: ClientConnection) -> None:
        started = await self.game_service.start_round()
        if not started:
            await connection.websocket.send_json({"type": "error", "message": "Round already active"})
            return

        await self._broadcast({"type": "round_started"})
        await self._broadcast_room_state()
        await self.chat_service.post_system_message("Round started")
        await self.automation.on_round_started()

    async def handle_tap(self, connection: ClientConnection) -> None:
        if connection.participant_id is None:
            return
        await self._handle_tap_by_participant_id(connection.participant_id)

    async def handle_disconnect(self, connection: ClientConnection) -> None:
        participant_id = connection.participant_id
        if participant_id is None:
            return

        self.broadcaster.detach(participant_id)
        removed = await self.game_service.remove_player(participant_id)
        if removed is None:
            return

        await self._broadcast({"type": "player_left", "username": removed.name})
        await self._broadcast_room_state()
        await self.chat_service.post_system_message(f"{removed.name} left the room")
        await self.automation.on_player_left(removed.name)

    async def round_watcher(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            status = await self.game_service.get_status()
            if status != "active":
                continue

            end_time = await self.game_service.get_round_end_time()
            if end_time is None:
                continue

            if time.time() >= end_time:
                changed = await self.game_service.end_round()
                if not changed:
                    continue
                snapshot = await self.game_service.get_snapshot()
                await self._broadcast({"type": "round_ended"})
                await self._broadcast_room_state()
                await self.chat_service.post_system_message("Round ended")
                await self.automation.on_round_ended(snapshot)

    async def _handle_tap_by_participant_id(self, participant_id: str) -> None:
        counted = await self.game_service.tap(participant_id)
        if counted:
            await self._broadcast({"type": "scores_updated"})
            await self._broadcast_room_state()

    async def _broadcast(self, event: dict) -> None:
        dead = await self.broadcaster.broadcast(event)
        await self._cleanup_dead_sockets(dead)

    async def _broadcast_room_state(self) -> None:
        dead = await self.broadcaster.broadcast_room_state()
        await self._cleanup_dead_sockets(dead)

    async def _cleanup_dead_sockets(self, dead_participant_ids: list[str]) -> None:
        if not dead_participant_ids or self._is_cleaning_dead_sockets:
            return
        self._is_cleaning_dead_sockets = True
        try:
            for participant_id in set(dead_participant_ids):
                self.broadcaster.detach(participant_id)
                removed = await self.game_service.remove_player(participant_id)
                if removed is None:
                    continue
                await self.broadcaster.broadcast({"type": "player_left", "username": removed.name})
                await self.broadcaster.broadcast_room_state()
                await self.chat_service.post_system_message(f"{removed.name} left the room")
                await self.automation.on_player_left(removed.name)
        finally:
            self._is_cleaning_dead_sockets = False
