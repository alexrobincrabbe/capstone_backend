from __future__ import annotations

from fastapi import WebSocket

from ..room import RoomManager


class RoomBroadcaster:
    def __init__(self, room: RoomManager) -> None:
        self.room = room
        self._connections: dict[str, WebSocket] = {}

    def attach(self, participant_id: str, websocket: WebSocket) -> None:
        self._connections[participant_id] = websocket

    def detach(self, participant_id: str) -> None:
        self._connections.pop(participant_id, None)

    async def broadcast(self, event: dict) -> list[str]:
        dead_participant_ids: list[str] = []
        for participant_id, socket in list(self._connections.items()):
            try:
                await socket.send_json(event)
            except Exception:
                dead_participant_ids.append(participant_id)
        return dead_participant_ids

    async def broadcast_room_state(self) -> list[str]:
        snapshot = await self.room.get_snapshot()
        return await self.broadcast({"type": "room_state", "state": snapshot})
