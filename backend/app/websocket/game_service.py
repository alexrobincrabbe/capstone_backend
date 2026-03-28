from __future__ import annotations

from ..room import RoomManager
from ..room.models import Participant


class GameService:
    def __init__(self, *, room: RoomManager) -> None:
        self.room = room

    async def join_player(self, username: str) -> Participant | None:
        return await self.room.add_participant(username, is_bot=False)

    async def remove_player(self, participant_id: str) -> Participant | None:
        return await self.room.remove_participant(participant_id)

    async def start_round(self) -> bool:
        return await self.room.start_round()

    async def end_round(self) -> bool:
        return await self.room.end_round()

    async def tap(self, participant_id: str) -> bool:
        return await self.room.tap(participant_id)

    async def get_status(self) -> str:
        return await self.room.get_status()

    async def get_round_end_time(self) -> float | None:
        return await self.room.get_round_end_time()

    async def get_participant_count(self) -> int:
        return await self.room.get_participant_count()

    async def get_snapshot(self) -> dict:
        return await self.room.get_snapshot()
