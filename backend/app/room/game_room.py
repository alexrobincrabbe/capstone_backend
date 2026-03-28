from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .models import Participant

ROUND_DURATION_SECONDS = 20


@dataclass
class RoomState:
    participants: dict[str, Participant] = field(default_factory=dict)
    name_to_participant_id: dict[str, str] = field(default_factory=dict)
    scores: dict[str, int] = field(default_factory=dict)  # participant_id -> score
    status: str = "waiting"  # waiting | active | finished
    round_end_time: float | None = None


class RoomManager:
    def __init__(self) -> None:
        self.state = RoomState()
        self._lock = asyncio.Lock()

    async def add_participant(self, name: str, *, is_bot: bool = False) -> Participant | None:
        async with self._lock:
            if name in self.state.name_to_participant_id:
                return None
            participant = Participant(
                id=str(uuid4()),
                name=name,
                is_bot=is_bot,
            )
            self.state.participants[participant.id] = participant
            self.state.name_to_participant_id[participant.name] = participant.id
            self.state.scores.setdefault(participant.id, 0)
            return participant

    async def get_participant_by_name(self, name: str) -> Participant | None:
        async with self._lock:
            participant_id = self.state.name_to_participant_id.get(name)
            if participant_id is None:
                return None
            return self.state.participants.get(participant_id)

    async def remove_participant(self, participant_id: str) -> Participant | None:
        async with self._lock:
            participant = self.state.participants.pop(participant_id, None)
            if participant is None:
                return None
            self.state.name_to_participant_id.pop(participant.name, None)
            self.state.scores.pop(participant.id, None)
            return participant

    async def start_round(self) -> bool:
        async with self._lock:
            if self.state.status == "active":
                return False
            self.state.status = "active"
            self.state.round_end_time = time.time() + ROUND_DURATION_SECONDS
            for participant_id in self.state.scores:
                self.state.scores[participant_id] = 0
            return True

    async def end_round(self) -> bool:
        async with self._lock:
            if self.state.status != "active":
                return False
            self.state.status = "finished"
            self.state.round_end_time = None
            return True

    async def tap(self, participant_id: str) -> bool:
        async with self._lock:
            if self.state.status != "active":
                return False
            if participant_id not in self.state.scores:
                return False
            self.state.scores[participant_id] += 1
            return True

    async def get_snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return self._snapshot_locked()

    async def get_round_end_time(self) -> float | None:
        async with self._lock:
            return self.state.round_end_time

    async def get_status(self) -> str:
        async with self._lock:
            return self.state.status

    async def get_participant_count(self) -> int:
        async with self._lock:
            return len(self.state.participants)

    def _snapshot_locked(self) -> dict[str, Any]:
        participants = sorted(self.state.participants.values(), key=lambda x: x.name)
        return {
            "players": [
                {"id": p.id, "name": p.name, "isBot": p.is_bot} for p in participants
            ],
            # Preserve current frontend behavior: name keyed scores.
            "scores": {p.name: self.state.scores.get(p.id, 0) for p in participants},
            "status": self.state.status,
            "roundEndTime": self.state.round_end_time,
        }
