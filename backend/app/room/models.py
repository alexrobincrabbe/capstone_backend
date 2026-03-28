from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Participant:
    id: str
    name: str
    is_bot: bool = False
