from __future__ import annotations

from .models import PlayerStatsSummary


class PlayerStatsService:
    async def get_summary(self, *, username: str) -> PlayerStatsSummary:
        raise NotImplementedError

    async def record_result(self, *, username: str, result: str) -> None:
        raise NotImplementedError


class InMemoryPlayerStatsService(PlayerStatsService):
    def __init__(self) -> None:
        self._results: dict[str, list[str]] = {}

    async def get_summary(self, *, username: str) -> PlayerStatsSummary:
        results = self._results.get(username, [])
        wins = sum(1 for r in results if r == "win")
        losses = sum(1 for r in results if r == "loss")
        return PlayerStatsSummary(
            username=username,
            wins_vs_bot=wins,
            losses_vs_bot=losses,
            recent_results=results[-5:],
        )

    async def record_result(self, *, username: str, result: str) -> None:
        self._results.setdefault(username, []).append(result)
