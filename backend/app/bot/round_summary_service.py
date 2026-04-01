from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chat.memory.semantic import SemanticMemoryService
from .player_stats import PlayerStatsService


@dataclass(frozen=True)
class RoundEndResult:
    summary_text: str
    outcome: dict[str, Any]


class RoundSummaryService:
    """Round-end stats, memories, and short chat summary (no transport)."""

    def __init__(
        self,
        *,
        player_stats: PlayerStatsService,
        semantic_memory: SemanticMemoryService,
        bot_name: str,
    ) -> None:
        self._player_stats = player_stats
        self._semantic_memory = semantic_memory
        self._bot_name = bot_name

    async def _store_round_summary_memory(self, *, username: str) -> None:
        summary = await self._player_stats.get_summary(username=username)
        wins = int(summary.wins_vs_bot or 0)
        losses = int(summary.losses_vs_bot or 0)
        total = wins + losses
        memory_text = (
            f"Results vs {self._bot_name}: wins={wins}, losses={losses}, ties=0, total={total}."
        )
        await self._semantic_memory.store_memory(
            username=username,
            memory_text=memory_text,
            metadata={"source": "round_summary"},
        )

    def _normalize_scores(self, room_state: dict | None) -> dict[str, int]:
        if room_state is None:
            return {}
        scores = room_state.get("scores")
        if not isinstance(scores, dict) or not scores:
            return {}
        normalized: dict[str, int] = {}
        for name, score in scores.items():
            if isinstance(name, str):
                try:
                    normalized[name] = int(score)
                except (TypeError, ValueError):
                    continue
        return normalized

    def _build_summary_text(
        self,
        *,
        bot_score: int | None,
        winners: list[str],
    ) -> str:
        if not winners:
            return "gg"
        if len(winners) == 1:
            sole = winners[0]
            if sole == self._bot_name:
                return "gg"
            return f"gg {sole}"
        return "gg, tie"

    async def handle_round_end(self, room_state: dict | None) -> RoundEndResult:
        bot_name = self._bot_name
        normalized_scores = self._normalize_scores(room_state)

        if not normalized_scores:
            return RoundEndResult(
                summary_text="gg",
                outcome={
                    "bot_name": bot_name,
                    "bot_score": None,
                    "top_score": 0,
                    "winners": [],
                    "scores": {},
                },
            )

        bot_score = normalized_scores.get(bot_name)
        top_score = max(normalized_scores.values())
        winners = sorted(name for name, score in normalized_scores.items() if score == top_score)

        outcome: dict[str, Any] = {
            "bot_name": bot_name,
            "bot_score": bot_score,
            "top_score": top_score,
            "winners": winners,
            "scores": normalized_scores,
        }

        summary_text = self._build_summary_text(bot_score=bot_score, winners=winners)

        if bot_score is not None:
            for name, score in normalized_scores.items():
                if name == bot_name:
                    continue
                if score > bot_score:
                    await self._player_stats.record_result(username=name, result="win")
                elif score < bot_score:
                    await self._player_stats.record_result(username=name, result="loss")
                await self._store_round_summary_memory(username=name)

        return RoundEndResult(summary_text=summary_text, outcome=outcome)
