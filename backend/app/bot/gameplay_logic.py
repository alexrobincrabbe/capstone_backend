from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from .config import BotConfig, TapFn


class BotGameplayLogic:
    """Gameplay-only behavior (skill-based tap timing and frequency)."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._last_tap_ts: float = 0.0

    def _now(self) -> float:
        return time.time()

    def _tap_min_interval(self) -> float:
        return max(0.06, 0.25 - (self.config.skill_level * 0.015))

    def _compute_next_tap_delay(self) -> float:
        mean = max(0.12, 1.0 - (self.config.skill_level * 0.06))
        min_delay = max(0.05, mean * 0.35)
        max_delay = min(1.2, mean * 1.35)
        return random.uniform(min_delay, max_delay)

    def _tap_chance(self) -> float:
        chance = 0.35 + (self.config.skill_level * 0.055)
        return min(0.95, max(0.05, chance))

    async def start_tap_loop(
        self,
        *,
        room: Any,
        tap: TapFn,
        stop_event: asyncio.Event,
    ) -> None:
        self._last_tap_ts = 0.0

        while not stop_event.is_set():
            try:
                status = await room.get_status()
            except Exception:
                status = "waiting"

            if status != "active":
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.3)
                except asyncio.TimeoutError:
                    continue
                continue

            delay = self._compute_next_tap_delay()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

            if stop_event.is_set():
                break

            now = self._now()
            if (now - self._last_tap_ts) < self._tap_min_interval():
                continue
            if random.random() > self._tap_chance():
                continue

            await tap()
            self._last_tap_ts = self._now()

