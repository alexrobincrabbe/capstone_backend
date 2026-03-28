from __future__ import annotations

import random
import time

from .config import BotConfig


class BotReplyPolicy:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self._last_bot_message_ts = 0.0

    def mark_sent(self) -> None:
        self._last_bot_message_ts = time.time()

    def allow_event_announcement(self, min_interval_seconds: float) -> bool:
        return (time.time() - self._last_bot_message_ts) >= min_interval_seconds

    def can_send_deterministic_reply(self, min_interval_seconds: float = 2.0) -> bool:
        return (time.time() - self._last_bot_message_ts) >= min_interval_seconds

    def seconds_since_last_bot_message(self) -> float | None:
        if self._last_bot_message_ts <= 0:
            return None
        return time.time() - self._last_bot_message_ts

    def should_consider_chat_reply(self, is_round_active: bool, *, is_directed: bool) -> bool:
        if is_directed:
            return True
        if (time.time() - self._last_bot_message_ts) < 8.0:
            return False

        base = 0.12
        chat_activity_bonus = 0.018 * self.config.skill_level
        active_bonus = 0.08 if is_round_active else 0.0
        reply_chance = min(0.55, base + chat_activity_bonus + active_bonus)
        return random.random() <= reply_chance
