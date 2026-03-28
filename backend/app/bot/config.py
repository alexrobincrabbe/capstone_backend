from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Awaitable, Callable

ChatSendFn = Callable[..., Awaitable[None]]
TapFn = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class BotConfig:
    name: str
    personality: str
    # 1..10 (higher = more frequent taps / replies)
    skill_level: int = 6
    recent_message_limit: int = 8
    llm_router_model: str = "gpt-4o-mini"
    llm_response_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    semantic_memory_db_path: str = "bot_memory.db"
    trace_enabled: bool = False

    @staticmethod
    def from_env() -> "BotConfig":
        name = (os.getenv("BOT_NAME") or "TapBot").strip() or "TapBot"
        personality = (os.getenv("BOT_PERSONALITY") or "friendly and hype").strip() or "friendly and hype"
        try:
            skill = int(os.getenv("BOT_SKILL_LEVEL") or "6")
        except ValueError:
            skill = 6
        skill = max(1, min(10, skill))
        try:
            recent_limit = int(os.getenv("BOT_RECENT_MESSAGE_LIMIT") or "8")
        except ValueError:
            recent_limit = 8
        recent_limit = max(2, min(20, recent_limit))
        trace_enabled = (os.getenv("BOT_TRACE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return BotConfig(
            name=name,
            personality=personality,
            skill_level=skill,
            recent_message_limit=recent_limit,
            llm_router_model=(os.getenv("BOT_LLM_ROUTER_MODEL") or "gpt-4o-mini").strip(),
            llm_response_model=(os.getenv("BOT_LLM_RESPONSE_MODEL") or "gpt-4o-mini").strip(),
            embedding_model=(os.getenv("BOT_EMBEDDING_MODEL") or "text-embedding-3-small").strip(),
            semantic_memory_db_path=(os.getenv("BOT_MEMORY_DB_PATH") or "bot_memory.db").strip(),
            trace_enabled=trace_enabled,
        )

