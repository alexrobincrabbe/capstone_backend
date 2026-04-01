from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..memory_db import memory_database_dsn_from_environ

ChatSendFn = Callable[..., Awaitable[None]]
TraceEmitFn = Callable[..., Awaitable[None]]
TapFn = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class BotConfig:
    name: str
    personality: str
    # 1..10 (higher = more frequent taps / replies)
    skill_level: int = 6
    recent_message_limit: int = 8
    context_history_limit: int = 4
    llm_router_model: str = "gpt-4o-mini"
    llm_response_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # Supabase / Postgres URI (see DATABASE_URL in env)
    database_url: str | None = None
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
        try:
            context_history_limit = int(os.getenv("BOT_CONTEXT_HISTORY_LIMIT") or "4")
        except ValueError:
            context_history_limit = 4
        context_history_limit = max(1, min(recent_limit, context_history_limit))
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
            context_history_limit=context_history_limit,
            llm_router_model=(os.getenv("BOT_LLM_ROUTER_MODEL") or "gpt-4o-mini").strip(),
            llm_response_model=(os.getenv("BOT_LLM_RESPONSE_MODEL") or "gpt-4o-mini").strip(),
            embedding_model=(os.getenv("BOT_EMBEDDING_MODEL") or "text-embedding-3-small").strip(),
            database_url=memory_database_dsn_from_environ(),
            trace_enabled=trace_enabled,
        )

