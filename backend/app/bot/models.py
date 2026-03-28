from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ChatRoute(str, Enum):
    IGNORE = "ignore"
    SIMPLE_REPLY = "simple_reply"
    MEMORY_REPLY = "memory_reply"
    MEMORY_UPDATE_AND_REPLY = "memory_update_and_reply"
    GAME_STATS_REPLY = "game_stats_reply"
    FULL_HISTORY_REPLY = "full_history_reply"
    WEB_REPLY = "web_reply"


@dataclass(frozen=True)
class RouteDecision:
    route: ChatRoute
    handled: bool = False
    reply_text: str | None = None
    memory_query: str | None = None
    should_store_memory: bool = False


@dataclass(frozen=True)
class SemanticMemoryRecord:
    username: str
    memory_text: str
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: float | None = None


@dataclass(frozen=True)
class PlayerStatsSummary:
    username: str
    wins_vs_bot: int = 0
    losses_vs_bot: int = 0
    recent_results: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChatTurn:
    sender: str
    text: str
    is_bot: bool = False


@dataclass(frozen=True)
class ReplyContext:
    username: str
    user_message: str
    route: ChatRoute
    memories: list[SemanticMemoryRecord] = field(default_factory=list)
    stats: PlayerStatsSummary | None = None
    recent_turns: list[ChatTurn] = field(default_factory=list)
