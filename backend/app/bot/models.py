from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ChatRoute(str, Enum):
    IGNORE = "ignore"
    SIMPLE_REPLY = "simple_reply"
    MEMORY_REPLY = "memory_reply"
    GAME_STATS_REPLY = "game_stats_reply"
    DETAILED_REPLY = "detailed_reply"
    WEB_REPLY = "web_reply"


@dataclass(frozen=True)
class RouteDecision:
    """Output of the main pre-generation router (`decide`). Memory retrieval is planned in the next graph node."""

    route: ChatRoute
    privacy_blocked: bool = False
    need_stats: bool = False
    need_history: bool = False
    ignore_reason: str | None = None
    # Semantic directedness at the bot (multi-party ambiguous turns are decided here; targeting only prefilters).
    directed_at_bot: bool | None = None
    # Optional router annotation for goodbye disambiguation:
    # "to_bot_or_room", "to_other_user", "leaving_self", "unclear", or None.
    goodbye_context: str | None = None


MemoryRetrievalMode = Literal["none", "broad_profile", "callback", "specific_fact", "general"]


@dataclass(frozen=True)
class MemoryRetrievalPlan:
    """LLM-authored advisory plan; deterministic code enforces caps and executes retrieval."""

    use_memory: bool
    query: str | None = None
    mode: MemoryRetrievalMode = "none"
    min_similarity: float = 0.0
    max_results: int = 1
    plan_source: str = "planner"
    fallback_reason: str | None = None


@dataclass(frozen=True)
class MemoryWriteDecision:
    should_write_memory: bool
    memory_write_text: str | None = None


@dataclass(frozen=True)
class SemanticMemoryRecord:
    username: str
    memory_text: str
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: float | None = None
    similarity: float | None = None


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
    memory_retrieval_mode: MemoryRetrievalMode | None = field(default=None)
    last_round_outcome: dict[str, Any] | str | None = None
