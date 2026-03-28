from __future__ import annotations

import logging
import re

from .llm_client import OpenAIClient
from .models import ChatRoute, RouteDecision
from .prompts import router_system_prompt, router_user_prompt

logger = logging.getLogger("uvicorn.error")


class DeterministicChatRouter:
    def route(self, *, text: str, username: str) -> RouteDecision:
        incoming = text.strip().lower()
        if not incoming:
            return RouteDecision(route=ChatRoute.IGNORE, handled=True)

        if re.search(r"\b(hi|hello|hey|yo)\b", incoming):
            return RouteDecision(
                route=ChatRoute.SIMPLE_REPLY,
                handled=True,
                reply_text=f"Hey {username}. Ready to tap?",
            )
        return RouteDecision(route=ChatRoute.IGNORE, handled=False)


class LLMChatRouter:
    async def classify(
        self,
        *,
        text: str,
        username: str,
        is_round_active: bool,
        recent_turns: list[str],
    ) -> RouteDecision:
        raise NotImplementedError


class OpenAILLMChatRouter(LLMChatRouter):
    def __init__(self, *, client: OpenAIClient, model: str, trace_enabled: bool = False) -> None:
        self.client = client
        self.model = model
        self.trace_enabled = trace_enabled

    async def classify(
        self,
        *,
        text: str,
        username: str,
        is_round_active: bool,
        recent_turns: list[str],
    ) -> RouteDecision:
        if self.trace_enabled:
            logger.info(
                "[bot.trace] router.input model=%s user=%s active=%s text=%r turns=%d",
                self.model,
                username,
                is_round_active,
                text[:200],
                len(recent_turns),
            )
        payload = await self.client.chat_json(
            model=self.model,
            system=router_system_prompt(),
            user=router_user_prompt(
                username=username,
                text=text,
                is_round_active=is_round_active,
                recent_turns=recent_turns[-8:],
                last_round_outcome=None,
            ),
        )
        if not payload:
            if self.trace_enabled:
                logger.info("[bot.trace] router.output empty_payload -> ignore")
            return RouteDecision(route=ChatRoute.IGNORE, handled=True)

        raw_route = str(payload.get("route") or "ignore")
        try:
            route = ChatRoute(raw_route)
        except ValueError:
            route = ChatRoute.IGNORE
        decision = RouteDecision(
            route=route,
            memory_query=payload.get("memory_query"),
            should_store_memory=bool(payload.get("should_store_memory", False)),
        )
        if self.trace_enabled:
            logger.info(
                "[bot.trace] router.output route=%s memory_query=%r store=%s",
                decision.route.value,
                (decision.memory_query or "")[:120],
                decision.should_store_memory,
            )
        return decision