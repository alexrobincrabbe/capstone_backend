from __future__ import annotations

import json
import logging
import re
from typing import Any

from ...models import ChatRoute, MemoryRetrievalPlan, MemoryRetrievalMode
from ..langchain_integration import get_chat_model
from ..prompts import memory_retrieval_plan_system_prompt, memory_retrieval_plan_user_prompt

logger = logging.getLogger("uvicorn.error")

_VALID_MODES: frozenset[str] = frozenset(
    {"none", "broad_profile", "callback", "specific_fact", "general"}
)


def _coerce_mode(raw: object) -> MemoryRetrievalMode:
    s = str(raw or "").strip().lower()
    if s in _VALID_MODES:
        return s  # type: ignore[return-value]
    return "general"


def _coerce_min_similarity(raw: object) -> float:
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _coerce_max_results_advisory(raw: object, *, use_memory: bool) -> int:
    if not use_memory:
        return 0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 3
    return max(1, n)


def _parse_json_content(content: str) -> dict[str, Any] | None:
    content = content.strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


class MemoryRetrievalPlanService:
    async def plan(
        self,
        *,
        username: str,
        text: str,
        recent_turns: list[str],
        last_round_outcome: dict[str, Any] | str | None,
        route: ChatRoute | None,
    ) -> MemoryRetrievalPlan:
        raise NotImplementedError


class OpenAIMemoryRetrievalPlanService(MemoryRetrievalPlanService):
    def __init__(self, *, model: str, trace_enabled: bool = False) -> None:
        self._model_name = model
        self._trace_enabled = trace_enabled
        self._model = get_chat_model(model=model, temperature=0.0)

    def _format_outcome(self, val: dict[str, Any] | str | None) -> str | None:
        if val is None:
            return None
        if isinstance(val, str):
            return val
        try:
            return json.dumps(val)
        except (TypeError, ValueError):
            return str(val)

    async def plan(
        self,
        *,
        username: str,
        text: str,
        recent_turns: list[str],
        last_round_outcome: dict[str, Any] | str | None,
        route: ChatRoute | None,
    ) -> MemoryRetrievalPlan:
        if self._model is None:
            return self._fallback(
                route=route,
                text=text,
                reason="model_unavailable",
            )

        outcome_txt = self._format_outcome(last_round_outcome)
        human = memory_retrieval_plan_user_prompt(
            username=username,
            text=text,
            recent_turns=recent_turns[-8:],
            last_round_outcome=outcome_txt,
            route=(route.value if route is not None else None),
        )
        payload: dict[str, Any] | None = None
        try:
            try:
                from langchain.schema import SystemMessage, HumanMessage
            except Exception:  # pragma: no cover
                from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=memory_retrieval_plan_system_prompt()),
                HumanMessage(content=human),
            ]
            result = await self._model.ainvoke(messages)
            content = (getattr(result, "content", None) or "").strip()
            if content:
                payload = _parse_json_content(content)
        except Exception as exc:
            if self._trace_enabled:
                logger.info("[bot.trace] memory_retrieval_plan.error %s", exc)
            payload = None

        if not payload:
            return self._fallback(
                route=route,
                text=text,
                reason="empty_or_unparseable_payload",
            )

        use_memory = bool(payload.get("use_memory", payload.get("recall_memory", False)))
        mq_raw = payload.get("query", payload.get("memory_query"))
        query = str(mq_raw).strip() if mq_raw not in (None, "") else None
        mode = _coerce_mode(payload.get("mode", "none" if not use_memory else "general"))
        min_sim = _coerce_min_similarity(payload.get("min_similarity", 0.0))
        max_res = _coerce_max_results_advisory(payload.get("max_results", 3), use_memory=use_memory)

        if use_memory:
            if mode == "none":
                mode = "general"
            if not query:
                query = text.strip()[:500] or None
                fallback_reason = "missing_query_in_planner_payload"
            else:
                fallback_reason = None
        else:
            mode = "none"
            query = None
            min_sim = 0.0
            max_res = 0
            fallback_reason = None

        plan = MemoryRetrievalPlan(
            use_memory=use_memory,
            query=query,
            mode=mode,
            min_similarity=min_sim,
            max_results=max_res,
            plan_source="planner",
            fallback_reason=fallback_reason,
        )
        if self._trace_enabled:
            logger.info(
                "[bot.trace] memory_retrieval_plan output use=%s mode=%s min_sim=%s max=%s q=%r",
                plan.use_memory,
                plan.mode,
                plan.min_similarity,
                plan.max_results,
                (plan.query or "")[:100],
            )
        return plan

    def _fallback(
        self, *, route: ChatRoute | None, text: str, reason: str
    ) -> MemoryRetrievalPlan:
        if route == ChatRoute.MEMORY_REPLY:
            t = text.strip()
            return MemoryRetrievalPlan(
                use_memory=True,
                query=t[:500] if t else None,
                mode="broad_profile",
                min_similarity=0.2,
                max_results=3,
                plan_source="fallback",
                fallback_reason=reason,
            )
        return MemoryRetrievalPlan(
            use_memory=False,
            query=None,
            mode="none",
            min_similarity=0.0,
            max_results=0,
            plan_source="fallback",
            fallback_reason=reason,
        )


class NullMemoryRetrievalPlanService(MemoryRetrievalPlanService):
    async def plan(
        self,
        *,
        username: str,
        text: str,
        recent_turns: list[str],
        last_round_outcome: dict[str, Any] | str | None,
        route: ChatRoute | None,
    ) -> MemoryRetrievalPlan:
        del username, recent_turns, last_round_outcome
        if route == ChatRoute.MEMORY_REPLY:
            t = text.strip()
            return MemoryRetrievalPlan(
                use_memory=True,
                query=t[:500] if t else None,
                mode="broad_profile",
                min_similarity=0.2,
                max_results=3,
                plan_source="fallback",
                fallback_reason="null_planner_service",
            )
        return MemoryRetrievalPlan(
            use_memory=False,
            query=None,
            mode="none",
            min_similarity=0.0,
            max_results=0,
            plan_source="fallback",
            fallback_reason="null_planner_service",
        )
