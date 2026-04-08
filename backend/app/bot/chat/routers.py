from __future__ import annotations

import json
import logging
import re
from typing import Any

from .langchain_integration import get_chat_model
from ..models import ChatRoute, RouteDecision
from .graph import collect_known_player_names
from .prompts import router_system_prompt, router_user_prompt

logger = logging.getLogger("uvicorn.error")

_LEGACY_ROUTE_ALIASES: dict[str, str] = {
    "memory_update_and_reply": "simple_reply",
    "full_history_reply": "detailed_reply",
}

_TARGETING_HINTS: frozenset[str] = frozenset({"directed", "not_for_bot", "ambiguous"})

# When directed_at_bot is True, these ignore_reasons contradict targeting / small-room rules (single LLM call — coerce).
_AUDIENCE_ONLY_IGNORE_REASONS: frozenset[str] = frozenset(
    {
        "not_clearly_to_bot",
        "not_clearly_directed_at_bot",
        "no_clearly_directed_at_bot",
        "not_directed_to_bot",
        "other_user_targeted",
        "talking_to_someone_else",
        "ambiguous_audience",
    }
)


def _normalize_targeting_hint(raw: str) -> str:
    t = (raw or "").strip().lower()
    return t if t in _TARGETING_HINTS else "ambiguous"


def _norm_ignore_reason_key(s: str | None) -> str:
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")


def _is_audience_only_ignore(reason: str | None) -> bool:
    r = _norm_ignore_reason_key(reason)
    if r in _AUDIENCE_ONLY_IGNORE_REASONS:
        return True
    if "not_clearly" in r and "direct" in r:
        return True
    return False


# Model sometimes ignores clear sign-offs; nudge route without fixing reply text (generator stays context-based).
_FAREWELL_ACK_OVERRIDE_REASONS: frozenset[str] = frozenset(
    {"acknowledgement_only", "acknowledgment_only", "no_response_needed", "low_signal"}
)


def _looks_like_farewell(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(
        re.search(
            r"\b(bye|goodbye|byee+|cya|see\s+ya|see\s+you|gtg|g2g|gotta\s+go|catch\s+you\s+later)\b",
            t,
        )
    )


def _parse_directed_at_bot_field(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        if raw == 1:
            return True
        if raw == 0:
            return False
        return None
    if isinstance(raw, str):
        t = raw.strip().lower()
        if t in ("true", "1", "yes"):
            return True
        if t in ("false", "0", "no"):
            return False
        if t in ("null", "none", ""):
            return None
    return None


def _format_history_excerpt(history: list[Any] | None, *, max_turns: int = 10, max_chars: int = 1600) -> str:
    if not history:
        return "(none)"
    turns = history[-max_turns:]
    lines: list[str] = []
    for t in turns:
        sender = str(getattr(t, "sender", "?"))[:40]
        body = str(getattr(t, "text", "")).strip()[:200]
        role = "bot" if getattr(t, "is_bot", False) else "user"
        lines.append(f"{role} {sender}: {body}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = "…\n" + out[-(max_chars - 2) :]
    return out


def _finalize_route_decision(
    *,
    route: ChatRoute,
    ignore_reason: str | None,
    privacy_blocked: bool,
    need_stats: bool,
    need_history: bool,
    directed_at_bot_from_model: bool | None,
    goodbye_context_from_model: str | None,
    targeting_hint: str,
    participant_count: int,
    route_parse_failed: bool,
    user_text: str = "",
    farewell_piggyback_likely: bool = False,
) -> RouteDecision:
    hint = _normalize_targeting_hint(targeting_hint)

    if hint == "not_for_bot":
        return RouteDecision(
            route=ChatRoute.IGNORE,
            privacy_blocked=False,
            need_stats=False,
            need_history=False,
            ignore_reason="explicitly_addressed_to_another_participant",
            directed_at_bot=False,
            goodbye_context=goodbye_context_from_model,
        )

    directed_at_bot: bool
    if hint == "directed":
        directed_at_bot = True
    elif hint == "ambiguous":
        if participant_count <= 2:
            # Do not assume Tom's "bye!" is for the bot when Alex just said they're leaving (history in prompt).
            if farewell_piggyback_likely:
                directed_at_bot = directed_at_bot_from_model is True
            else:
                directed_at_bot = True
        elif directed_at_bot_from_model is None:
            if farewell_piggyback_likely:
                directed_at_bot = False
            else:
                return RouteDecision(
                    route=ChatRoute.IGNORE,
                    privacy_blocked=privacy_blocked,
                    need_stats=False,
                    need_history=False,
                    ignore_reason="ambiguous_directedness_unset",
                    directed_at_bot=False,
                    goodbye_context=goodbye_context_from_model,
                )
        else:
            directed_at_bot = directed_at_bot_from_model
    else:
        # Defensive: unknown hint after normalize should not occur.
        directed_at_bot = True

    if not directed_at_bot:
        ir = (ignore_reason or "").strip() or "not_clearly_directed_at_bot"
        return RouteDecision(
            route=ChatRoute.IGNORE,
            privacy_blocked=privacy_blocked,
            need_stats=False,
            need_history=False,
            ignore_reason=ir[:220],
            directed_at_bot=False,
            goodbye_context=goodbye_context_from_model,
        )

    r = route
    ir = ignore_reason
    ns, nh = need_stats, need_history
    pb = privacy_blocked

    if r == ChatRoute.IGNORE and not route_parse_failed and not privacy_blocked:
        if _is_audience_only_ignore(ir):
            r = ChatRoute.SIMPLE_REPLY
            ir = None

    if (
        r == ChatRoute.IGNORE
        and not route_parse_failed
        and not privacy_blocked
        and _looks_like_farewell(user_text)
    ):
        ir_key = _norm_ignore_reason_key(ir)
        if ir_key in _FAREWELL_ACK_OVERRIDE_REASONS or ir_key.startswith("acknowledg"):
            r = ChatRoute.SIMPLE_REPLY
            ir = None

    if r != ChatRoute.IGNORE:
        ir = None
    elif route_parse_failed:
        ir = ir or "invalid_route_value"
    elif not ir:
        ir = "unspecified"

    if r == ChatRoute.GAME_STATS_REPLY:
        ns = True
    elif r == ChatRoute.DETAILED_REPLY:
        # Detailed route chooses richer generation style; history breadth is controlled by need_history.
        pass

    return RouteDecision(
        route=r,
        privacy_blocked=pb,
        need_stats=ns,
        need_history=nh,
        ignore_reason=ir,
        directed_at_bot=True,
        goodbye_context=goodbye_context_from_model,
    )


def _normalize_llm_content(raw: Any) -> str:
    """LangChain may return a string or a list of content parts (e.g. multimodal)."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block.get("text", block)))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "".join(parts)
    return str(raw)


def _parse_router_json_payload(content: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output (plain JSON, fenced ```json```, leading prose, etc.)."""
    text = (content or "").strip()
    if not text:
        return None

    def _try_load(s: str) -> dict[str, Any] | None:
        s = s.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    parsed = _try_load(text)
    if parsed:
        return parsed

    for m in re.finditer(r"```(?:json)?\s*\r?\n?(.*?)\r?\n?```", text, re.DOTALL | re.IGNORECASE):
        parsed = _try_load(m.group(1))
        if parsed:
            return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return _try_load(text[start : end + 1])
    return None


def _parse_goodbye_context_field(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    allowed = {"to_bot_or_room", "to_other_user", "leaving_self", "unclear"}
    return s if s in allowed else None


class DeterministicChatRouter:
    def route(self, *, text: str, username: str) -> RouteDecision:
        incoming = text.strip().lower()
        if not incoming:
            return RouteDecision(
                route=ChatRoute.IGNORE,
                ignore_reason="empty_message",
                directed_at_bot=None,
            )

        if re.search(r"\b(hi|hello|hey|yo)\b", incoming):
            return RouteDecision(route=ChatRoute.SIMPLE_REPLY, directed_at_bot=None)
        return RouteDecision(
            route=ChatRoute.IGNORE,
            ignore_reason="no_greeting_match",
            directed_at_bot=None,
        )


class LLMChatRouter:
    async def classify(
        self,
        *,
        text: str,
        username: str,
        is_round_active: bool,
        recent_turns: list[str],
        last_round_outcome: dict[str, Any] | str | None = None,
        participant_count: int = 0,
        bot_name: str = "",
        targeting_hint: str = "ambiguous",
        history: list[Any] | None = None,
        participant_names: list[str] | None = None,
        farewell_piggyback_likely: bool = False,
    ) -> RouteDecision:
        raise NotImplementedError


class OpenAILLMChatRouter(LLMChatRouter):
    def __init__(self, *, client=None, model: str, trace_enabled: bool = False) -> None:
        # Supports LangChain path primarily
        del client
        self.model_name = model
        self.trace_enabled = trace_enabled
        self._model = get_chat_model(model=model, temperature=0.1)

    def _format_outcome_for_prompt(self, last_round_outcome: dict[str, Any] | str | None) -> str | None:
        if last_round_outcome is None:
            return None
        if isinstance(last_round_outcome, str):
            return last_round_outcome
        try:
            return json.dumps(last_round_outcome)
        except (TypeError, ValueError):
            return str(last_round_outcome)

    async def classify(
        self,
        *,
        text: str,
        username: str,
        is_round_active: bool,
        recent_turns: list[str],
        last_round_outcome: dict[str, Any] | str | None = None,
        participant_count: int = 0,
        bot_name: str = "",
        targeting_hint: str = "ambiguous",
        history: list[Any] | None = None,
        participant_names: list[str] | None = None,
        farewell_piggyback_likely: bool = False,
    ) -> RouteDecision:
        hint = _normalize_targeting_hint(targeting_hint)
        if hint == "not_for_bot":
            return RouteDecision(
                route=ChatRoute.IGNORE,
                ignore_reason="explicitly_addressed_to_another_participant",
                directed_at_bot=False,
            )

        ambiguous_mp = hint == "ambiguous" and participant_count > 2
        history_full = list(history or [])
        history_excerpt = _format_history_excerpt(history_full if ambiguous_mp else None)
        known_sorted = sorted(
            collect_known_player_names(
                {"history": history_full, "participant_names": list(participant_names or [])},
                bot_name=bot_name,
            )
        )
        known_players_line = ", ".join(known_sorted) if known_sorted else "(none)"

        if self.trace_enabled:
            logger.info(
                "[bot.trace] router.input model=%s user=%s active=%s hint=%s participants=%d bot=%r "
                "ambiguous_mp=%s text=%r turns=%d history_excerpt_chars=%d",
                self.model_name,
                username,
                is_round_active,
                hint,
                participant_count,
                (bot_name or "")[:80],
                ambiguous_mp,
                text[:200],
                len(recent_turns),
                len(history_excerpt),
            )

        payload: dict[str, Any] | None = None
        raw_model_text = ""
        if self._model is not None:
            try:
                try:
                    from langchain.schema import SystemMessage, HumanMessage  # LC < 0.2
                except Exception:  # pragma: no cover
                    from langchain_core.messages import SystemMessage, HumanMessage  # LC >= 0.2

                outcome_txt = self._format_outcome_for_prompt(last_round_outcome)
                messages = [
                    SystemMessage(content=router_system_prompt()),
                    HumanMessage(
                        content=router_user_prompt(
                            username=username,
                            text=text,
                            is_round_active=is_round_active,
                            recent_turns=recent_turns[-8:],
                            last_round_outcome=outcome_txt,
                            participant_count=participant_count,
                            bot_name=bot_name,
                            targeting_hint=hint,
                            history_excerpt=history_excerpt,
                            known_players=known_players_line,
                            farewell_piggyback_likely=farewell_piggyback_likely,
                        )
                    ),
                ]
                result = await self._model.ainvoke(messages)
                raw_model_text = _normalize_llm_content(getattr(result, "content", None)).strip()
                if raw_model_text:
                    payload = _parse_router_json_payload(raw_model_text)
            except Exception:
                payload = None

        if not payload:
            if self.trace_enabled:
                logger.info(
                    "[bot.trace] router.output empty_payload -> ignore (could not parse model JSON)"
                )
                if raw_model_text:
                    logger.info("[bot.trace] router.output raw_preview=%r", raw_model_text[:500])
            return RouteDecision(
                route=ChatRoute.IGNORE,
                ignore_reason="empty_router_payload",
                directed_at_bot=False if ambiguous_mp else None,
            )

        raw_in = str(payload.get("route") or "ignore").strip()
        legacy_memory_update = raw_in == "memory_update_and_reply"
        raw_route = _LEGACY_ROUTE_ALIASES.get(raw_in, raw_in)
        legacy_store_hint = bool(payload.get("should_store_memory", False))

        route_parse_failed = False
        try:
            route = ChatRoute(raw_route)
        except ValueError:
            route = ChatRoute.IGNORE
            route_parse_failed = True

        ir_raw = payload.get("ignore_reason")
        ignore_reason = str(ir_raw).strip()[:220] if ir_raw not in (None, "") else None

        privacy_blocked = bool(payload.get("privacy_blocked", False))
        need_stats = bool(payload.get("need_stats", False))
        need_history = bool(payload.get("need_history", False))

        if legacy_memory_update or legacy_store_hint:
            if route == ChatRoute.IGNORE:
                route = ChatRoute.SIMPLE_REPLY
                ignore_reason = None

        if route == ChatRoute.GAME_STATS_REPLY:
            need_stats = True
        elif route == ChatRoute.DETAILED_REPLY:
            # keep need_history from model output; detailed route does not force full history
            need_history = bool(payload.get("need_history", False))

        directed_raw = _parse_directed_at_bot_field(payload.get("directed_at_bot"))
        goodbye_context = _parse_goodbye_context_field(payload.get("goodbye_context"))

        decision = _finalize_route_decision(
            route=route,
            ignore_reason=ignore_reason,
            privacy_blocked=privacy_blocked,
            need_stats=need_stats,
            need_history=need_history,
            directed_at_bot_from_model=directed_raw,
            goodbye_context_from_model=goodbye_context,
            targeting_hint=hint,
            participant_count=participant_count,
            route_parse_failed=route_parse_failed,
            user_text=text,
            farewell_piggyback_likely=farewell_piggyback_likely,
        )
        if self.trace_enabled:
            logger.info(
                "[bot.trace] router.output route=%s directed_at_bot=%s privacy=%s stats=%s history=%s ignore_reason=%r",
                decision.route.value,
                decision.directed_at_bot,
                decision.privacy_blocked,
                decision.need_stats,
                decision.need_history,
                decision.ignore_reason,
            )
        return decision
