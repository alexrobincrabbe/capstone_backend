from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, TypedDict
import logging
import random
import re

from langgraph.graph import StateGraph, END

from ..models import (
    ChatRoute,
    MemoryRetrievalMode,
    ReplyContext,
    RouteDecision,
    ChatTurn,
)
from .memory.write_decision_service import is_redundant_username_memory

MAX_MEMORY_RETRIEVAL_RESULTS = 3
DEFAULT_CONTEXT_HISTORY_LIMIT = 4


class ChatState(TypedDict, total=False):
    sender: str
    text: str
    is_round_active: bool
    participant_count: int
    participant_names: List[str]
    bot_name: str
    now_ts: float
    last_greet_ts: float
    rejoin_window_seconds: float
    session_started_at: float
    seconds_since_last_bot_message: Optional[float]
    skill_level: int
    context_history_limit: int

    recent_turns: List[str]
    history: List[ChatTurn]

    event_type: Optional[str]
    event_payload: dict[str, Any]
    reply: str

    round_summary: Optional[str]
    round_outcome: Any
    flush_deferred: bool
    join_announcement_blocked: bool

    spam_blocked: bool
    spam_preblocked: bool
    spam_repeat_count: int
    spam_rapid_fire: bool
    target_blocked: bool
    policy_allowed: bool

    # deterministic targeting hints
    is_explicitly_directed: bool
    is_explicitly_not_for_bot: bool
    short_farewell_opener: bool
    recent_other_human_leave: bool
    farewell_piggyback_likely: bool

    # Conversation directed at the bot: policy uses a provisional value; decide_node sets final
    # is_directed using RouteDecision.directed_at_bot when the router classifies ambiguous rooms.
    is_directed: bool
    router_directed_at_bot: Optional[bool]

    # main router outputs
    route: Optional[ChatRoute]
    privacy_blocked: bool
    need_stats: bool
    need_history: bool
    ignore_reason: Optional[str]

    # memory retrieval plan (LLM advisory + coerced fields; execution clamps in gather_context)
    use_memory: bool
    memory_query: Optional[str]
    memory_mode: Optional[str]
    memory_min_similarity: float
    memory_max_results: int
    memory_plan_source: Optional[str]
    memory_plan_fallback_reason: Optional[str]

    # gathered context
    memories: list
    stats: Any
    context_history: List[ChatTurn]

    # post-generation memory write outputs
    should_write_memory: bool
    memory_write_text: Optional[str]
    memory_persisted: bool
    memory_write_skip_reason: Optional[str]

    # bookkeeping / hints for router
    set_last_greet: bool
    last_round_outcome: Any
    trace: list[dict[str, Any]]


@dataclass(frozen=True)
class GraphDeps:
    router: Any  # LLMChatRouter (main route / privacy / stats / history)
    memory: Any  # SemanticMemoryService
    player_stats: Any  # PlayerStatsService
    responder: Any  # BotResponseGenerator
    round_summary: Any  # RoundSummaryService
    memory_retrieval_plan: Any  # MemoryRetrievalPlanService
    memory_write: Any  # MemoryWriteDecisionService


def _is_join_event(text: str) -> bool:
    return text.strip().startswith("EVENT: player_joined")


def _is_round_ended_event(text: str) -> bool:
    return text.strip().startswith("EVENT: round_ended")


def _parse_join_username(state: ChatState) -> str:
    payload = state.get("event_payload") or {}
    username = payload.get("username")
    if username is not None:
        return str(username).strip()

    raw = (state.get("text") or "").strip()
    match = re.search(r"username=(\S+)", raw)
    if match:
        return match.group(1).strip()

    return (state.get("sender") or "").strip()


_JOIN_USERNAME_IN_TEXT = re.compile(r"username=([^\s]+)", re.IGNORECASE)
# Final word before optional end punctuation: "where you from rex?" -> rex (multi-party addressee).
_TRAILING_ADDRESSEE = re.compile(
    r"(?<![\w-])([a-z][\w-]{1,31})\s*[\?\.!]?\s*$",
    re.IGNORECASE,
)


def collect_known_player_names(state: ChatState, *, bot_name: str) -> set[str]:
    """Human names from participant_names plus chat history (senders and join-event usernames)."""
    bot = (bot_name or "").strip().lower()
    names: set[str] = set()
    for n in state.get("participant_names") or []:
        if isinstance(n, str) and n.strip():
            ln = n.strip().lower()
            if ln != bot:
                names.add(ln)
    for turn in list(state.get("history") or []):
        s = (getattr(turn, "sender", "") or "").strip().lower()
        if s and s not in ("system", bot):
            names.add(s)
        txt = getattr(turn, "text", "") or ""
        for m in _JOIN_USERNAME_IN_TEXT.finditer(txt):
            u = m.group(1).strip().lower()
            if u and u != bot:
                names.add(u)
    return names


_LEAVE_OR_FAREWELL_IN_TURN = re.compile(
    r"\b("
    r"I\s*am\s+leav(?:e|ing)|leav(?:e|ing)\b|head(?:ing)?\s+out|"
    r"I\s+have\s+to\s+go|have\s+to\s+go|need\s+to\s+(?:go|run|head)|"
    r"I'?m\s+out|gotta\s+go|gtg|g2g|"
    r"see\s+you\s*(?:all|later|guys|everyone|around)?|see\s*ya|cya|"
    r"good\s*bye|catch\s+you\s+later|take\s+care"
    r")\b",
    re.IGNORECASE,
)

_SHORT_FAREWELL_OPENER = re.compile(
    r"^\s*(bye|goodbye|byee+|cya|see\s+ya|see\s+you|gtg|g2g)\b",
    re.IGNORECASE,
)


def _strip_trailing_turns_from_sender(history: List[ChatTurn], sender_lower: str) -> list[ChatTurn]:
    h: list[ChatTurn] = list(history)
    while h and (getattr(h[-1], "sender", "") or "").strip().lower() == sender_lower:
        h.pop()
    return h


def recent_other_human_signaled_leave(state: ChatState, *, bot_name: str) -> bool:
    """
    Another human (not the current sender) recently had leaving/farewell wording.
    A minimal 'bye!' right after is often for them / the table, not for the bot.
    """
    bot_l = (bot_name or "").strip().lower()
    sender_l = (state.get("sender") or "").strip().lower()
    hist = _strip_trailing_turns_from_sender(list(state.get("history") or []), sender_l)
    for t in reversed(hist[-14:]):
        if getattr(t, "is_bot", False):
            continue
        author = (getattr(t, "sender", "") or "").strip().lower()
        if not author or author in ("system", sender_l, bot_l):
            continue
        if _LEAVE_OR_FAREWELL_IN_TURN.search(getattr(t, "text", "") or ""):
            return True
    return False


def is_short_farewell_opener_line(raw: str) -> bool:
    r = raw.strip()
    return bool(r) and len(r) <= 80 and bool(_SHORT_FAREWELL_OPENER.match(r))


# Bare post-game ack (same channel energy as bot/system gg); replying again is usually noise.
_POST_GAME_ACK_ONLY = re.compile(
    r"^\s*(gg|g\.g\.|good\s*game|nice\s*game|wp|well\s*played)\s*[!\.?,]*\s*$",
    re.IGNORECASE,
)


def recent_bot_or_round_signaled_post_game(history: List[ChatTurn], *, bot_name: str) -> bool:
    """Recent bot line is gg-style, or system marked round end — context for player echo gg."""
    bot_l = (bot_name or "").strip().lower()
    for t in reversed(list(history or [])[-8:]):
        if getattr(t, "is_bot", False):
            if (getattr(t, "sender", "") or "").strip().lower() != bot_l:
                continue
            body = (getattr(t, "text", "") or "").lower()
            if re.search(r"\bgg\b|good\s*game|nice\s*(game|one)|well\s*played", body):
                return True
            continue
        if (getattr(t, "sender", "") or "").strip().lower() == "system":
            tx = (getattr(t, "text", "") or "").lower()
            if "round_ended" in tx or "round ended" in tx:
                return True
    return False


def is_post_round_gg_echo(user_text: str, history: List[ChatTurn], *, bot_name: str) -> bool:
    if not _POST_GAME_ACK_ONLY.match((user_text or "").strip()):
        return False
    return recent_bot_or_round_signaled_post_game(history, bot_name=bot_name)


def build_chat_graph(deps: GraphDeps):
    workflow = StateGraph(ChatState)
    logger = logging.getLogger("uvicorn.error")
    trace_context_by_node: dict[str, str] = {
        "event_gate": "Classifies incoming input as chat, player_joined event, or round_ended event.",
        "player_join": "Builds the greeting reply for join events, considering rejoin timing.",
        "round_end": "Builds round summary and outcome context after a round_ended event.",
        "spam_filter": "Detects repeated same-message spam from the sender and may block/reply.",
        "targeting": "Determines whether the current message appears directed at the bot.",
        "policy": "Applies deterministic/probabilistic prefilter using directedness and cooldown hints.",
        "decide": "Runs main route classification, privacy checks, and directedness resolution.",
        "plan_memory_retrieval": "Plans memory query/mode/limits when retrieval is already required.",
        "gather_context": "Loads memories/stats/history needed for response generation.",
        "generate": "Generates the bot text reply from the assembled ReplyContext.",
        "decide_memory_write": "Decides whether to persist a long-term memory after replying.",
        "persist_memory": "Persists memory to semantic store when write decision is affirmative.",
        "sanitize": "Removes potentially misleading callback language in zero-history contexts.",
        "humanize": "Applies light stylistic variation to make replies feel more natural.",
    }

    def _trace_view(state: ChatState) -> dict[str, Any]:
        keys = (
            "sender",
            "text",
            "event_type",
            "participant_count",
            "is_round_active",
            "is_directed",
            "is_explicitly_directed",
            "is_explicitly_not_for_bot",
            "short_farewell_opener",
            "recent_other_human_leave",
            "farewell_piggyback_likely",
            "target_blocked",
            "policy_allowed",
            "route",
            "ignore_reason",
            "privacy_blocked",
            "need_stats",
            "need_history",
            "use_memory",
            "memory_query",
            "memory_mode",
            "memory_min_similarity",
            "memory_max_results",
            "memory_plan_source",
            "memory_plan_fallback_reason",
            "spam_blocked",
            "spam_preblocked",
            "spam_repeat_count",
            "spam_rapid_fire",
            "join_announcement_blocked",
            "set_last_greet",
            "last_round_outcome",
            "round_outcome",
            "seconds_since_last_bot_message",
            "context_history_limit",
            "should_write_memory",
            "memory_persisted",
            "memory_write_skip_reason",
            "reply",
        )
        out: dict[str, Any] = {}
        for k in keys:
            if k not in state:
                continue
            v = state.get(k)
            if k == "text" and isinstance(v, str):
                out[k] = v[:220]
            elif k == "reply" and isinstance(v, str):
                out[k] = v[:220]
            elif k == "memory_query" and isinstance(v, str):
                out[k] = v[:220]
            elif hasattr(v, "value"):
                out[k] = getattr(v, "value")
            else:
                out[k] = v
        out["memories_count"] = len(state.get("memories") or [])
        out["history_count"] = len(state.get("history") or [])
        out["context_history_count"] = len(state.get("context_history") or [])
        return out

    def _append_trace(state: ChatState, item: dict[str, Any]) -> None:
        trace = list(state.get("trace") or [])
        trace.append(item)
        state["trace"] = trace

    def _trace_transition(state: ChatState, from_node: str, to_node: str, reason: str) -> None:
        _append_trace(
            state,
            {
                "kind": "edge",
                "from": from_node,
                "to": to_node,
                "reason": reason,
            },
        )

    def _traced_node(name: str, fn):
        async def _run(state: ChatState) -> ChatState:
            before = _trace_view(state)
            out = await fn(state)
            after = _trace_view(out)
            _append_trace(
                out,
                {
                    "kind": "node",
                    "node": name,
                    "context": trace_context_by_node.get(name, ""),
                    "before": before,
                    "after": after,
                },
            )
            return out

        return _run

    async def event_gate_node(state: ChatState) -> ChatState:
        out: ChatState = dict(state)
        text = (state.get("text") or "").strip()
        event_type_in = (state.get("event_type") or "").strip() or None

        if event_type_in == "round_ended" or _is_round_ended_event(text):
            out["event_type"] = "round_ended"
            return out

        if event_type_in == "player_joined" or _is_join_event(text):
            out["event_type"] = "player_joined"
            seconds_since = state.get("seconds_since_last_bot_message")
            allow = seconds_since is None or float(seconds_since) >= 1.0
            if not allow:
                out["join_announcement_blocked"] = True
            return out

        out["event_type"] = "chat"
        return out

    async def greet_player_node(state: ChatState) -> ChatState:
        out: ChatState = dict(state)

        if out.get("join_announcement_blocked"):
            return out

        name = _parse_join_username(out)
        last_greet = float(out.get("last_greet_ts") or 0.0)
        now_ts = float(out.get("now_ts") or 0.0)
        window = float(out.get("rejoin_window_seconds") or 0.0)

        if last_greet > 0.0 and (now_ts - last_greet) <= window:
            out["reply"] = f"wb {name}"
            out["set_last_greet"] = True
            return out

        out["reply"] = f"hey {name}"
        out["set_last_greet"] = True
        return out

    async def round_end_node(state: ChatState) -> ChatState:
        payload = dict(state.get("event_payload") or {})
        room_state = payload.get("room_state")

        result = await deps.round_summary.handle_round_end(room_state)

        out: ChatState = dict(state)
        out["round_summary"] = result.summary_text
        out["round_outcome"] = result.outcome
        out["flush_deferred"] = True
        if result.summary_text:
            out["reply"] = result.summary_text
        return out

    async def spam_guard_node(state: ChatState) -> ChatState:
        sender = (state.get("sender") or "").strip()
        sender_l = sender.lower()
        text = (state.get("text") or "").strip().lower()
        history: List[ChatTurn] = state.get("history", [])
        out: ChatState = dict(state)
        preblocked = bool(state.get("spam_preblocked")) or bool(state.get("spam_blocked"))
        repeat_count_pre = int(state.get("spam_repeat_count") or 0)
        rapid_fire = bool(state.get("spam_rapid_fire"))
        out["spam_preblocked"] = bool(state.get("spam_preblocked"))
        out["spam_repeat_count"] = repeat_count_pre
        out["spam_rapid_fire"] = rapid_fire

        if preblocked:
            out["spam_blocked"] = True
            logger.info(
                "[bot.trace] graph.spam_flag sender=%s prior_repeat_count=%d rapid_fire=%s spam_blocked=%s reply=%s source=engine_precheck",
                sender,
                repeat_count_pre,
                rapid_fire,
                True,
                "yes" if out.get("reply") else "no",
            )
            return out

        out["spam_blocked"] = False

        if not sender or not text:
            return out

        # Count recent duplicate messages from the same sender even if other turns
        # are interleaved (e.g. bot replies between repeats).
        repeat_count = 0
        skipped_current = False
        for turn in reversed(list(history)[-30:]):
            turn_sender_l = (getattr(turn, "sender", "") or "").strip().lower()
            if turn_sender_l != sender_l:
                continue

            prev_text = (getattr(turn, "text", "") or "").strip().lower()
            if prev_text != text:
                continue

            if not skipped_current:
                # Engine records the incoming message before graph execution;
                # skip that newest copy and count prior repeats only.
                skipped_current = True
                continue

            repeat_count += 1
            if repeat_count >= 4:
                break

        if repeat_count >= 1:
            if repeat_count < 4:
                out["spam_blocked"] = True
            else:
                out["reply"] = f"{sender}, please stop spamming the chat."

            logger.info(
                "[bot.trace] graph.spam_flag sender=%s prior_repeat_count=%d spam_blocked=%s reply=%s",
                sender,
                repeat_count,
                bool(out.get("spam_blocked")),
                "yes" if out.get("reply") else "no",
            )
            return out

        return out

    async def targeting_node(state: ChatState) -> ChatState:
        raw = (state.get("text") or "").strip()
        text = raw.lower()

        if text.startswith("event:"):
            return state

        participants = int(state.get("participant_count") or 0)
        bot_name = (state.get("bot_name") or "").strip().lower()
        known_names = collect_known_player_names(state, bot_name=state.get("bot_name") or "")
        sender_l = (state.get("sender") or "").strip().lower()
        short_farewell_opener = is_short_farewell_opener_line(raw)
        recent_other_human_leave = recent_other_human_signaled_leave(
            state, bot_name=state.get("bot_name") or ""
        )

        # "bye tap!" / "hey tap" — vocative is a prefix nickname of the bot (avoid matching bare "tap" mid-sentence).
        if bot_name and len(bot_name) >= 4:
            voc = re.match(
                r"^\s*(?:bye|goodbye|cya|see\s+ya|see\s+you|hey|hi|hello|yo|thanks|thx)\s*[,!]?\s*([a-z][\w-]{1,31})\b",
                raw,
                re.IGNORECASE,
            )
            if voc:
                w = voc.group(1).lower()
                if w != bot_name and len(w) >= 3 and bot_name.startswith(w) and len(w) < len(bot_name):
                    out0: ChatState = dict(state)
                    out0["is_explicitly_directed"] = True
                    out0["is_explicitly_not_for_bot"] = False
                    out0["target_blocked"] = False
                    return out0

        # Sign-offs to the room include the bot; mark directed so the router does not drop them as non-addressee.
        # (Reply wording comes from the response model, not a graph shortcut.)
        if participants > 2:
            if (
                re.search(r"\b(bye|goodbye|cya|see\s+ya|see\s+you)\b", text)
                and re.search(r"\b(all|everyone|folks|guys|y'all|everybody)\b", text)
                and len(raw) <= 120
            ):
                out_f: ChatState = dict(state)
                out_f["short_farewell_opener"] = short_farewell_opener
                out_f["recent_other_human_leave"] = recent_other_human_leave
                out_f["is_explicitly_directed"] = True
                out_f["is_explicitly_not_for_bot"] = False
                out_f["target_blocked"] = False
                return out_f
            if short_farewell_opener and not recent_other_human_leave:
                out_f = dict(state)
                out_f["short_farewell_opener"] = short_farewell_opener
                out_f["recent_other_human_leave"] = recent_other_human_leave
                out_f["is_explicitly_directed"] = True
                out_f["is_explicitly_not_for_bot"] = False
                out_f["target_blocked"] = False
                return out_f

        greet_match = re.match(
            r"^(hi|hey|hello|yo|hiya|sup)[\s,!?-]*([a-z][\w-]{1,31})\b",
            text,
            re.IGNORECASE,
        )
        name_prefix = re.match(r"^([a-z][\w-]{1,31})\s*[:,]\s*", raw, re.IGNORECASE)

        target_name = None
        if greet_match:
            candidate = greet_match.group(2).lower()
            if candidate in known_names:
                target_name = candidate
        elif name_prefix:
            candidate = name_prefix.group(1).strip().lower()
            if candidate in known_names:
                target_name = candidate
        elif participants > 2 and known_names:
            # "where you from rex?" — name at end; only when we know room human names (from history or payload).
            tail_m = _TRAILING_ADDRESSEE.search(raw.strip())
            if tail_m:
                candidate = tail_m.group(1).lower()
                if candidate in known_names and candidate != bot_name and candidate != sender_l:
                    target_name = candidate

        out: ChatState = dict(state)
        out["short_farewell_opener"] = short_farewell_opener
        out["recent_other_human_leave"] = recent_other_human_leave
        out["is_explicitly_directed"] = False
        out["is_explicitly_not_for_bot"] = False
        out["target_blocked"] = False

        if target_name:
            if target_name == bot_name:
                out["is_explicitly_directed"] = True
                return out

            out["is_explicitly_not_for_bot"] = True
            out["target_blocked"] = True
            return out

        if bot_name and bot_name in text:
            out["is_explicitly_directed"] = True
            return out

        # Two-party default: directed at bot, unless this is a piggyback bye after someone else just left (see recent history).
        if participants <= 2 and not (recent_other_human_leave and short_farewell_opener):
            out["is_explicitly_directed"] = True

        return out

    async def policy_node(state: ChatState) -> ChatState:
        explicitly_directed = bool(state.get("is_explicitly_directed"))
        participants = int(state.get("participant_count") or 0)
        seconds_since = state.get("seconds_since_last_bot_message")
        skill_level = int(state.get("skill_level") or 0)

        provisional_directed = explicitly_directed or participants <= 2
        # More than two participants and no explicit recipient: router still must run for directedness + routing.
        needs_ambiguous_router = participants > 2 and not explicitly_directed
        is_directed_for_gate = provisional_directed or needs_ambiguous_router

        if is_directed_for_gate:
            allow = True
        else:
            # Non-directed multi-party chatter: throttle by recency, then probabilistic gate by skill.
            if seconds_since is not None and float(seconds_since) < 8.0:
                allow = False
            else:
                skill = max(0, min(5, skill_level))
                base_chance = 0.10 + (skill * 0.12)  # 10%..70%
                allow = random.random() < base_chance

        out: ChatState = dict(state)
        out["policy_allowed"] = allow
        out["is_directed"] = is_directed_for_gate
        return out

    def _merge_route_flags(d: RouteDecision) -> tuple[bool, bool]:
        need_stats = d.need_stats or d.route == ChatRoute.GAME_STATS_REPLY
        need_history = d.need_history
        return need_stats, need_history

    async def decide_node(state: ChatState) -> ChatState:
        """
        Main router: route, privacy, stats/history, and final semantic handling
        of ambiguous directedness in multi-player rooms.
        """
        router = getattr(deps, "router", None)
        if router is None:
            out = dict(state)
            out["route"] = ChatRoute.IGNORE
            out["privacy_blocked"] = False
            out["need_stats"] = False
            out["need_history"] = False
            out["ignore_reason"] = "router_unavailable"
            out["router_directed_at_bot"] = None
            logger.info(
                "[bot.trace] graph.decide route=ignore sender=%s reason=%r text=%r",
                (state.get("sender") or "").strip()[:80],
                out["ignore_reason"],
                (state.get("text") or "")[:160],
            )
            return out

        participants = int(state.get("participant_count") or 0)
        explicitly_directed = bool(state.get("is_explicitly_directed"))
        explicitly_not_for_bot = bool(state.get("is_explicitly_not_for_bot"))
        raw_in = (state.get("text") or "").strip()
        if is_post_round_gg_echo(raw_in, list(state.get("history") or []), bot_name=state.get("bot_name") or ""):
            out_wr: ChatState = dict(state)
            out_wr["route"] = ChatRoute.IGNORE
            out_wr["privacy_blocked"] = False
            out_wr["need_stats"] = False
            out_wr["need_history"] = False
            out_wr["ignore_reason"] = "post_round_gg_echo"
            out_wr["router_directed_at_bot"] = False
            out_wr["is_directed"] = False
            logger.info(
                "[bot.trace] graph.decide route=ignore sender=%s reason=post_round_gg_echo text=%r",
                (state.get("sender") or "").strip()[:80],
                (state.get("text") or "")[:160],
            )
            return out_wr

        recent_other_human_leave = recent_other_human_signaled_leave(
            state, bot_name=state.get("bot_name") or ""
        )
        short_farewell_opener = is_short_farewell_opener_line(raw_in)
        farewell_piggyback_likely = recent_other_human_leave and short_farewell_opener

        decision = await router.classify(
            text=state["text"],
            username=state["sender"],
            is_round_active=bool(state.get("is_round_active")),
            recent_turns=list(state.get("recent_turns") or []),
            last_round_outcome=state.get("last_round_outcome"),
            participant_count=participants,
            bot_name=state.get("bot_name") or "",
            targeting_hint=(
                "directed"
                if explicitly_directed
                else "not_for_bot"
                if explicitly_not_for_bot
                else "ambiguous"
            ),
            history=list(state.get("history") or []),
            participant_names=list(state.get("participant_names") or []),
            farewell_piggyback_likely=farewell_piggyback_likely,
        )

        need_stats, need_history = _merge_route_flags(decision)
        out: ChatState = dict(state)
        out["route"] = decision.route
        out["privacy_blocked"] = decision.privacy_blocked
        out["need_stats"] = need_stats
        out["need_history"] = need_history
        # Decide owns whether memory retrieval should happen at all.
        out["use_memory"] = decision.route == ChatRoute.MEMORY_REPLY
        out["router_directed_at_bot"] = decision.directed_at_bot
        out["short_farewell_opener"] = short_farewell_opener
        out["recent_other_human_leave"] = recent_other_human_leave
        out["farewell_piggyback_likely"] = farewell_piggyback_likely

        if explicitly_directed:
            out["is_directed"] = True
        elif explicitly_not_for_bot:
            out["is_directed"] = False
        elif decision.directed_at_bot is not None:
            # Router-owned semantic directedness for ambiguous multi-party turns.
            out["is_directed"] = bool(decision.directed_at_bot)
        else:
            # Router did not set directed_at_bot (e.g. empty payload); avoid inferring from ignore_reason.
            out["is_directed"] = decision.route != ChatRoute.IGNORE

        if decision.route == ChatRoute.IGNORE:
            reason = decision.ignore_reason or "unspecified"
            out["ignore_reason"] = reason
            logger.info(
                "[bot.trace] graph.decide route=ignore sender=%s reason=%r text=%r",
                (state.get("sender") or "").strip()[:80],
                reason,
                (state.get("text") or "")[:160],
            )
        else:
            out["ignore_reason"] = None

        return out

    async def plan_memory_retrieval_node(state: ChatState) -> ChatState:
        """LLM produces retrieval details only; decide_node already chooses whether memory is used."""
        out: ChatState = dict(state)
        if not state.get("use_memory"):
            out["memory_query"] = None
            out["memory_mode"] = "none"
            out["memory_min_similarity"] = 0.0
            out["memory_max_results"] = 0
            out["memory_plan_source"] = "skipped"
            out["memory_plan_fallback_reason"] = "skipped_by_decide"
            return out
        svc = getattr(deps, "memory_retrieval_plan", None)
        if svc is None:
            t = (state.get("text") or "").strip()
            out["memory_query"] = (t[:500] if t else None)
            out["memory_mode"] = "broad_profile"
            out["memory_min_similarity"] = 0.2
            out["memory_max_results"] = 3
            out["memory_plan_source"] = "fallback"
            out["memory_plan_fallback_reason"] = "planner_dependency_missing"
            return out

        plan = await svc.plan(
            username=state.get("sender") or "",
            text=state.get("text") or "",
            recent_turns=list(state.get("recent_turns") or []),
            last_round_outcome=state.get("last_round_outcome"),
            route=state.get("route"),
        )
        out["use_memory"] = plan.use_memory
        out["memory_query"] = plan.query
        out["memory_mode"] = plan.mode
        out["memory_min_similarity"] = float(plan.min_similarity)
        out["memory_max_results"] = int(plan.max_results)
        out["memory_plan_source"] = plan.plan_source
        out["memory_plan_fallback_reason"] = plan.fallback_reason
        return out

    async def gather_context_node(state: ChatState) -> ChatState:
        """Deterministic fetch: apply `MAX_MEMORY_RETRIEVAL_RESULTS` regardless of planner advisory count."""
        out: ChatState = dict(state)

        memories = []
        stats = None
        context_history: List[ChatTurn] = []

        if state.get("use_memory"):
            qraw = (state.get("memory_query") or state.get("text") or "").strip()
            if qraw:
                try:
                    advisory = int(state.get("memory_max_results") or MAX_MEMORY_RETRIEVAL_RESULTS)
                except (TypeError, ValueError):
                    advisory = MAX_MEMORY_RETRIEVAL_RESULTS
                advisory = max(1, advisory)
                limit = min(advisory, MAX_MEMORY_RETRIEVAL_RESULTS)
                min_sim = float(state.get("memory_min_similarity") or 0.0)
                min_sim = max(0.0, min(1.0, min_sim))
                memories = await deps.memory.retrieve_relevant_memories(
                    username=state["sender"],
                    query=qraw,
                    limit=limit,
                    min_similarity=min_sim if min_sim > 0.0 else None,
                )

        if state.get("need_stats"):
            stats = await deps.player_stats.get_summary(username=state["sender"])

        all_history = list(state.get("history") or [])
        context_history_limit = int(state.get("context_history_limit") or DEFAULT_CONTEXT_HISTORY_LIMIT)
        context_history_limit = max(1, context_history_limit)
        if state.get("need_history"):
            context_history = all_history
        else:
            context_history = all_history[-context_history_limit:]

        out["memories"] = memories
        out["stats"] = stats
        out["context_history"] = context_history
        return out

    async def generate_node(state: ChatState) -> ChatState:
        route = state.get("route") or ChatRoute.SIMPLE_REPLY

        raw_mode = (state.get("memory_mode") or "").strip()
        mem_mode: MemoryRetrievalMode | None = (
            raw_mode
            if raw_mode in {"none", "broad_profile", "callback", "specific_fact", "general"}
            else None
        )

        context = ReplyContext(
            username=state["sender"],
            user_message=state["text"],
            route=route,
            memories=state.get("memories", []),
            stats=state.get("stats"),
            recent_turns=state.get("context_history", []),
            memory_retrieval_mode=mem_mode,
            last_round_outcome=state.get("last_round_outcome"),
        )

        reply = await deps.responder.generate(context=context)

        out: ChatState = dict(state)
        out["reply"] = reply or ""
        return out

    async def decide_memory_write_node(state: ChatState) -> ChatState:
        """Post-generation LLM (or heuristic): whether to persist long-term memory."""
        out: ChatState = dict(state)
        svc = getattr(deps, "memory_write", None)
        if svc is None:
            return out
        try:
            decision = await svc.decide(
                username=state.get("sender") or "",
                user_message=state.get("text") or "",
                bot_reply=state.get("reply") or "",
                memories=list(state.get("memories") or []),
                stats=state.get("stats"),
            )
        except Exception:
            out["should_write_memory"] = False
            out["memory_write_text"] = None
            out["memory_persisted"] = False
            out["memory_write_skip_reason"] = "memory_write_decision_error"
            return out
        out["should_write_memory"] = decision.should_write_memory
        out["memory_write_text"] = decision.memory_write_text
        out["memory_persisted"] = False
        out["memory_write_skip_reason"] = None
        return out

    async def persist_memory_node(state: ChatState) -> ChatState:
        out: ChatState = dict(state)
        out["memory_persisted"] = False
        out["memory_write_skip_reason"] = None
        if not state.get("should_write_memory"):
            out["memory_write_skip_reason"] = "write_not_requested"
            return out

        memory_text = (state.get("memory_write_text") or "").strip()
        if not memory_text:
            out["memory_write_skip_reason"] = "empty_memory_text"
            return out

        if is_redundant_username_memory(memory_text, (state.get("sender") or "").strip()):
            out["memory_write_skip_reason"] = "redundant_username_memory"
            return out

        existing = list(state.get("memories") or [])
        existing_texts = {
            (getattr(m, "memory_text", "") or "").strip().lower() for m in existing
        }
        if memory_text.lower() in existing_texts:
            out["memory_write_skip_reason"] = "already_present_in_recalled_memories"
            return out

        try:
            await deps.memory.store_memory(
                username=state.get("sender") or "",
                memory_text=memory_text,
                metadata={"source": "chat"},
            )
        except Exception:
            out["memory_write_skip_reason"] = "store_memory_failed"
            return out

        out["memory_persisted"] = True
        return out

    async def sanitize_node(state: ChatState) -> ChatState:
        reply = (state.get("reply") or "").strip()
        if not reply:
            return state

        stats = state.get("stats")
        memories = state.get("memories") or []
        session_started_at = float(state.get("session_started_at") or 0.0)

        wins = int(getattr(stats, "wins_vs_bot", 0) or 0) if stats is not None else 0
        losses = int(getattr(stats, "losses_vs_bot", 0) or 0) if stats is not None else 0
        has_game_history = (wins + losses) > 0

        has_round_summary_memory = any(
            isinstance(getattr(m, "metadata", None), dict)
            and (getattr(m, "metadata", {}) or {}).get("source") == "round_summary"
            for m in memories
        )
        has_prior_session_memory = any(
            (getattr(m, "created_at", None) or 0.0) < session_started_at
            for m in memories
        )

        if has_game_history or has_round_summary_memory or has_prior_session_memory:
            return state

        sanitized = reply
        replacements = [
            (r"\bgood to see you again\b", "good to see you"),
            (r"\bsee you again\b", "see you"),
            (r"\bwelcome back\b", "welcome"),
            (r"\blooks like you'?re back\b", "nice to chat"),
            (r"\byou'?re back\b", "you are here"),
            (r"\bback for some more\b", "up for a chat"),
            (r"\bagain\b", ""),
        ]
        for pattern, replacement in replacements:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        sanitized = re.sub(r"\s{2,}", " ", sanitized).strip(" ,.!?") or reply

        out: ChatState = dict(state)
        out["reply"] = sanitized
        return out

    def _add_typo(text: str) -> str:
        words = text.split()
        candidate_idxs = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
        if not candidate_idxs:
            return text

        idx = random.choice(candidate_idxs)
        word = words[idx]
        if len(word) < 2:
            return text

        j = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[j], chars[j + 1] = chars[j + 1], chars[j]
        words[idx] = "".join(chars)
        return " ".join(words)

    async def humanize_node(state: ChatState) -> ChatState:
        reply = (state.get("reply") or "").strip()
        if not reply:
            return state

        out_text = reply
        if random.random() < 0.35:
            out_text = re.sub(r"[.!?,;:]+$", "", out_text)
        if random.random() < 0.25:
            out_text = out_text.lower()
        if random.random() < 0.2:
            out_text = _add_typo(out_text)

        out: ChatState = dict(state)
        out["reply"] = out_text
        return out

    # ---------- routing ----------

    def _after_event_gate(state: ChatState) -> str:
        event_type = state.get("event_type")
        if event_type == "round_ended":
            _trace_transition(state, "event_gate", "round_end", "round_ended event")
            return "round_end"
        if event_type == "player_joined":
            if state.get("join_announcement_blocked"):
                _trace_transition(
                    state, "event_gate", "__end__", "join announcement blocked by policy"
                )
                return END  # type: ignore[return-value]
            _trace_transition(state, "event_gate", "player_join", "player_joined event")
            return "player_join"
        _trace_transition(state, "event_gate", "spam_filter", "standard chat flow")
        return "spam_filter"

    def _after_round_end(state: ChatState) -> str:
        if (state.get("reply") or "").strip():
            _trace_transition(state, "round_end", "sanitize", "round summary reply exists")
            return "sanitize"
        _trace_transition(state, "round_end", "__end__", "no round summary reply")
        return END  # type: ignore[return-value]

    def _after_player_join(state: ChatState) -> str:
        if (state.get("reply") or "").strip():
            _trace_transition(state, "player_join", "sanitize", "greeting reply exists")
            return "sanitize"
        _trace_transition(state, "player_join", "__end__", "no greeting reply")
        return END  # type: ignore[return-value]

    def _after_spam_filter(state: ChatState) -> str:
        if (state.get("reply") or "").strip():
            _trace_transition(state, "spam_filter", "sanitize", "spam generated direct reply")
            return "sanitize"
        if state.get("spam_blocked"):
            _trace_transition(state, "spam_filter", "__end__", "spam blocked without reply")
            return END  # type: ignore[return-value]
        _trace_transition(state, "spam_filter", "targeting", "not blocked by spam guard")
        return "targeting"

    def _after_targeting(state: ChatState) -> str:
        if state.get("target_blocked") or state.get("is_explicitly_not_for_bot"):
            _trace_transition(state, "targeting", "__end__", "message addressed to other player")
            return END  # type: ignore[return-value]
        _trace_transition(state, "targeting", "policy", "message potentially for bot")
        return "policy"

    def _after_policy(state: ChatState) -> str:
        if not state.get("policy_allowed"):
            _trace_transition(state, "policy", "__end__", "policy disallowed response")
            return END  # type: ignore[return-value]
        _trace_transition(state, "policy", "decide", "policy allowed response")
        return "decide"

    def _after_decide(state: ChatState) -> str:
        if (state.get("reply") or "").strip():
            _trace_transition(state, "decide", "sanitize", "decision provided immediate reply")
            return "sanitize"
        if state.get("privacy_blocked"):
            _trace_transition(state, "decide", "__end__", "privacy blocked")
            return END  # type: ignore[return-value]
        if state.get("route") == ChatRoute.IGNORE:
            _trace_transition(state, "decide", "__end__", "route ignore")
            return END  # type: ignore[return-value]
        if state.get("use_memory"):
            _trace_transition(state, "decide", "plan_memory_retrieval", "memory route requires retrieval planning")
            return "plan_memory_retrieval"
        _trace_transition(state, "decide", "gather_context", "no memory retrieval required")
        return "gather_context"

    def _after_decide_memory_write(state: ChatState) -> str:
        if state.get("should_write_memory"):
            _trace_transition(state, "decide_memory_write", "persist_memory", "memory write approved")
            return "persist_memory"
        _trace_transition(state, "decide_memory_write", "sanitize", "memory write skipped")
        return "sanitize"

    workflow.add_node("event_gate", _traced_node("event_gate", event_gate_node))
    workflow.add_node("player_join", _traced_node("player_join", greet_player_node))
    workflow.add_node("round_end", _traced_node("round_end", round_end_node))
    workflow.add_node("spam_filter", _traced_node("spam_filter", spam_guard_node))
    workflow.add_node("targeting", _traced_node("targeting", targeting_node))
    workflow.add_node("policy", _traced_node("policy", policy_node))
    workflow.add_node("decide", _traced_node("decide", decide_node))
    workflow.add_node(
        "plan_memory_retrieval",
        _traced_node("plan_memory_retrieval", plan_memory_retrieval_node),
    )
    workflow.add_node("gather_context", _traced_node("gather_context", gather_context_node))
    workflow.add_node("generate", _traced_node("generate", generate_node))
    workflow.add_node(
        "decide_memory_write",
        _traced_node("decide_memory_write", decide_memory_write_node),
    )
    workflow.add_node("persist_memory", _traced_node("persist_memory", persist_memory_node))
    workflow.add_node("sanitize", _traced_node("sanitize", sanitize_node))
    workflow.add_node("humanize", _traced_node("humanize", humanize_node))

    workflow.set_entry_point("event_gate")

    workflow.add_conditional_edges(
        "event_gate",
        _after_event_gate,
        {
            "round_end": "round_end",
            "player_join": "player_join",
            "spam_filter": "spam_filter",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "round_end",
        _after_round_end,
        {
            "sanitize": "sanitize",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "player_join",
        _after_player_join,
        {
            "sanitize": "sanitize",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "spam_filter",
        _after_spam_filter,
        {
            "targeting": "targeting",
            "sanitize": "sanitize",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "targeting",
        _after_targeting,
        {
            "policy": "policy",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "policy",
        _after_policy,
        {
            "decide": "decide",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "decide",
        _after_decide,
        {
            "plan_memory_retrieval": "plan_memory_retrieval",
            "gather_context": "gather_context",
            "sanitize": "sanitize",
            "__end__": END,
        },
    )

    workflow.add_edge("plan_memory_retrieval", "gather_context")
    workflow.add_edge("gather_context", "generate")
    workflow.add_edge("generate", "decide_memory_write")

    workflow.add_conditional_edges(
        "decide_memory_write",
        _after_decide_memory_write,
        {
            "persist_memory": "persist_memory",
            "sanitize": "sanitize",
        },
    )

    workflow.add_edge("persist_memory", "sanitize")
    workflow.add_edge("sanitize", "humanize")
    workflow.add_edge("humanize", END)

    return workflow.compile()