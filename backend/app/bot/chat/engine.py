from __future__ import annotations

from collections import deque
import logging
import time
import uuid

from ..config import BotConfig, ChatSendFn, TraceEmitFn
from .graph import build_chat_graph, GraphDeps
from .memory.extraction import HeuristicMemoryExtractionService, MemoryExtractionService
from ..models import ChatTurn
from ..player_stats import InMemoryPlayerStatsService, PlayerStatsService
from ..round_summary_service import RoundSummaryService
from .response_generator import BotResponseGenerator, OpenAIBotResponseGenerator
from .routers import LLMChatRouter, OpenAILLMChatRouter
from .memory.semantic import SemanticMemoryService
from .engine_utils import (
    defer_message,
    send_chat as send_chat_util,
    flush_deferred_messages,
)
from .deps import BotChatDependencies

logger = logging.getLogger("uvicorn.error")



class BotChatEngine:
    def __init__(
        self,
        config: BotConfig,
        *,
        llm_router: LLMChatRouter | None = None,
        semantic_memory: SemanticMemoryService | None = None,
        player_stats: PlayerStatsService | None = None,
        response_generator: BotResponseGenerator | None = None,
        memory_extractor: MemoryExtractionService | None = None,
    ) -> None:
        self.config = config

        deps = BotChatDependencies.build(
            config,
            llm_router=llm_router,
            semantic_memory=semantic_memory,
            player_stats=player_stats,
            response_generator=response_generator,
            memory_extractor=memory_extractor,
        )

        self.llm_router = deps.llm_router
        self.semantic_memory = deps.semantic_memory
        self.player_stats = deps.player_stats
        self.response_generator = deps.response_generator
        self.memory_extractor = deps.memory_extractor
        self._memory_retrieval_planning = deps.memory_retrieval_planning
        self._memory_write_decision = deps.memory_write_decision

        self._chat_router_model = deps.chat_router_model
        self._chat_response_model = deps.chat_response_model
        self._embeddings_model = deps.embeddings_model

        self._init_runtime_state()
        self._log_init()

        self._round_summary = RoundSummaryService(
            player_stats=self.player_stats,
            semantic_memory=self.semantic_memory,
            bot_name=self.config.name,
        )

        self._graph = build_chat_graph(
            GraphDeps(
                router=self.llm_router,
                memory=self.semantic_memory,
                player_stats=self.player_stats,
                responder=self.response_generator,
                round_summary=self._round_summary,
                memory_retrieval_plan=self._memory_retrieval_planning,
                memory_write=self._memory_write_decision,
            )
        )

    def _init_runtime_state(self) -> None:
        self._recent_turns: deque[ChatTurn] = deque(maxlen=self.config.recent_message_limit)
        self._repeat_message_state: dict[str, tuple[str, int, float, float]] = {}
        self._last_round_outcome: dict[str, object] | None = None
        self._deferred_messages: list[tuple[str, str, int, bool, int, bool]] = []
        self._last_greeted_at: dict[str, float] = {}
        self._last_bot_message_ts = 0.0
        self._rejoin_window_seconds = 20 * 60
        self._session_started_at = time.time()

    def seconds_since_last_bot_message(self) -> float | None:
        if self._last_bot_message_ts <= 0:
            return None
        return time.time() - self._last_bot_message_ts

    def _precheck_spam(
        self, *, sender: str, text: str, now_ts: float
    ) -> tuple[bool, int, bool]:
        sender_l = sender.strip().lower()
        norm_text = (text or "").strip().lower()
        if not sender_l or not norm_text:
            return False, 0, False

        last_text, streak, last_seen_ts, streak_started_ts = self._repeat_message_state.get(
            sender_l, ("", 0, 0.0, now_ts)
        )
        rapid_fire = float(last_seen_ts) > 0.0 and (now_ts - float(last_seen_ts)) <= 1.0

        # Keep duplicate streak local to recent time; old repeats should not trigger.
        same_text = last_text == norm_text and (now_ts - float(last_seen_ts)) <= 120.0
        if same_text:
            streak = int(streak) + 1
        else:
            streak = 1
            streak_started_ts = now_ts

        self._repeat_message_state[sender_l] = (norm_text, streak, now_ts, float(streak_started_ts))

        prior_repeat_count = max(0, streak - 1)
        return (prior_repeat_count >= 1) or rapid_fire, prior_repeat_count, rapid_fire

    def _snapshot_participant_names(self) -> list[str]:
        """Distinct human senders seen recently (bot/system excluded); augments graph targeting with API-style hints."""
        bot = self.config.name.strip().lower()
        seen: set[str] = set()
        names: list[str] = []
        for t in self._recent_turns:
            s = (getattr(t, "sender", "") or "").strip()
            if not s:
                continue
            sl = s.lower()
            if sl in ("system", bot):
                continue
            if sl not in seen:
                seen.add(sl)
                names.append(s)
        return names

    def _log_init(self) -> None:
        if self.config.trace_enabled:
            logger.info(
                "[bot.trace] init recent_limit=%d router_model=%s response_model=%s embedding_model=%s",
                self.config.recent_message_limit,
                self.config.llm_router_model,
                self.config.llm_response_model,
                self.config.embedding_model,
            )

    async def on_player_joined(
        self,
        username: str,
        send_chat: ChatSendFn,
        emit_trace: TraceEmitFn | None = None,
    ) -> None:
        event_text = f"EVENT: player_joined username={username}"
        if self.config.trace_enabled:
            logger.info("[bot.trace] event.enqueue type=player_joined user=%s", username)

        self.record_message(sender="system", text=event_text, is_bot=False)

        key = username.strip().lower()
        state = {
            "sender": username,
            "text": event_text,
            "event_type": "player_joined",
            "event_payload": {"username": username},
            "is_round_active": False,
            "participant_count": 0,
            "participant_names": self._snapshot_participant_names(),
            "bot_name": self.config.name,
            "last_greet_ts": float(self._last_greeted_at.get(key, 0.0)),
            "rejoin_window_seconds": float(self._rejoin_window_seconds),
            "now_ts": float(time.time()),
            "seconds_since_last_bot_message": self.seconds_since_last_bot_message(),
            "skill_level": int(self.config.skill_level),
            "context_history_limit": int(self.config.context_history_limit),
            "recent_turns": [],
            "history": [],
            "session_started_at": self._session_started_at,
        }

        result = await self._graph.ainvoke(state)
        reply = result.get("reply", "")
        set_last_greet = bool(result.get("set_last_greet", False))
        trace_id = str(uuid.uuid4())
        trace = list(result.get("trace") or [])
        trace_source = {
            "sender": username,
            "text": event_text,
            "eventType": "player_joined",
            "isRoundActive": False,
        }
        await self._emit_trace(
            trace_id=trace_id,
            trace=trace,
            source=trace_source,
            generated_reply=reply,
            emit_trace=emit_trace,
        )

        if reply:
            await send_chat_util(
                self,
                text=reply,
                send_chat=send_chat,
                trace=trace,
                trace_source=trace_source,
            )

        if set_last_greet:
            self._last_greeted_at[key] = time.time()

    async def on_round_started(self, send_chat: ChatSendFn) -> None:
        if self.config.trace_enabled:
            logger.info("[bot.trace] event.enqueue type=round_started")
        self.record_message(sender="system", text="EVENT: round_started", is_bot=False)

    async def on_round_ended(
        self,
        send_chat: ChatSendFn,
        room_state: dict | None = None,
        emit_trace: TraceEmitFn | None = None,
    ) -> None:
        event_text = "EVENT: round_ended"
        if self.config.trace_enabled:
            logger.info("[bot.trace] event.enqueue type=round_ended room_state=%s", "present" if room_state else "none")
        self.record_message(sender="system", text=event_text, is_bot=False)
        state = {
            "sender": "system",
            "text": event_text,
            "event_type": "round_ended",
            "event_payload": {"room_state": room_state},
            "is_round_active": False,
            "participant_count": 0,
            "participant_names": self._snapshot_participant_names(),
            "bot_name": self.config.name,
            "session_started_at": self._session_started_at,
            "seconds_since_last_bot_message": self.seconds_since_last_bot_message(),
            "skill_level": int(self.config.skill_level),
            "context_history_limit": int(self.config.context_history_limit),
            "recent_turns": [f"{turn.sender}: {turn.text}" for turn in self._recent_turns][-self.config.recent_message_limit :],
            "history": list(self._recent_turns),
        }
        result = await self._graph.ainvoke(state)
        reply = (result.get("reply") or "").strip()
        round_outcome = result.get("round_outcome")
        trace_id = str(uuid.uuid4())
        trace = list(result.get("trace") or [])
        trace_source = {
            "sender": "system",
            "text": event_text,
            "eventType": "round_ended",
            "isRoundActive": False,
        }
        await self._emit_trace(
            trace_id=trace_id,
            trace=trace,
            source=trace_source,
            generated_reply=reply,
            emit_trace=emit_trace,
        )
        if round_outcome is not None:
            self._last_round_outcome = round_outcome
        if reply:
            await send_chat_util(
                self,
                text=reply,
                send_chat=send_chat,
                trace=trace,
                trace_source=trace_source,
            )
        if result.get("flush_deferred"):
            await flush_deferred_messages(self, send_chat=send_chat)

    async def on_chat_message(
        self,
        *,
        sender: str,
        text: str,
        is_round_active: bool,
        participant_count: int,
        send_chat: ChatSendFn,
        emit_trace: TraceEmitFn | None = None,
        pre_spam_blocked: bool | None = None,
        pre_repeat_count: int | None = None,
        pre_spam_rapid_fire: bool | None = None,
    ) -> None:
        now_ts = float(time.time())
        if (
            pre_spam_blocked is None
            or pre_repeat_count is None
            or pre_spam_rapid_fire is None
        ):
            pre_spam_blocked, pre_repeat_count, pre_spam_rapid_fire = self._precheck_spam(
                sender=sender, text=text, now_ts=now_ts
            )
        if self.config.trace_enabled:
            logger.info(
                "[bot.trace] chat.in sender=%s active=%s participants=%d text=%r",
                sender,
                is_round_active,
                participant_count,
                text[:220],
            )
        self.record_message(sender=sender, text=text, is_bot=False)
        if sender == self.config.name:
            return
        if is_round_active:
            await defer_message(
                self,
                sender,
                text,
                participant_count,
                pre_spam_blocked=bool(pre_spam_blocked),
                pre_repeat_count=int(pre_repeat_count or 0),
                pre_spam_rapid_fire=bool(pre_spam_rapid_fire),
            )
            return
        recent_turns = [f"{turn.sender}: {turn.text}" for turn in self._recent_turns]
        state = {
            "sender": sender,
            "text": text,
            "is_round_active": is_round_active,
            "participant_count": participant_count,
            "participant_names": self._snapshot_participant_names(),
            "bot_name": self.config.name,
            "now_ts": now_ts,
            "last_greet_ts": float(self._last_greeted_at.get(sender.strip().lower(), 0.0)),
            "rejoin_window_seconds": float(self._rejoin_window_seconds),
            "seconds_since_last_bot_message": self.seconds_since_last_bot_message(),
            "skill_level": int(self.config.skill_level),
            "context_history_limit": int(self.config.context_history_limit),
            "session_started_at": self._session_started_at,
            "recent_turns": recent_turns[-self.config.recent_message_limit :],
            "history": list(self._recent_turns),
            "last_round_outcome": self._last_round_outcome,
            "spam_preblocked": pre_spam_blocked,
            "spam_repeat_count": pre_repeat_count,
            "spam_rapid_fire": pre_spam_rapid_fire,
        }
        result = await self._graph.ainvoke(state)
        reply = result.get("reply", "")
        trace_id = str(uuid.uuid4())
        trace = list(result.get("trace") or [])
        trace_source = {
            "sender": sender,
            "text": text,
            "eventType": "chat",
            "isRoundActive": bool(is_round_active),
        }
        await self._emit_trace(
            trace_id=trace_id,
            trace=trace,
            source=trace_source,
            generated_reply=reply,
            emit_trace=emit_trace,
        )
        if reply:
            if self.config.trace_enabled:
                logger.info("[bot.trace] chat.reply sending len=%d (graph)", len(reply))
            await send_chat_util(
                self,
                text=reply,
                send_chat=send_chat,
                trace=trace,
                trace_source=trace_source,
            )

    async def _emit_trace(
        self,
        *,
        trace_id: str,
        trace: list[dict],
        source: dict,
        generated_reply: str,
        emit_trace: TraceEmitFn | None,
    ) -> None:
        if emit_trace is None:
            return
        await emit_trace(
            trace_id=trace_id,
            source=source,
            trace=trace,
            generated_reply=generated_reply,
        )

    def record_message(self, *, sender: str, text: str, is_bot: bool) -> None:
        self._recent_turns.append(ChatTurn(sender=sender, text=text, is_bot=is_bot))

    # send_chat moved to engine_utils.send_chat
