from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import random
import logging
import re
import time

from .config import BotConfig, ChatSendFn
from .llm_client import OpenAIClient
from .memory_extraction import HeuristicMemoryExtractionService, MemoryExtractionService
from .models import ChatRoute, ChatTurn, ReplyContext, RouteDecision
from .player_stats import InMemoryPlayerStatsService, PlayerStatsService
from .reply_policy import BotReplyPolicy
from .response_generator import BotResponseGenerator, OpenAIBotResponseGenerator
from .routers import LLMChatRouter, OpenAILLMChatRouter
from .semantic_memory import SQLiteSemanticMemoryService, SemanticMemoryService

logger = logging.getLogger("uvicorn.error")


class BotChatEngine:
    def __init__(
        self,
        config: BotConfig,
        *,
        reply_policy: BotReplyPolicy | None = None,
        llm_router: LLMChatRouter | None = None,
        semantic_memory: SemanticMemoryService | None = None,
        player_stats: PlayerStatsService | None = None,
        response_generator: BotResponseGenerator | None = None,
        memory_extractor: MemoryExtractionService | None = None,
    ) -> None:
        self.config = config
        self.llm_client = OpenAIClient()
        self.reply_policy = reply_policy or BotReplyPolicy(config)
        self.llm_router = llm_router if llm_router is not None else (
            OpenAILLMChatRouter(
                client=self.llm_client,
                model=config.llm_router_model,
                trace_enabled=config.trace_enabled,
            )
            if self.llm_client.is_available
            else None
        )
        self.semantic_memory = semantic_memory or SQLiteSemanticMemoryService(
            db_path=config.semantic_memory_db_path,
            embedding_model=config.embedding_model,
            llm_client=self.llm_client,
            trace_enabled=config.trace_enabled,
        )
        self.player_stats = player_stats or InMemoryPlayerStatsService()
        self.response_generator = response_generator if response_generator is not None else (
            OpenAIBotResponseGenerator(
                client=self.llm_client,
                model=config.llm_response_model,
                bot_name=config.name,
                trace_enabled=config.trace_enabled,
            )
            if self.llm_client.is_available
            else None
        )
        if self.llm_router is None or self.response_generator is None:
            logger.warning("Bot chat replies disabled: LLM router/response generator unavailable")
        self.memory_extractor = memory_extractor or HeuristicMemoryExtractionService()
        self._recent_turns: deque[ChatTurn] = deque(maxlen=config.recent_message_limit)
        self._repeat_message_state: dict[str, tuple[str, int, float, float]] = {}
        self._last_round_outcome: dict[str, object] | None = None
        self._deferred_messages: list[tuple[str, str, int]] = []
        self._last_greeted_at: dict[str, float] = {}
        self._last_goodbye_replied_at: dict[str, float] = {}
        self._rejoin_window_seconds = 20 * 60
        self._session_started_at = time.time()
        if self.config.trace_enabled:
            logger.info(
                "[bot.trace] init recent_limit=%d router_model=%s response_model=%s embedding_model=%s",
                self.config.recent_message_limit,
                self.config.llm_router_model,
                self.config.llm_response_model,
                self.config.embedding_model,
            )

    async def on_player_joined(self, username: str, send_chat: ChatSendFn) -> None:
        event_text = f"EVENT: player_joined username={username}"
        if self.config.trace_enabled:
            logger.info("[bot.trace] event.enqueue type=player_joined user=%s", username)
        self.record_message(sender="system", text=event_text, is_bot=False)
        key = username.strip().lower()
        now = time.time()
        last_greet_ts = self._last_greeted_at.get(key)
        if last_greet_ts is not None and (now - last_greet_ts) <= self._rejoin_window_seconds:
            if self.config.trace_enabled:
                logger.info(
                    "[bot.trace] event.player_joined recent_rejoin user=%s age=%.1fs",
                    username,
                    now - last_greet_ts,
                )
            if self.reply_policy.allow_event_announcement(min_interval_seconds=1.0):
                await self._send_chat(text=f"wb {username}", send_chat=send_chat)
                self._last_greeted_at[key] = time.time()
            else:
                if self.config.trace_enabled:
                    logger.info("[bot.trace] event.player_joined wb_skipped_by_cooldown user=%s", username)
            return

        replied = await self._process_event_message(
            event_text=event_text,
            subject_username=username,
            send_chat=send_chat,
        )
        if replied:
            self._last_greeted_at[key] = time.time()

    async def on_round_started(self, send_chat: ChatSendFn) -> None:
        if self.config.trace_enabled:
            logger.info("[bot.trace] event.enqueue type=round_started")
        self.record_message(sender="system", text="EVENT: round_started", is_bot=False)

    async def on_round_ended(self, send_chat: ChatSendFn, room_state: dict | None = None) -> None:
        summary = await self._summarize_round(room_state)
        if self.config.trace_enabled:
            logger.info("[bot.trace] round.end summary=%r", summary)
        event_text = (
            f"EVENT: round_ended summary={summary}"
            if summary is not None
            else "EVENT: round_ended"
        )
        if self.config.trace_enabled:
            logger.info("[bot.trace] event.enqueue type=round_ended payload=%r", event_text[:200])
        self.record_message(sender="system", text=event_text, is_bot=False)
        await self._process_event_message(event_text=event_text, subject_username=None, send_chat=send_chat)
        await self._flush_deferred_messages(send_chat=send_chat)

    async def on_chat_message(
        self,
        *,
        sender: str,
        text: str,
        is_round_active: bool,
        participant_count: int,
        send_chat: ChatSendFn,
    ) -> None:
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
            self._deferred_messages.append((sender, text, participant_count))
            if self.config.trace_enabled:
                logger.info(
                    "[bot.trace] chat.deferred_during_round sender=%s participants=%d queue_size=%d text=%r",
                    sender,
                    participant_count,
                    len(self._deferred_messages),
                    text[:160],
                )
            return

        normalized = text.strip().lower()
        now = time.time()
        last_text, count, last_ts, last_warn_ts = self._repeat_message_state.get(
            sender, ("", 0, 0.0, 0.0)
        )
        if normalized and last_text == normalized and (now - last_ts) < 30.0:
            count += 1
        else:
            count = 1
        self._repeat_message_state[sender] = (normalized, count, now, last_warn_ts)
        if count >= 2:
            if self.config.trace_enabled:
                logger.info("[bot.trace] chat.spam repeat_count=%d text=%r", count, normalized[:120])
            if count >= 5 and (now - last_warn_ts) > 45.0:
                self._repeat_message_state[sender] = (normalized, count, now, now)
                await self._send_chat(
                    text=f"{sender}, please stop spamming the chat.",
                    send_chat=send_chat,
                )
            return

        # Deterministic goodbye handling so "bye/byee/gtg" is not silently ignored.
        goodbye_reply = self._goodbye_reply_for(text=text)
        if goodbye_reply is not None:
            sender_key = sender.strip().lower()
            last_goodbye = self._last_goodbye_replied_at.get(sender_key, 0.0)
            if (now - last_goodbye) >= 30.0:
                if self.config.trace_enabled:
                    logger.info("[bot.trace] chat.goodbye_reply user=%s", sender)
                await self._send_chat(text=goodbye_reply, send_chat=send_chat)
                self._last_goodbye_replied_at[sender_key] = time.time()
            elif self.config.trace_enabled:
                logger.info("[bot.trace] chat.goodbye_skip_recent user=%s", sender)
            return

        if self._is_other_player_private_question(sender=sender, text=text):
            if self.config.trace_enabled:
                logger.info("[bot.trace] privacy.ignore_other_player_question sender=%s", sender)
            return

        is_directed = self._is_directed_message(
            text=text,
            participant_count=participant_count,
        )

        if not self.reply_policy.should_consider_chat_reply(
            is_round_active=is_round_active,
            is_directed=is_directed,
        ):
            if self.config.trace_enabled:
                logger.info("[bot.trace] chat.gate blocked is_directed=%s", is_directed)
            return

        if self.llm_router is None:
            if self.config.trace_enabled:
                logger.info("[bot.trace] chat.router no_decision")
            return
        recent_turns = [f"{turn.sender}: {turn.text}" for turn in self._recent_turns]
        decision = await self.llm_router.classify(
            text=text,
            username=sender,
            is_round_active=is_round_active,
            recent_turns=recent_turns[-self.config.recent_message_limit :],
        )
        if decision.route == ChatRoute.IGNORE:
            if self.config.trace_enabled:
                logger.info("[bot.trace] chat.router ignored")
            return

        memories = []
        stats = None
        history = []
        has_memory_lookup = False
        if decision.route in {ChatRoute.MEMORY_REPLY, ChatRoute.MEMORY_UPDATE_AND_REPLY}:
            has_memory_lookup = True
            memories = await self.semantic_memory.retrieve_relevant_memories(
                username=sender,
                query=decision.memory_query or text,
                limit=3,
            )
            if self.config.trace_enabled:
                logger.info(
                    "[bot.trace] decision.memory retrieved=%d snippets=%r",
                    len(memories),
                    [m.memory_text[:120] for m in memories],
                )
            # Add lightweight stats context for "remember me"/history style turns.
            stats = await self.player_stats.get_summary(username=sender)
        if decision.route == ChatRoute.GAME_STATS_REPLY:
            stats = await self.player_stats.get_summary(username=sender)
        if decision.route == ChatRoute.FULL_HISTORY_REPLY:
            history = list(self._recent_turns)

        context = ReplyContext(
            username=sender,
            user_message=text,
            route=decision.route,
            memories=memories,
            stats=stats,
            recent_turns=history,
        )
        remember_reply = self._remember_me_reply(
            username=sender,
            text=text,
            memories=memories,
            stats=stats,
        )
        if remember_reply:
            if self.config.trace_enabled:
                logger.info("[bot.trace] chat.remember_reply used")
            await self._send_chat(text=remember_reply, send_chat=send_chat)
            if decision.should_store_memory or decision.route == ChatRoute.MEMORY_UPDATE_AND_REPLY:
                await self._maybe_store_memory(
                    username=sender,
                    text=text,
                    existing_memories=memories if has_memory_lookup else None,
                )
            return

        if self.response_generator is None:
            return
        reply = await self.response_generator.generate(context=context)
        if reply:
            reply = self._avoid_false_familiarity(
                reply=reply,
                stats=stats,
                memories=memories,
            )
            if self.config.trace_enabled:
                logger.info("[bot.trace] chat.reply sending len=%d", len(reply))
            await self._send_chat(text=reply, send_chat=send_chat)

        should_force_store = self._should_force_memory_store(text)
        if decision.should_store_memory or decision.route == ChatRoute.MEMORY_UPDATE_AND_REPLY or should_force_store:
            if self.config.trace_enabled and should_force_store and not decision.should_store_memory:
                logger.info("[bot.trace] memory.force_store reason=heuristic text=%r", text[:160])
            await self._maybe_store_memory(
                username=sender,
                text=text,
                existing_memories=memories if has_memory_lookup else None,
            )

    def record_message(self, *, sender: str, text: str, is_bot: bool) -> None:
        self._recent_turns.append(ChatTurn(sender=sender, text=text, is_bot=is_bot))

    async def _maybe_store_memory(
        self,
        *,
        username: str,
        text: str,
        existing_memories: list | None = None,
    ) -> None:
        if existing_memories is None:
            has_existing = await self.semantic_memory.has_memories(username=username)
            existing_count = 1 if has_existing else 0
        else:
            existing_count = len(existing_memories)
        if existing_count == 0:
            # Bootstrap user memory quickly when we know almost nothing yet.
            await self.semantic_memory.store_memory(
                username=username,
                memory_text=f"Last seen: {self._utc_now_text()}",
                metadata={"source": "last_seen"},
            )
            profile_memories = self._extract_profile_memories(username=username, text=text)
            for memory_text in profile_memories:
                await self.semantic_memory.store_memory(
                    username=username,
                    memory_text=memory_text,
                    metadata={"source": "profile_bootstrap"},
                )
            if not profile_memories and len(text.strip()) >= 8:
                await self.semantic_memory.store_memory(
                    username=username,
                    memory_text=f"{username} said on first chat: {text.strip()[:180]}",
                    metadata={"source": "first_chat_bootstrap"},
                )

        extracted = await self.memory_extractor.extract_memory(username=username, user_message=text)
        if extracted:
            await self.semantic_memory.store_memory(
                username=username,
                memory_text=extracted,
                metadata={"source": "chat"},
            )
        # Keep a rolling last-seen breadcrumb for future "remember me" queries.
        await self.semantic_memory.store_memory(
            username=username,
            memory_text=f"Last seen: {self._utc_now_text()}",
            metadata={"source": "last_seen"},
        )

    async def _send_chat(self, *, text: str, send_chat: ChatSendFn) -> None:
        final_text = self._humanize_text(text)
        await send_chat(sender=self.config.name, text=final_text, is_bot=True, system=False)
        self.reply_policy.mark_sent()
        self.record_message(sender=self.config.name, text=final_text, is_bot=True)

    def _style(self) -> str:
        personality = (self.config.personality or "").lower()
        if "sarcast" in personality or "snark" in personality:
            return "sarcastic"
        if "competitive" in personality or "win" in personality:
            return "competitive"
        if "formal" in personality or "polite" in personality:
            return "formal"
        if "chaos" in personality or "wild" in personality:
            return "chaotic"
        return "friendly"

    def _is_directed_message(self, *, text: str, participant_count: int) -> bool:
        # If only user+bot are in the room, assume the message targets the bot.
        if participant_count <= 2:
            return True
        lowered = text.lower()
        bot_name = self.config.name.lower()
        if bot_name in lowered:
            return True
        if re.search(r"\b(you|your|u)\b", lowered):
            return True
        return False

    async def _summarize_round(self, room_state: dict | None) -> str | None:
        if room_state is None:
            return "Round ended. GG."
        scores = room_state.get("scores")
        if not isinstance(scores, dict) or not scores:
            return "Round ended. GG."

        normalized_scores: dict[str, int] = {}
        for name, score in scores.items():
            if isinstance(name, str):
                try:
                    normalized_scores[name] = int(score)
                except (TypeError, ValueError):
                    continue
        if not normalized_scores:
            return "Round ended. GG."

        bot_name = self.config.name
        bot_score = normalized_scores.get(bot_name)
        top_score = max(normalized_scores.values())
        winners = sorted(name for name, score in normalized_scores.items() if score == top_score)
        self._last_round_outcome = {
            "bot_name": bot_name,
            "bot_score": bot_score,
            "top_score": top_score,
            "winners": winners,
            "scores": normalized_scores,
        }

        # Keep player-vs-bot stats structured and separate from semantic memory.
        if bot_score is not None:
            for name, score in normalized_scores.items():
                if name == bot_name:
                    continue
                if score > bot_score:
                    await self.player_stats.record_result(username=name, result="win")
                elif score < bot_score:
                    await self.player_stats.record_result(username=name, result="loss")
                await self._store_round_summary_memory(username=name, bot_name=bot_name)

        if bot_score is None:
            return f"top={top_score}"
        if bot_name in winners and len(winners) == 1:
            return f"bot_won score={bot_score}"
        if bot_name in winners:
            return f"tie score={top_score}"
        leader = winners[0] if winners else "someone"
        return f"winner={leader} top={top_score} bot={bot_score}"

    async def _process_event_message(
        self,
        *,
        event_text: str,
        subject_username: str | None,
        send_chat: ChatSendFn,
    ) -> bool:
        if self.llm_router is None or self.response_generator is None:
            if self.config.trace_enabled:
                logger.info("[bot.trace] event.skip no_router_or_generator event=%r", event_text[:160])
            return False
        if not self.reply_policy.allow_event_announcement(min_interval_seconds=1.0):
            if self.config.trace_enabled:
                logger.info("[bot.trace] event.skip cooldown event=%r", event_text[:160])
            return False
        recent_turns = [f"{turn.sender}: {turn.text}" for turn in self._recent_turns]
        if self.config.trace_enabled:
            logger.info(
                "[bot.trace] event.route input event=%r subject=%s turns=%d",
                event_text[:200],
                subject_username,
                len(recent_turns),
            )
        decision = await self.llm_router.classify(
            text=event_text,
            username=subject_username or "system",
            is_round_active=False,
            recent_turns=recent_turns[-self.config.recent_message_limit :],
        )
        if self.config.trace_enabled:
            logger.info(
                "[bot.trace] event.route decision route=%s memory_query=%r store=%s",
                decision.route.value,
                (decision.memory_query or "")[:120],
                decision.should_store_memory,
            )
        if decision.route == ChatRoute.IGNORE:
            return False

        memories = []
        stats = None
        history = []
        if subject_username:
            if decision.route in {ChatRoute.MEMORY_REPLY, ChatRoute.MEMORY_UPDATE_AND_REPLY}:
                memories = await self.semantic_memory.retrieve_relevant_memories(
                    username=subject_username,
                    query=decision.memory_query or event_text,
                    limit=3,
                )
                if self.config.trace_enabled:
                    logger.info(
                        "[bot.trace] event.memory retrieved=%d snippets=%r",
                        len(memories),
                        [m.memory_text[:120] for m in memories],
                    )
            if decision.route == ChatRoute.GAME_STATS_REPLY:
                stats = await self.player_stats.get_summary(username=subject_username)
                if self.config.trace_enabled:
                    logger.info("[bot.trace] event.stats loaded user=%s", subject_username)
        if decision.route == ChatRoute.FULL_HISTORY_REPLY:
            history = list(self._recent_turns)
            if self.config.trace_enabled:
                logger.info("[bot.trace] event.history turns=%d", len(history))

        context = ReplyContext(
            username=subject_username or "system",
            user_message=event_text,
            route=decision.route,
            memories=memories,
            stats=stats,
            recent_turns=history,
        )
        reply = await self.response_generator.generate(context=context)
        if self.config.trace_enabled:
            logger.info("[bot.trace] event.response generated len=%d", len(reply or ""))
        if reply:
            await self._send_chat(text=reply, send_chat=send_chat)
            return True
        return False

    async def _flush_deferred_messages(self, *, send_chat: ChatSendFn) -> None:
        if not self._deferred_messages:
            if self.config.trace_enabled:
                logger.info("[bot.trace] deferred.flush count=0")
            return
        pending = list(self._deferred_messages)
        self._deferred_messages.clear()
        if self.config.trace_enabled:
            logger.info("[bot.trace] deferred.flush count=%d", len(pending))
        for sender, text, participant_count in pending:
            if self.config.trace_enabled:
                logger.info(
                    "[bot.trace] deferred.replay sender=%s participants=%d text=%r",
                    sender,
                    participant_count,
                    text[:160],
                )
            await self.on_chat_message(
                sender=sender,
                text=text,
                is_round_active=False,
                participant_count=participant_count,
                send_chat=send_chat,
            )

    def _humanize_text(self, text: str) -> str:
        out = text.strip()
        if not out:
            return out

        # Keep light random imperfection so replies feel less machine-perfect.
        if random.random() < 0.35:
            out = re.sub(r"[.!?,;:]+$", "", out)
        if random.random() < 0.25:
            out = out.lower()
        if random.random() < 0.2:
            out = self._add_typo(out)
        return out

    def _add_typo(self, text: str) -> str:
        words = text.split()
        candidate_idxs = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
        if not candidate_idxs:
            return text
        idx = random.choice(candidate_idxs)
        word = words[idx]
        # Swap two adjacent letters.
        j = random.randint(0, len(word) - 2)
        chars = list(word)
        chars[j], chars[j + 1] = chars[j + 1], chars[j]
        words[idx] = "".join(chars)
        return " ".join(words)

    @staticmethod
    def _utc_now_text() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _extract_profile_memories(self, *, username: str, text: str) -> list[str]:
        memories: list[str] = []
        stripped = text.strip()
        lowered = stripped.lower()

        preferred_name_match = re.search(
            r"\b(call me|my name is|i(?:\s*am|'m))\s+([A-Za-z][A-Za-z0-9_-]{1,31})\b",
            stripped,
            re.IGNORECASE,
        )
        if preferred_name_match:
            preferred = preferred_name_match.group(2)
            # Avoid storing obvious non-name tokens.
            if preferred.lower() not in {"a", "an", "the", "male", "female", "man", "woman", "guy", "girl"}:
                memories.append(f"Preferred name: {preferred}")

        if re.search(r"\b(i(?:\s*am|'m)\s+(male|a man|man|guy))\b", lowered):
            memories.append("Gender: male")
        elif re.search(r"\b(i(?:\s*am|'m)\s+(female|a woman|woman|girl))\b", lowered):
            memories.append("Gender: female")

        if self.config.trace_enabled and memories:
            logger.info("[bot.trace] memory.profile_extracted user=%s memories=%r", username, memories)
        return memories

    async def _store_round_summary_memory(self, *, username: str, bot_name: str) -> None:
        summary = await self.player_stats.get_summary(username=username)
        wins = int(summary.wins_vs_bot or 0)
        losses = int(summary.losses_vs_bot or 0)
        total = wins + losses
        memory_text = (
            f"Results vs {bot_name}: wins={wins}, losses={losses}, ties=0, total={total}."
        )
        await self.semantic_memory.store_memory(
            username=username,
            memory_text=memory_text,
            metadata={"source": "round_summary"},
        )

    def _should_force_memory_store(self, text: str) -> bool:
        lowered = text.strip().lower()
        if len(lowered) < 10:
            return False
        if re.search(r"\b(i|my|me)\b", lowered) is None:
            return False
        # Capture lightweight personal facts/events even when router picks simple_reply.
        return re.search(
            r"\b(exam|test|interview|job|work|school|college|university|stressed|stressful|anxious|sick|ill|family|birthday|travel|vacation|moved|graduat)\w*\b",
            lowered,
        ) is not None

    def _goodbye_reply_for(self, *, text: str) -> str | None:
        lowered = text.strip().lower()
        if not lowered:
            return None
        if re.search(r"\b(bye|byee+|goodbye|cya|see ya|see you|gtg|g2g|i have to go|gotta go|catch you later)\b", lowered):
            return "alright, catch you later"
        return None

    def _remember_me_reply(
        self,
        *,
        username: str,
        text: str,
        memories: list,
        stats,
    ) -> str | None:
        lowered = text.strip().lower()
        if not re.search(r"\b(remember me|do you remember me|you remember me|know me)\b", lowered):
            return None

        has_stats = False
        if stats is not None:
            wins = int(getattr(stats, "wins_vs_bot", 0) or 0)
            losses = int(getattr(stats, "losses_vs_bot", 0) or 0)
            has_stats = (wins + losses) > 0
        has_prior_session_memory = any(
            (getattr(m, "created_at", None) or 0.0) < self._session_started_at for m in memories
        )
        has_history = has_stats or has_prior_session_memory

        if has_history:
            choices = [
                f"yeah i remember you {username}, good to see you again",
                f"yep, i remember you. we've played before",
                f"i do, we've crossed paths before. good to see you",
            ]
            return random.choice(choices)
        return "not really yet, but i got you now. good to see you"

    def _is_other_player_private_question(self, *, sender: str, text: str) -> bool:
        lowered = text.strip().lower()
        if "?" not in lowered and not re.search(r"\b(who|where|what|when|which|tell me|do you know)\b", lowered):
            return False
        private_topic_patterns = [
            r"\b(where.*live|live.*where|from where|where.*from)\b",
            r"\b(old|age|how old)\b",
            r"\b(real name|full name|last name|surname)\b",
            r"\b(phone|number|email|contact|address)\b",
            r"\b(work|job|company|employer|school|college|university)\b",
            r"\b(single|dating|boyfriend|girlfriend|partner|married)\b",
            r"\b(gender|male|female|sex)\b",
        ]
        if not any(re.search(pattern, lowered) for pattern in private_topic_patterns):
            return False

        sender_lower = sender.strip().lower()
        known_people = {
            turn.sender.strip().lower()
            for turn in self._recent_turns
            if turn.sender.strip() and not turn.is_bot and turn.sender.strip().lower() != "system"
        }
        known_people.discard(sender_lower)
        known_people.discard(self.config.name.strip().lower())
        if not known_people:
            return False

        # If message targets any known participant other than the sender, ignore it.
        return any(name in lowered for name in known_people)

    def _avoid_false_familiarity(self, *, reply: str, stats, memories: list) -> str:
        wins = int(getattr(stats, "wins_vs_bot", 0) or 0) if stats is not None else 0
        losses = int(getattr(stats, "losses_vs_bot", 0) or 0) if stats is not None else 0
        has_game_history = (wins + losses) > 0
        has_round_summary_memory = any(
            isinstance(getattr(m, "metadata", None), dict)
            and (getattr(m, "metadata", {}) or {}).get("source") == "round_summary"
            for m in memories
        )
        has_prior_session_memory = any(
            (getattr(m, "created_at", None) or 0.0) < self._session_started_at
            for m in memories
        )
        if has_game_history or has_round_summary_memory or has_prior_session_memory:
            return reply

        sanitized = reply
        # Prevent "we met before" tone when only same-session bootstrap memories exist.
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
        sanitized = re.sub(r"\s{2,}", " ", sanitized).strip(" ,.!?")
        return sanitized or reply
