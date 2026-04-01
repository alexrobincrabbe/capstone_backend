from __future__ import annotations

import logging
import re

from .langchain_integration import get_chat_model
from ..models import ChatRoute, ReplyContext
from .prompts import (
    response_system_prompt_join,
    response_system_prompt_rich,
    response_system_prompt_simple,
    response_user_prompt,
)

logger = logging.getLogger("uvicorn.error")


class BotResponseGenerator:
    async def generate(self, *, context: ReplyContext) -> str:
        raise NotImplementedError


class OpenAIBotResponseGenerator(BotResponseGenerator):
    def __init__(
        self,
        *,
        client=None,
        model: str,
        bot_name: str,
        trace_enabled: bool = False,
    ) -> None:
        # 'client' kept for backward compatibility; unused after refactor.
        self.model_name = model
        self.bot_name = bot_name
        self.trace_enabled = trace_enabled
        self._model = get_chat_model(model=model, temperature=0.5)

    async def generate(self, *, context: ReplyContext) -> str:
        prompt = response_user_prompt(context=context)
        if context.user_message.startswith("EVENT: player_joined"):
            system_prompt = response_system_prompt_join(bot_name=self.bot_name)
            mode = "join"
        elif context.route == ChatRoute.SIMPLE_REPLY:
            system_prompt = response_system_prompt_simple(bot_name=self.bot_name)
            mode = "simple"
        else:
            system_prompt = response_system_prompt_rich(bot_name=self.bot_name)
            mode = "rich"
        if self.trace_enabled:
            logger.info(
                "[bot.trace] response.context route=%s memories=%d has_stats=%s recent_turns=%d",
                context.route.value,
                len(context.memories),
                context.stats is not None,
                len(context.recent_turns),
            )
            logger.info(
                "[bot.trace] response.system_prompt_mode=%s",
                mode,
            )
            logger.info("[bot.trace] response.prompt %r", prompt[:500])
        text = None
        if self._model is not None:
            try:
                try:
                    from langchain.schema import SystemMessage, HumanMessage  # LC < 0.2
                except Exception:  # pragma: no cover
                    from langchain_core.messages import SystemMessage, HumanMessage  # LC >= 0.2

                messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                result = await self._model.ainvoke(messages)
                text = (getattr(result, "content", None) or "").strip()
            except Exception:
                text = None
        if text:
            if self.trace_enabled:
                logger.info("[bot.trace] response.output %r", text[:240])
            return self._sanitize_output(text)

        # Heuristic fallbacks to avoid silence if LLM returns empty content
        if context.user_message.startswith("EVENT: player_joined"):
            return f"wb {context.username}"
        if context.route == ChatRoute.SIMPLE_REPLY:
            return f"{context.username}, good call. Keep tapping."
        # Generic short acknowledgment
        ack = f"hey {context.username}" if context.username else "hey"
        if self.trace_enabled:
            logger.info("[bot.trace] response.fallback %r", ack)
        return ack

    def _sanitize_output(self, text: str) -> str:
        cleaned = text.strip()
        # Remove accidental speaker label prefixes like "TapBot: ...".
        pattern = re.compile(rf"^\s*{re.escape(self.bot_name)}\s*:\s*", re.IGNORECASE)
        cleaned = pattern.sub("", cleaned)
        return cleaned.strip()

