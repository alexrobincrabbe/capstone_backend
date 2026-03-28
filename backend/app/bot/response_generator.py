from __future__ import annotations

import logging
import re

from .llm_client import OpenAIClient
from .models import ChatRoute, ReplyContext
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
        client: OpenAIClient,
        model: str,
        bot_name: str,
        trace_enabled: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.bot_name = bot_name
        self.trace_enabled = trace_enabled

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
        text = await self.client.chat_text(
            model=self.model,
            system=system_prompt,
            user=prompt,
        )
        if text:
            if self.trace_enabled:
                logger.info("[bot.trace] response.output %r", text[:240])
            return self._sanitize_output(text)
        if context.route == ChatRoute.SIMPLE_REPLY:
            return f"{context.username}, good call. Keep tapping."
        return ""

    def _sanitize_output(self, text: str) -> str:
        cleaned = text.strip()
        # Remove accidental speaker label prefixes like "TapBot: ...".
        pattern = re.compile(rf"^\s*{re.escape(self.bot_name)}\s*:\s*", re.IGNORECASE)
        cleaned = pattern.sub("", cleaned)
        return cleaned.strip()
