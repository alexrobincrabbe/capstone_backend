from __future__ import annotations

import json
import logging
import re
from typing import Any

from ...models import MemoryWriteDecision
from ..langchain_integration import get_chat_model
from ..prompts import memory_write_system_prompt, memory_write_user_prompt
from .extraction import MemoryExtractionService

logger = logging.getLogger("uvicorn.error")


def is_redundant_username_memory(memory_text: str, username: str) -> bool:
    """
    True if the proposed memory only restates the chat username (already keyed per-user in storage).
    """
    t = (memory_text or "").strip()
    u = (username or "").strip()
    if not t or not u:
        return False
    esc = re.escape(u)
    patterns = [
        rf"^(the\s+)?user'?s\s+(name|username|display\s*name)\s+is\s+{esc}\.?\s*$",
        rf"^user'?s\s+(name|username)\s+is\s+{esc}\.?\s*$",
        rf"^username\s+is\s+{esc}\.?\s*$",
        rf"^(the\s+)?name\s+is\s+{esc}\.?\s*$",
        rf"^{esc}\s+is\s+(the\s+)?user'?s\s+(name|username)\.?\s*$",
        rf"^{esc}\s+is\s+(their|his|her)\s+username\.?\s*$",
        rf"^(user(name)?|display\s*name)\s*[:=-]\s*{esc}\.?\s*$",
    ]
    return any(re.search(p, t, re.IGNORECASE | re.UNICODE) for p in patterns)


class MemoryWriteDecisionService:
    async def decide(
        self,
        *,
        username: str,
        user_message: str,
        bot_reply: str,
        memories: list[Any],
        stats: Any | None,
    ) -> MemoryWriteDecision:
        raise NotImplementedError


class OpenAIMemoryWriteDecisionService(MemoryWriteDecisionService):
    def __init__(self, *, model: str, trace_enabled: bool = False) -> None:
        self._model_name = model
        self._trace_enabled = trace_enabled
        self._model = get_chat_model(model=model, temperature=0.0)

    async def decide(
        self,
        *,
        username: str,
        user_message: str,
        bot_reply: str,
        memories: list[Any],
        stats: Any | None,
    ) -> MemoryWriteDecision:
        if self._model is None:
            return MemoryWriteDecision(should_write_memory=False, memory_write_text=None)

        mem_lines = []
        for m in memories:
            txt = getattr(m, "memory_text", None) or str(m)
            mem_lines.append(txt)
        human = memory_write_user_prompt(
            username=username,
            user_message=user_message,
            bot_reply=bot_reply,
            recalled_memories=mem_lines,
            stats=stats,
        )
        try:
            try:
                from langchain.schema import SystemMessage, HumanMessage
            except Exception:  # pragma: no cover
                from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=memory_write_system_prompt()),
                HumanMessage(content=human),
            ]
            result = await self._model.ainvoke(messages)
            content = (getattr(result, "content", None) or "").strip()
            payload = json.loads(content) if content else None
        except Exception as exc:
            if self._trace_enabled:
                logger.info("[bot.trace] memory_write.error %s", exc)
            payload = None

        if not payload:
            return MemoryWriteDecision(should_write_memory=False, memory_write_text=None)

        should = bool(payload.get("should_write_memory", False))
        raw = payload.get("memory_write_text")
        text_out = str(raw).strip() if raw not in (None, "") else None

        if self._trace_enabled:
            logger.info(
                "[bot.trace] memory_write.output should=%s text=%r",
                should,
                (text_out or "")[:120],
            )

        if should and not text_out:
            return MemoryWriteDecision(should_write_memory=False, memory_write_text=None)
        if should and text_out and is_redundant_username_memory(text_out, username):
            if self._trace_enabled:
                logger.info("[bot.trace] memory_write.blocked redundant_username text=%r", text_out[:120])
            return MemoryWriteDecision(should_write_memory=False, memory_write_text=None)
        return MemoryWriteDecision(should_write_memory=should, memory_write_text=text_out)


class HeuristicMemoryWriteDecisionService(MemoryWriteDecisionService):
    """Non-LLM path: wraps the heuristic extractor."""

    def __init__(self, extractor: MemoryExtractionService) -> None:
        self._extractor = extractor

    async def decide(
        self,
        *,
        username: str,
        user_message: str,
        bot_reply: str,
        memories: list[Any],
        stats: Any | None,
    ) -> MemoryWriteDecision:
        try:
            extracted = await self._extractor.extract_memory(
                username=username,
                user_message=user_message,
                bot_reply=bot_reply,
                memories=memories,
                stats=stats,
            )
        except TypeError:
            extracted = await self._extractor.extract_memory(  # type: ignore[call-arg]
                username=username,
                user_message=user_message,
            )
        if extracted and str(extracted).strip():
            text = str(extracted).strip()
            if is_redundant_username_memory(text, username):
                return MemoryWriteDecision(should_write_memory=False, memory_write_text=None)
            return MemoryWriteDecision(should_write_memory=True, memory_write_text=text)
        return MemoryWriteDecision(should_write_memory=False, memory_write_text=None)
