from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from ..config import ChatSendFn

logger = logging.getLogger("uvicorn.error")


async def defer_message(
    engine,
    sender: str,
    text: str,
    participant_count: int,
    *,
    pre_spam_blocked: bool,
    pre_repeat_count: int,
    pre_spam_rapid_fire: bool,
) -> None:
    engine._deferred_messages.append(
        (sender, text, participant_count, pre_spam_blocked, pre_repeat_count, pre_spam_rapid_fire)
    )
    if engine.config.trace_enabled:
        logger.info(
            "[bot.trace] chat.deferred_during_round sender=%s participants=%d queue_size=%d text=%r",
            sender,
            participant_count,
            len(engine._deferred_messages),
            text[:160],
        )


async def send_chat(
    engine,
    *,
    text: str,
    send_chat: ChatSendFn,
    trace: list[dict[str, Any]] | None = None,
    trace_source: dict[str, Any] | None = None,
) -> None:
    final_text = text
    await send_chat(
        sender=engine.config.name,
        text=final_text,
        is_bot=True,
        system=False,
        trace=trace,
        trace_source=trace_source,
    )
    engine._last_bot_message_ts = utc_now_ts()
    engine.record_message(sender=engine.config.name, text=final_text, is_bot=True)


async def flush_deferred_messages(engine, *, send_chat: ChatSendFn) -> None:
    if not engine._deferred_messages:
        if engine.config.trace_enabled:
            logger.info("[bot.trace] deferred.flush count=0")
        return
    pending = list(engine._deferred_messages)
    engine._deferred_messages.clear()
    if engine.config.trace_enabled:
        logger.info("[bot.trace] deferred.flush count=%d", len(pending))
    for sender, text, participant_count, pre_spam_blocked, pre_repeat_count, pre_spam_rapid_fire in pending:
        if engine.config.trace_enabled:
            logger.info(
                "[bot.trace] deferred.replay sender=%s participants=%d text=%r",
                sender,
                participant_count,
                text[:160],
            )
        await engine.on_chat_message(
            sender=sender,
            text=text,
            is_round_active=False,
            participant_count=participant_count,
            send_chat=send_chat,
            pre_spam_blocked=pre_spam_blocked,
            pre_repeat_count=pre_repeat_count,
            pre_spam_rapid_fire=pre_spam_rapid_fire,
        )


def utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def utc_now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()

