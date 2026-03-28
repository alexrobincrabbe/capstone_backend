from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger("uvicorn.error")


def _read_env(name: str) -> str:
    # Some editors save .env with UTF-8 BOM, which can turn OPENAI_API_KEY
    # into \ufeffOPENAI_API_KEY. Support both to avoid silent failures.
    value = os.getenv(name)
    if value is None:
        value = os.getenv(f"\ufeff{name}")
    return (value or "").strip()


class OpenAIClient:
    def __init__(self) -> None:
        self.api_key = _read_env("OPENAI_API_KEY")
        self._client: OpenAI | None = None
        if self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key)
            except Exception as exc:
                logger.warning("Bot LLM disabled: failed to initialize OpenAI client: %s", exc)
        if self._client is None and not self.api_key:
            logger.warning("Bot LLM disabled: OPENAI_API_KEY is missing")
        elif self._client is None:
            logger.warning("Bot LLM disabled: OpenAI client unavailable")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    async def chat_json(self, *, model: str, system: str, user: str) -> dict[str, Any] | None:
        if self._client is None:
            return None

        def _call() -> dict[str, Any] | None:
            completion = self._client.chat.completions.create(
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = completion.choices[0].message.content
            if not content:
                return None
            import json

            return json.loads(content)

        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            logger.warning("OpenAI chat_json failed: %s", exc)
            return None

    async def chat_text(self, *, model: str, system: str, user: str) -> str | None:
        if self._client is None:
            return None

        def _call() -> str | None:
            completion = self._client.chat.completions.create(
                model=model,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return completion.choices[0].message.content

        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            logger.warning("OpenAI chat_text failed: %s", exc)
            return None

    async def embedding(self, *, model: str, text: str) -> list[float] | None:
        if self._client is None:
            return None

        def _call() -> list[float] | None:
            resp = self._client.embeddings.create(model=model, input=text)
            data = resp.data[0].embedding if resp.data else None
            return list(data) if data is not None else None

        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            logger.warning("OpenAI embedding failed: %s", exc)
            return None
