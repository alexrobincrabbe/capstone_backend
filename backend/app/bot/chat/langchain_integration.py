from __future__ import annotations

import os
import logging

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore[import]
except Exception:  # pragma: no cover - optional import guard
    ChatOpenAI = None  # type: ignore[assignment]
    OpenAIEmbeddings = None  # type: ignore[assignment]

logger = logging.getLogger("uvicorn.error")


def _read_env(name: str) -> str:
    value = os.getenv(name) or os.getenv(f"\ufeff{name}") or ""
    return value.strip()


def _ensure_openai_key_env() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    bom_key = os.getenv("\ufeffOPENAI_API_KEY")
    if bom_key:
        os.environ["OPENAI_API_KEY"] = bom_key.strip()


def get_chat_model(*, model: str, temperature: float = 0.2):
    if ChatOpenAI is None:
        logger.warning("LangChain ChatOpenAI import unavailable")
        return None
    _ensure_openai_key_env()
    api_key = _read_env("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; ChatOpenAI unavailable")
        return None
    try:
        return ChatOpenAI(model=model, temperature=temperature)
    except Exception:
        logger.exception("Failed to construct ChatOpenAI(model=%s)", model)
        return None


def get_embeddings_model(*, model: str):
    if OpenAIEmbeddings is None:
        logger.warning("LangChain OpenAIEmbeddings import unavailable")
        return None
    _ensure_openai_key_env()
    api_key = _read_env("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; OpenAIEmbeddings unavailable")
        return None
    try:
        return OpenAIEmbeddings(model=model)
    except Exception:
        logger.exception("Failed to construct OpenAIEmbeddings(model=%s)", model)
        return None

