from __future__ import annotations

from dataclasses import dataclass
import logging

from ..config import BotConfig
from .langchain_integration import get_chat_model, get_embeddings_model
from .routers import LLMChatRouter, OpenAILLMChatRouter
from .response_generator import BotResponseGenerator, OpenAIBotResponseGenerator
from .memory.semantic import PostgresSemanticMemoryService, SemanticMemoryService
from ..player_stats import InMemoryPlayerStatsService, PlayerStatsService
from .memory.extraction import HeuristicMemoryExtractionService, MemoryExtractionService
from .memory.retrieval_plan_service import (
    MemoryRetrievalPlanService,
    NullMemoryRetrievalPlanService,
    OpenAIMemoryRetrievalPlanService,
)
from .memory.write_decision_service import (
    HeuristicMemoryWriteDecisionService,
    MemoryWriteDecisionService,
    OpenAIMemoryWriteDecisionService,
)

logger = logging.getLogger("uvicorn.error")


@dataclass
class BotChatDependencies:
    llm_router: LLMChatRouter | None
    semantic_memory: SemanticMemoryService
    player_stats: PlayerStatsService
    response_generator: BotResponseGenerator | None
    memory_extractor: MemoryExtractionService
    memory_retrieval_planning: MemoryRetrievalPlanService
    memory_write_decision: MemoryWriteDecisionService
    chat_router_model: object | None
    chat_response_model: object | None
    embeddings_model: object | None

    @classmethod
    def build(
        cls,
        config: BotConfig,
        *,
        llm_router: LLMChatRouter | None = None,
        semantic_memory: SemanticMemoryService | None = None,
        player_stats: PlayerStatsService | None = None,
        response_generator: BotResponseGenerator | None = None,
        memory_extractor: MemoryExtractionService | None = None,
    ) -> "BotChatDependencies":
        chat_router_model = get_chat_model(
            model=config.llm_router_model,
            temperature=0.1,
        )
        chat_response_model = get_chat_model(
            model=config.llm_response_model,
            temperature=0.5,
        )
        embeddings_model = get_embeddings_model(model=config.embedding_model)

        resolved_llm_router = llm_router
        if resolved_llm_router is None and chat_router_model is not None:
            resolved_llm_router = OpenAILLMChatRouter(
                model=config.llm_router_model,
                trace_enabled=config.trace_enabled,
            )

        if semantic_memory is not None:
            resolved_semantic_memory = semantic_memory
        else:
            if not config.database_url:
                raise RuntimeError(
                    "DATABASE_URL or SUPABASE_DATABASE_URL must be set (Supabase → Project Settings → "
                    "Database → Connection string → URI)."
                )
            resolved_semantic_memory = PostgresSemanticMemoryService(
                dsn=config.database_url,
                embedding_model=config.embedding_model,
                llm_client=None,
                trace_enabled=config.trace_enabled,
            )

        resolved_player_stats = player_stats or InMemoryPlayerStatsService()

        resolved_response_generator = response_generator
        if resolved_response_generator is None and chat_response_model is not None:
            resolved_response_generator = OpenAIBotResponseGenerator(
                model=config.llm_response_model,
                bot_name=config.name,
                trace_enabled=config.trace_enabled,
            )

        resolved_memory_extractor = memory_extractor or HeuristicMemoryExtractionService()

        if chat_router_model is not None:
            resolved_memory_planning: MemoryRetrievalPlanService = OpenAIMemoryRetrievalPlanService(
                model=config.llm_router_model,
                trace_enabled=config.trace_enabled,
            )
            resolved_memory_write: MemoryWriteDecisionService = OpenAIMemoryWriteDecisionService(
                model=config.llm_router_model,
                trace_enabled=config.trace_enabled,
            )
        else:
            resolved_memory_planning = NullMemoryRetrievalPlanService()
            resolved_memory_write = HeuristicMemoryWriteDecisionService(resolved_memory_extractor)

        if resolved_llm_router is None or resolved_response_generator is None:
            logger.warning("Bot chat replies disabled: LLM router/response generator unavailable")

        return cls(
            llm_router=resolved_llm_router,
            semantic_memory=resolved_semantic_memory,
            player_stats=resolved_player_stats,
            response_generator=resolved_response_generator,
            memory_extractor=resolved_memory_extractor,
            memory_retrieval_planning=resolved_memory_planning,
            memory_write_decision=resolved_memory_write,
            chat_router_model=chat_router_model,
            chat_response_model=chat_response_model,
            embeddings_model=embeddings_model,
        )

