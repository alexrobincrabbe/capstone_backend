from __future__ import annotations

import asyncio
import json
import logging
import math
import time

import psycopg

from ..langchain_integration import get_embeddings_model
from ...models import SemanticMemoryRecord

logger = logging.getLogger("uvicorn.error")


class SemanticMemoryService:
    async def retrieve_relevant_memories(
        self,
        *,
        username: str,
        query: str,
        limit: int = 3,
        min_similarity: float | None = None,
    ) -> list[SemanticMemoryRecord]:
        raise NotImplementedError

    async def store_memory(
        self, *, username: str, memory_text: str, metadata: dict[str, str] | None = None
    ) -> None:
        raise NotImplementedError

    async def has_memories(self, *, username: str) -> bool:
        raise NotImplementedError


class PostgresSemanticMemoryService(SemanticMemoryService):
    """Semantic memory stored in PostgreSQL (e.g. Supabase)."""

    def __init__(
        self,
        *,
        dsn: str,
        embedding_model: str,
        llm_client=None,
        trace_enabled: bool = False,
    ) -> None:
        cleaned = (dsn or "").strip()
        if not cleaned:
            raise ValueError(
                "Postgres DSN is empty; set DATABASE_URL or SUPABASE_DATABASE_URL "
                "to your Supabase connection string (URI)."
            )
        self.dsn = cleaned
        self.embedding_model = embedding_model
        self._embeddings = get_embeddings_model(model=embedding_model)
        self.llm_client = llm_client
        self.trace_enabled = trace_enabled
        self._init_db()

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn)

    def _init_db(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS semantic_memories (
                        id BIGSERIAL PRIMARY KEY,
                        username TEXT NOT NULL,
                        memory_text TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        created_at DOUBLE PRECISION NOT NULL DEFAULT 0
                    )
                    """
                )
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'semantic_memories'
                    """
                )
                cols = {row[0] for row in cur.fetchall()}
                if cols and "created_at" not in cols:
                    cur.execute(
                        """
                        ALTER TABLE semantic_memories
                        ADD COLUMN created_at DOUBLE PRECISION NOT NULL DEFAULT 0
                        """
                    )
            conn.commit()

    async def store_memory(
        self, *, username: str, memory_text: str, metadata: dict[str, str] | None = None
    ) -> None:
        cleaned_text = memory_text.strip()
        cleaned_metadata = metadata or {}
        metadata_json = json.dumps(cleaned_metadata)
        embedding = None
        if self._embeddings is not None:
            try:

                def _embed() -> list[float] | None:
                    try:
                        return list(self._embeddings.embed_query(cleaned_text))  # type: ignore[union-attr]
                    except Exception:
                        return None

                embedding = await asyncio.to_thread(_embed)
            except Exception:
                embedding = None
        elif self.llm_client is not None:
            embedding = await self.llm_client.embedding(model=self.embedding_model, text=cleaned_text)
        if embedding is None:
            embedding = []
        created_at = time.time()

        def _write() -> None:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    if cleaned_metadata.get("source") == "last_seen":
                        cur.execute(
                            """
                            DELETE FROM semantic_memories
                            WHERE username = %s
                              AND memory_text LIKE 'Last seen:%%'
                            """,
                            (username,),
                        )
                    if cleaned_metadata.get("source") == "round_summary":
                        cur.execute(
                            """
                            DELETE FROM semantic_memories
                            WHERE username = %s
                              AND metadata_json = %s
                            """,
                            (username, metadata_json),
                        )
                    cur.execute(
                        """
                        SELECT 1
                        FROM semantic_memories
                        WHERE username = %s
                          AND memory_text = %s
                          AND metadata_json = %s
                        LIMIT 1
                        """,
                        (username, cleaned_text, metadata_json),
                    )
                    if cur.fetchone() is not None:
                        conn.commit()
                        return
                    cur.execute(
                        """
                        INSERT INTO semantic_memories(
                            username, memory_text, metadata_json, embedding_json, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            username,
                            cleaned_text,
                            metadata_json,
                            json.dumps(embedding),
                            created_at,
                        ),
                    )
                conn.commit()

        await asyncio.to_thread(_write)
        if self.trace_enabled:
            logger.info(
                "[bot.trace] memory.store user=%s text=%r embedding_dims=%d",
                username,
                memory_text[:160],
                len(embedding),
            )

    async def retrieve_relevant_memories(
        self,
        *,
        username: str,
        query: str,
        limit: int = 3,
        min_similarity: float | None = None,
    ) -> list[SemanticMemoryRecord]:
        query_embedding = None
        if self._embeddings is not None:
            try:

                def _embed_q() -> list[float] | None:
                    try:
                        return list(self._embeddings.embed_query(query.strip()))  # type: ignore[union-attr]
                    except Exception:
                        return None

                query_embedding = await asyncio.to_thread(_embed_q)
            except Exception:
                query_embedding = None
        elif self.llm_client is not None:
            query_embedding = await self.llm_client.embedding(model=self.embedding_model, text=query.strip())

        def _read() -> list[tuple[str, str, str, float]]:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT memory_text, metadata_json, embedding_json, created_at
                        FROM semantic_memories
                        WHERE username = %s
                        """,
                        (username,),
                    )
                    rows = cur.fetchall()
            return [(row[0], row[1], row[2], float(row[3] or 0.0)) for row in rows]

        rows = await asyncio.to_thread(_read)
        if self.trace_enabled:
            logger.info("[bot.trace] memory.retrieve user=%s candidates=%d", username, len(rows))
        records_with_score: list[tuple[float, SemanticMemoryRecord]] = []
        for memory_text, metadata_json, embedding_json, created_at in rows:
            embedding = json.loads(embedding_json or "[]")
            score = self._cosine_similarity(query_embedding, embedding)
            record = SemanticMemoryRecord(
                username=username,
                memory_text=memory_text,
                metadata=json.loads(metadata_json or "{}"),
                created_at=created_at,
                similarity=round(score, 6) if query_embedding else None,
            )
            records_with_score.append((score, record))

        if query_embedding is None:
            if self.trace_enabled:
                logger.info("[bot.trace] memory.retrieve no_query_embedding fallback=recency")
            picked = records_with_score[-limit:]
            return [record for _, record in picked]

        records_with_score.sort(key=lambda item: item[0], reverse=True)
        if min_similarity is not None:
            records_with_score = [(s, r) for s, r in records_with_score if s >= min_similarity]

        if self.trace_enabled:
            top_scores = [round(score, 4) for score, _ in records_with_score[:limit]]
            logger.info(
                "[bot.trace] memory.retrieve top_scores=%s limit=%d min_sim=%s",
                top_scores,
                limit,
                min_similarity,
            )
        return [record for _, record in records_with_score[:limit]]

    async def has_memories(self, *, username: str) -> bool:
        def _read() -> bool:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 1
                        FROM semantic_memories
                        WHERE username = %s
                        LIMIT 1
                        """,
                        (username,),
                    )
                    return cur.fetchone() is not None

        return await asyncio.to_thread(_read)

    @staticmethod
    def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
