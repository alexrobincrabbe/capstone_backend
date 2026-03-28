from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import time
from pathlib import Path

from .llm_client import OpenAIClient
from .models import SemanticMemoryRecord

logger = logging.getLogger("uvicorn.error")


class SemanticMemoryService:
    async def retrieve_relevant_memories(
        self, *, username: str, query: str, limit: int = 3
    ) -> list[SemanticMemoryRecord]:
        raise NotImplementedError

    async def store_memory(
        self, *, username: str, memory_text: str, metadata: dict[str, str] | None = None
    ) -> None:
        raise NotImplementedError

    async def has_memories(self, *, username: str) -> bool:
        raise NotImplementedError


class SQLiteSemanticMemoryService(SemanticMemoryService):
    def __init__(
        self,
        *,
        db_path: str,
        embedding_model: str,
        llm_client: OpenAIClient,
        trace_enabled: bool = False,
    ) -> None:
        self.db_path = str(Path(db_path))
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.trace_enabled = trace_enabled
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    memory_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            cols = [row[1] for row in conn.execute("PRAGMA table_info(semantic_memories)").fetchall()]
            if "created_at" not in cols:
                conn.execute("ALTER TABLE semantic_memories ADD COLUMN created_at REAL NOT NULL DEFAULT 0")
            conn.commit()

    async def store_memory(
        self, *, username: str, memory_text: str, metadata: dict[str, str] | None = None
    ) -> None:
        cleaned_text = memory_text.strip()
        cleaned_metadata = metadata or {}
        metadata_json = json.dumps(cleaned_metadata)
        embedding = await self.llm_client.embedding(model=self.embedding_model, text=cleaned_text)
        if embedding is None:
            # Fail gracefully when API key/client is unavailable.
            embedding = []
        created_at = time.time()

        def _write() -> None:
            with self._connect() as conn:
                # Keep only one rolling last-seen memory per user.
                if cleaned_metadata.get("source") == "last_seen":
                    conn.execute(
                        """
                        DELETE FROM semantic_memories
                        WHERE username = ?
                          AND memory_text LIKE 'Last seen:%'
                        """,
                        (username,),
                    )
                if cleaned_metadata.get("source") == "round_summary":
                    conn.execute(
                        """
                        DELETE FROM semantic_memories
                        WHERE username = ?
                          AND metadata_json = ?
                        """,
                        (username, metadata_json),
                    )
                # Avoid exact duplicate rows for same user/text/source.
                existing = conn.execute(
                    """
                    SELECT 1
                    FROM semantic_memories
                    WHERE username = ?
                      AND memory_text = ?
                      AND metadata_json = ?
                    LIMIT 1
                    """,
                    (username, cleaned_text, metadata_json),
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return
                conn.execute(
                    """
                    INSERT INTO semantic_memories(username, memory_text, metadata_json, embedding_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
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
        self, *, username: str, query: str, limit: int = 3
    ) -> list[SemanticMemoryRecord]:
        query_embedding = await self.llm_client.embedding(model=self.embedding_model, text=query.strip())

        def _read() -> list[tuple[str, str, str, float]]:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT memory_text, metadata_json, embedding_json, created_at
                    FROM semantic_memories
                    WHERE username = ?
                    """,
                    (username,),
                ).fetchall()
            return [(row[0], row[1], row[2], float(row[3] or 0.0)) for row in rows]

        rows = await asyncio.to_thread(_read)
        if self.trace_enabled:
            logger.info("[bot.trace] memory.retrieve user=%s candidates=%d", username, len(rows))
        records_with_score: list[tuple[float, SemanticMemoryRecord]] = []
        for memory_text, metadata_json, embedding_json, created_at in rows:
            record = SemanticMemoryRecord(
                username=username,
                memory_text=memory_text,
                metadata=json.loads(metadata_json or "{}"),
                created_at=created_at,
            )
            embedding = json.loads(embedding_json or "[]")
            score = self._cosine_similarity(query_embedding, embedding)
            records_with_score.append((score, record))

        if query_embedding is None:
            # If no embedding available, fallback to recency by insertion order from query.
            if self.trace_enabled:
                logger.info("[bot.trace] memory.retrieve no_query_embedding fallback=recency")
            return [record for _, record in records_with_score[-limit:]]

        records_with_score.sort(key=lambda item: item[0], reverse=True)
        if self.trace_enabled:
            top_scores = [round(score, 4) for score, _ in records_with_score[:limit]]
            logger.info(
                "[bot.trace] memory.retrieve top_scores=%s limit=%d",
                top_scores,
                limit,
            )
        return [record for _, record in records_with_score[:limit]]

    async def has_memories(self, *, username: str) -> bool:
        def _read() -> bool:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT 1
                    FROM semantic_memories
                    WHERE username = ?
                    LIMIT 1
                    """,
                    (username,),
                ).fetchone()
            return row is not None

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
