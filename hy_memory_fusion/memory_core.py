"""Memory Core — Unified Qdrant backend.

Provides two high-level APIs:
- remember(): Write raw text → distill → dedup → store
- recall(): Query → multi-signal ranking → dialectic synthesis → answer

Internal APIs:
- hybrid_search(): Low-level Qdrant search with hybrid modes
- get_facts_by_ids(): Batch fetch by fact IDs
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Literal

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from hy_memory_fusion.config import FusionConfig
from hy_memory_fusion.write_pipeline import WritePipeline, ExtractedFact
from hy_memory_fusion.read_pipeline import ReadPipeline, RankedFact

logger = logging.getLogger(__name__)


class MemoryCore:
    """Unified memory backend combining Write and Read pipelines."""

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        qdrant_client: Optional[AsyncQdrantClient] = None,
        llm_client: Optional[AsyncOpenAI] = None,
        embed_client: Optional[AsyncOpenAI] = None,
        reader_client: Optional[AsyncOpenAI] = None,
        writer_client: Optional[AsyncOpenAI] = None,
    ):
        self.config = config or FusionConfig.from_env()

        # Shared embedding client
        self._embed_client = embed_client or AsyncOpenAI(
            base_url=self.config.embedder.base_url,
            api_key=self.config.embedder.api_key,
            timeout=self.config.embedder.timeout,
        )

        # Writer LLM client (SVO extraction)
        self._writer_client = writer_client or AsyncOpenAI(
            base_url=self.config.writer.base_url,
            api_key=self.config.writer.api_key,
            timeout=self.config.writer.timeout,
        )

        # Reader LLM client (synthesis)
        self._reader_client = reader_client or AsyncOpenAI(
            base_url=self.config.reader.base_url,
            api_key=self.config.reader.api_key,
            timeout=self.config.reader.timeout,
        )

        # Generic LLM client (for backward compat)
        self._llm_client = llm_client or self._writer_client

        # Qdrant client
        self._qdrant = qdrant_client or AsyncQdrantClient(url=self.config.qdrant.url)
        self._collection = self.config.qdrant.collection

        # Pipelines
        self._writer = WritePipeline(
            config=self.config,
            llm_client=self._writer_client,
            embed_client=self._embed_client,
        )
        self._reader = ReadPipeline(
            config=self.config,
            llm_client=self._reader_client,
            embed_client=self._embed_client,
        )

        self._initialized = False

    async def initialize(self) -> None:
        """Create Qdrant collection if not exists."""
        collections = await self._qdrant.get_collections()
        names = [c.name for c in collections.collections]

        if self._collection not in names:
            await self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self.config.qdrant.vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection: %s", self._collection)

        self._initialized = True

    async def embed(self, text: str) -> list[float]:
        """Shared embedding method — public API for external callers."""
        return await self._writer.embed(text)

    async def remember(self, text: str, user_id: str = "default") -> list[dict[str, Any]]:
        """Ingest raw text → distill → dedup → store → return stored facts.

        Fetches existing facts from Qdrant for dedup before calling writer.ingest().
        """
        if not self._initialized:
            await self.initialize()

        # Fetch existing facts for dedup — use Qdrant search for efficiency
        existing_facts = []
        if self.config.distillation.enabled:
            try:
                # For small collections (<1000), scroll is fine.
                # For larger collections, each new fact is searched individually
                # against Qdrant in WritePipeline._dedup(), so we fetch a
                # representative sample here for batch dedup.
                all_points = await self._qdrant.scroll(
                    collection_name=self._collection,
                    limit=500,
                    with_payload=True,
                    with_vectors=True,
                )
                for point in all_points[0]:
                    payload = point.payload or {}
                    payload["embedding"] = point.vector
                    existing_facts.append(payload)
                logger.debug("Fetched %d existing facts for dedup", len(existing_facts))
            except Exception as e:
                logger.debug("Could not fetch existing facts for dedup: %s", e)

        # Distill + dedup
        facts = await self._writer.ingest(text, user_id=user_id, existing_facts=existing_facts)

        # Filter by importance threshold
        facts = [f for f in facts if f.importance >= self.config.distillation.importance_threshold]

        if not facts:
            logger.info("No facts above importance threshold")
            return []

        # Store in Qdrant
        points = []
        for fact in facts:
            if not fact.embedding:
                continue
            payload = fact.to_dict()
            payload["user_id"] = user_id
            payload["access_count"] = 0
            points.append(
                PointStruct(
                    id=fact.fact_id,
                    vector=fact.embedding,
                    payload=payload,
                )
            )

        if points:
            await self._qdrant.upsert(
                collection_name=self._collection,
                points=points,
            )
            logger.info("Stored %d facts for user %s", len(points), user_id)

        return [f.to_dict() for f in facts]

    async def recall(
        self,
        query: str,
        user_id: str = "default",
        reasoning_level: str = "low",
    ) -> dict[str, Any]:
        """Query → multi-signal ranking → dialectic synthesis → answer.

        Uses MemoryCore.embed() for query embedding (not writer._embed).
        """
        if not self._initialized:
            await self.initialize()

        # Search
        facts = await self._reader.search(query, self, user_id=user_id)

        # Synthesize
        result = await self._reader.synthesize(query, facts, reasoning_level)
        result["facts"] = [f.to_dict() for f in facts]
        return result

    # ── Vector store interface (used by ReadPipeline.search) ──────────────

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search Qdrant by vector similarity.

        This is the vector_store interface that ReadPipeline.search() calls.
        """
        results = await self._qdrant.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        return [
            {
                "fact_id": point.id,
                "text": point.payload.get("text", ""),
                "score": point.score,
                "embedding": point.vector,
                **{k: v for k, v in (point.payload or {}).items() if k != "text"},
            }
            for point in results
        ]

    async def update_access(self, fact_ids: list[str]) -> None:
        """Increment access counter for retrieved facts.

        Note: This uses read-modify-write without locking. Under concurrent
        access, increments may be lost (acceptable for approximate counters).
        For exact counting, use Qdrant's atomic operations or a separate counter.
        """
        for fid in fact_ids:
            try:
                # Qdrant doesn't have atomic increment, use set_payload with read-modify-write
                points = await self._qdrant.retrieve(
                    collection_name=self._collection,
                    ids=[fid],
                    with_payload=True,
                )
                if points:
                    current = points[0].payload.get("access_count", 0)
                    await self._qdrant.set_payload(
                        collection_name=self._collection,
                        payload={"access_count": current + 1},
                        points=[fid],
                    )
            except Exception as e:
                logger.debug("Access update failed for %s: %s", fid, e)

    # ── Low-level hybrid search ──────────────────────────────────────────

    async def hybrid_search(
        self,
        query: str,
        mode: Literal["semantic", "hybrid", "fusion"] = "hybrid",
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
        user_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Low-level search with mode selection.

        Modes:
        - semantic: Pure vector similarity
        - hybrid: Vector + metadata filters
        - fusion: Multi-query fusion (planned)
        """
        if not self._initialized:
            await self.initialize()

        # Get query embedding
        query_embedding = await self.embed(query)
        if not query_embedding:
            return []

        # Build filter
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if user_id:
                conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
            qdrant_filter = Filter(must=conditions)
        elif user_id:
            qdrant_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

        # Search
        results = await self._qdrant.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        if mode == "semantic":
            return [
                {
                    "fact_id": point.id,
                    "text": point.payload.get("text", ""),
                    "score": point.score,
                    **{k: v for k, v in (point.payload or {}).items() if k != "text"},
                }
                for point in results
            ]

        # hybrid / fusion: use full read pipeline ranking
        raw = [
            {
                "fact_id": point.id,
                "text": point.payload.get("text", ""),
                "score": point.score,
                "embedding": point.payload.get("embedding", []),
                **{k: v for k, v in (point.payload or {}).items() if k not in ("text", "embedding")},
            }
            for point in results
        ]
        ranked = self._reader.rank(query_embedding, raw)
        return [r.to_dict() for r in ranked[:limit]]

    # ── Batch operations ─────────────────────────────────────────────────

    async def get_facts_by_ids(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        """Batch fetch facts by IDs."""
        if not fact_ids:
            return []

        results = await self._qdrant.retrieve(
            collection_name=self._collection,
            ids=fact_ids,
            with_payload=True,
        )

        return [
            {"fact_id": point.id, **(point.payload or {})}
            for point in results
        ]
