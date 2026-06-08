"""Memory Core — Unified memory system combining Write + Read pipelines.

Ties together:
- Qdrant vector store for persistence
- WritePipeline for auto-distillation (Hy-Memory style)
- ReadPipeline for dialectic reasoning (Honcho style)
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

from hy_memory_fusion.config import FusionConfig
from hy_memory_fusion.write_pipeline import WritePipeline, ExtractedFact
from hy_memory_fusion.read_pipeline import ReadPipeline, SearchResult, DialecticResponse

logger = logging.getLogger(__name__)


class MemoryCore:
    """Unified memory system: auto-distill on write, dialectic reasoning on read.

    Usage:
        config = FusionConfig.from_env(".env")
        core = MemoryCore(config)

        # Write: auto-distill conversation into structured facts
        facts = await core.remember("Ulysses prefers dark mode UIs")

        # Read: search + dialectic reasoning
        answer = await core.recall("What UI preferences does the user have?")
    """

    def __init__(self, config: FusionConfig):
        self.config = config

        # LLM clients
        self._llm = OpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url or None,
        )
        self._embed = OpenAI(
            api_key=config.embedder.api_key,
            base_url=config.embedder.base_url,
        )

        # Pipelines
        self._writer = WritePipeline(config, self._llm, self._embed)
        self._reader = ReadPipeline(config, self._llm)

        # Vector store
        self._qdrant: QdrantClient | None = None
        self._collection = config.vector_store.collection_name

    def connect(self) -> None:
        """Connect to Qdrant and ensure collection exists."""
        kwargs: dict[str, Any] = {}
        if self.config.vector_store.url:
            kwargs["url"] = self.config.vector_store.url
        if self.config.vector_store.api_key:
            kwargs["api_key"] = self.config.vector_store.api_key

        self._qdrant = QdrantClient(**kwargs)

        # Ensure collection exists
        collections = [c.name for c in self._qdrant.get_collections().collections]
        if self._collection not in collections:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self.config.vector_store.embedding_dims,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self._collection}")
        else:
            logger.info(f"Using existing collection: {self._collection}")

    def close(self) -> None:
        """Close connections."""
        if self._qdrant:
            self._qdrant.close()

    # ── Write API ────────────────────────────────────────────────

    async def remember(self, text: str, user_id: str = "default") -> list[ExtractedFact]:
        """Ingest text → extract SVO facts → store in Qdrant.

        This is the main write API. It:
        1. Extracts atomic facts (SVO triplets) via LLM
        2. Embeds each fact
        3. Stores in Qdrant with metadata

        Returns the list of extracted facts.
        """
        self._ensure_connected()

        # Extract + embed
        facts = await self._writer.ingest(text, user_id)

        # Store in Qdrant
        points = []
        for fact in facts:
            if not fact.embedding:
                logger.warning(f"Skipping fact {fact.fact_id}: no embedding")
                continue
            points.append(
                PointStruct(
                    id=fact.fact_id,
                    vector=fact.embedding,
                    payload={
                        **fact.to_dict(),
                        "user_id": user_id,
                    },
                )
            )

        if points:
            self._qdrant.upsert(collection_name=self._collection, points=points)
            logger.info(f"Stored {len(points)} facts for user={user_id}")

        return facts

    # ── Read API ─────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        user_id: str = "default",
        depth: str | None = None,
    ) -> DialecticResponse:
        """Search memories + dialectic reasoning.

        This is the main read API. It:
        1. Embeds the query
        2. Searches Qdrant for relevant facts
        3. Runs dialectic reasoning (multi-level LLM synthesis)

        Returns a DialecticResponse with answer, confidence, citations.
        """
        self._ensure_connected()

        # Embed query
        query_embedding = await self._writer._embed(query)
        if not query_embedding:
            return DialecticResponse(
                answer="Failed to embed query.",
                depth="error",
                evidence_count=0,
                confidence=0.0,
                contradictions=[],
                citations=[],
            )

        # Search Qdrant
        search_results = self._qdrant.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=self.config.recall.top_k,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            score_threshold=self.config.recall.min_score,
        )

        evidence = [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                fact_id=hit.id,
                metadata=hit.payload,
            )
            for hit in search_results
        ]

        # Dialectic reasoning
        return await self._reader.search_and_reason(query, evidence, depth)

    # ── Utilities ────────────────────────────────────────────────

    async def list_facts(self, user_id: str = "default", limit: int = 50) -> list[dict]:
        """List all stored facts for a user."""
        self._ensure_connected()
        results = self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=limit,
        )
        return [point.payload for point in results[0]]

    async def delete_fact(self, fact_id: str) -> None:
        """Delete a specific fact."""
        self._ensure_connected()
        self._qdrant.delete(collection_name=self._collection, points_selector=[fact_id])

    def _ensure_connected(self) -> None:
        if not self._qdrant:
            raise RuntimeError("Not connected. Call connect() first.")
