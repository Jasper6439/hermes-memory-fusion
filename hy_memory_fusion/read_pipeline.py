"""Read Pipeline — Honcho-style dialectic reasoning.

Responsibilities:
- Retrieve relevant facts from vector store
- Apply multi-signal ranking (semantic, recency, importance, access)
- Synthesize answers with dialectic reasoning (5 levels)
- Update access counters for retrieved facts
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from openai import AsyncOpenAI

from hy_memory_fusion.config import FusionConfig

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are a memory synthesis engine. Given retrieved memories and a query,
produce a concise, factual answer. Cite memory IDs when possible.

Query: {query}

Retrieved memories (ranked by relevance):
{memories}

Respond in this JSON format:
{{"answer": "...", "confidence": 0.0-1.0, "sources": ["fact_id1", ...], "reasoning": "brief explanation"}}"""

HONCHO_REASONING_PROMPT = """You are a dialectic reasoning engine. Analyze the following memories
and context to answer the query with depth appropriate to its complexity.

Query: {query}
Reasoning level: {reasoning_level}

Available memories:
{memories}

Instructions by reasoning level:
- minimal: Quick factual lookup. Direct answer only.
- low: Simple question with clear answer. Brief explanation.
- medium: Multi-aspect question. Synthesize across memories.
- high: Complex behavioral patterns. Deep analysis with contradictions.
- max: Thorough audit. Leave no stone unturned.

Respond with a comprehensive answer appropriate to the reasoning level."""


@dataclass
class RankedFact:
    """A fact with its computed relevance score."""

    fact_id: str
    text: str
    score: float
    semantic_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    access_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "score": round(self.score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "recency_score": round(self.recency_score, 4),
            "importance_score": round(self.importance_score, 4),
            "access_score": round(self.access_score, 4),
            "metadata": self.metadata,
        }


async def _retry(coro_factory, *, max_retries: int = 2, delay: float = 1.0, label: str = "call"):
    """Retry an async coroutine factory with exponential backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = delay * (2 ** attempt)
                logger.warning("%s attempt %d failed: %s, retrying in %.1fs", label, attempt+1, e, wait)
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


class ReadPipeline:
    """Honcho-style read pipeline with multi-signal ranking and dialectic reasoning."""

    def __init__(
        self,
        config: FusionConfig,
        llm_client: AsyncOpenAI,
        embed_client: AsyncOpenAI,
    ):
        self.config = config
        self.llm = llm_client
        self.embed_client = embed_client

    async def search(
        self,
        query: str,
        vector_store: Any,
        user_id: str = "default",
    ) -> list[RankedFact]:
        """Search and rank facts using all four signals.

        Args:
            query: Search query.
            vector_store: Must have `search(query_embedding, limit)` returning list of dicts.
            user_id: User identifier.

        Returns:
            Ranked list of RankedFact objects.
        """
        # 1. Get query embedding
        query_embedding = await self._embed(query)
        if not query_embedding:
            return []

        # 2. Vector search (semantic baseline)
        raw_results = await vector_store.search(query_embedding, limit=self.config.recall.max_results * 3)

        # 3. Multi-signal scoring
        ranked = self._rank(query_embedding, raw_results)

        # 4. Filter by min_score
        ranked = [r for r in ranked if r.score >= self.config.recall.min_score]

        # 5. Trim to max_results
        ranked = ranked[: self.config.recall.max_results]

        # 6. Update access counters (fire-and-forget)
        if ranked:
            asyncio.create_task(self._update_access(vector_store, [r.fact_id for r in ranked]))

        return ranked

    async def synthesize(
        self,
        query: str,
        facts: list[RankedFact],
        reasoning_level: str = "low",
    ) -> dict[str, Any]:
        """Synthesize an answer from ranked facts using dialectic reasoning.

        Args:
            query: Original query.
            facts: Ranked facts from search().
            reasoning_level: minimal/low/medium/high/max.

        Returns:
            Dict with answer, confidence, sources, reasoning.
        """
        if not facts:
            return {"answer": "No relevant memories found.", "confidence": 0.0, "sources": [], "reasoning": ""}

        # Format facts for prompt
        memory_text = "\n".join(
            f"- [{f.fact_id}] {f.text} (score: {f.score:.2f}, importance: {f.metadata.get('importance', '?')})"
            for f in facts
        )

        # Choose prompt based on reasoning level
        if reasoning_level in ("minimal", "low"):
            prompt = SYNTHESIS_PROMPT.format(query=query, memories=memory_text)
        else:
            prompt = HONCHO_REASONING_PROMPT.format(
                query=query,
                reasoning_level=reasoning_level,
                memories=memory_text,
            )

        async def _call():
            response = await self.llm.chat.completions.create(
                model=self.config.reader.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.reader.temperature,
                max_tokens=self.config.reader.max_tokens,
            )
            return response.choices[0].message.content or ""

        content = ""
        try:
            content = await _retry(
                _call,
                max_retries=self.config.pipeline.max_retries,
                delay=self.config.pipeline.retry_delay,
                label="synthesis",
            )

            # Parse JSON response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                content = content.rsplit("```", 1)[0]

            result = json.loads(content)
            return {
                "answer": result.get("answer", content),
                "confidence": float(result.get("confidence", 0.5)),
                "sources": result.get("sources", [f.fact_id for f in facts[:3]]),
                "reasoning": result.get("reasoning", ""),
            }

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Synthesis response parse failed: %s, using raw text", e)
            return {
                "answer": content if content else "Synthesis failed",
                "confidence": 0.3,
                "sources": [f.fact_id for f in facts[:3]],
                "reasoning": str(e),
            }

    def _rank(
        self,
        query_embedding: list[float],
        raw_results: list[dict[str, Any]],
    ) -> list[RankedFact]:
        """Multi-signal ranking: semantic + recency + importance + access.

        Uses RecallConfig weights to combine signals.
        """
        from hy_memory_fusion.write_pipeline import _cosine_similarity

        cfg = self.config.recall
        now = datetime.now(timezone.utc)
        ranked: list[RankedFact] = []

        for result in raw_results:
            # 1. Semantic similarity
            vec = result.get("embedding") or result.get("vector") or []
            sem_score = _cosine_similarity(query_embedding, vec) if vec else 0.0

            # 2. Recency score (exponential decay, half-life = 30 days)
            created = result.get("created_at", "")
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    age_days = max((now - dt).total_seconds() / 86400, 0.01)
                    rec_score = 0.5 ** (age_days / 30)  # half-life decay
                except (ValueError, TypeError):
                    rec_score = 0.0
            else:
                rec_score = 0.0

            # 3. Importance score (direct from fact metadata)
            imp_score = float(result.get("importance", 0.5))

            # 4. Access score (logarithmic normalization)
            access_count = int(result.get("access_count", 0))
            acc_score = min(1.0, (access_count / (access_count + 10)))  # saturates at ~10 accesses

            # Weighted combination
            combined = (
                cfg.semantic_weight * sem_score
                + cfg.recency_weight * rec_score
                + cfg.importance_weight * imp_score
                + cfg.access_weight * acc_score
            )

            ranked.append(
                RankedFact(
                    fact_id=result.get("fact_id", result.get("id", "")),
                    text=result.get("text", ""),
                    score=combined,
                    semantic_score=sem_score,
                    recency_score=rec_score,
                    importance_score=imp_score,
                    access_score=acc_score,
                    metadata={
                        "importance": result.get("importance"),
                        "category": result.get("category"),
                        "created_at": result.get("created_at"),
                        "access_count": access_count,
                    },
                )
            )

        # Sort by combined score descending
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    async def _update_access(self, vector_store: Any, fact_ids: list[str]) -> None:
        """Update access counters for retrieved facts."""
        try:
            if hasattr(vector_store, "update_access"):
                await vector_store.update_access(fact_ids)
        except Exception as e:
            logger.debug("Access counter update failed: %s", e)

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""

        async def _call():
            resp = await self.embed_client.embeddings.create(
                model=self.config.embedder.model,
                input=text,
            )
            return resp.data[0].embedding

        try:
            return await _retry(
                _call,
                max_retries=self.config.pipeline.max_retries,
                delay=self.config.pipeline.retry_delay,
                label="query_embed",
            )
        except Exception as e:
            logger.error("Query embedding failed: %s", e)
            return []
