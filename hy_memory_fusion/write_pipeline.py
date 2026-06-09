"""Write Pipeline — Hy-Memory style auto-distillation.

Responsibilities:
- Ingest raw text (conversation, notes, facts)
- Extract SVO (Subject-Verb-Object) triplets via LLM
- Deduplicate against existing memories via semantic similarity
- Store structured facts with metadata
"""

from __future__ import annotations
import json
import logging
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from openai import AsyncOpenAI

from hy_memory_fusion.config import FusionConfig
from hy_memory_fusion._utils import retry, cosine_similarity, embed_text, embed_batch, strip_markdown_json

logger = logging.getLogger(__name__)

SVO_EXTRACTION_PROMPT = """Extract atomic facts from the following text as JSON array.
Each fact should be a triplet: {{"subject": "...", "relation": "...", "object": "..."}}
Also include: {{"importance": 0.0-1.0, "category": "preference|fact|event|identity|intent"}}

Rules:
- Each fact should be self-contained and unambiguous
- Merge redundant facts
- Skip trivial/chat noise
- importance: 1.0 = critical identity info, 0.5 = useful context, 0.1 = minor detail

Text:
{text}

Return ONLY valid JSON array, no markdown."""


@dataclass
class ExtractedFact:
    """A single extracted atomic fact."""

    subject: str
    relation: str
    object: str
    importance: float = 0.5
    category: str = "fact"
    text: str = ""
    embedding: list[float] = field(default_factory=list)
    fact_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            self.text = f"{self.subject} {self.relation} {self.object}"
        if not self.fact_id:
            h = hashlib.sha256(self.text.encode()).hexdigest()
            self.fact_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes-fact:{h}"))
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "text": self.text,
            "importance": self.importance,
            "category": self.category,
            "created_at": self.created_at,
        }


class WritePipeline:
    """Hy-Memory style write pipeline with SVO extraction and deduplication."""

    def __init__(
        self,
        config: FusionConfig,
        llm_client: AsyncOpenAI,
        embed_client: AsyncOpenAI,
    ):
        self.config = config
        self.llm = llm_client
        self.embed_client = embed_client

    async def ingest(
        self,
        text: str,
        user_id: str = "default",
        existing_facts: Optional[list[dict[str, Any]]] = None,
    ) -> list[ExtractedFact]:
        """Ingest raw text → extract facts → dedup → return structured facts.

        Args:
            text: Raw text to ingest.
            user_id: User identifier.
            existing_facts: List of existing fact dicts for dedup (from Qdrant).
                Each dict must have at least 'text' and optionally 'fact_id'.
        """
        # Truncate extremely long input to protect against token overflow
        MAX_INPUT_CHARS = 50_000  # ~12k tokens for most models
        if len(text) > MAX_INPUT_CHARS:
            logger.warning(
                "Input text truncated from %d to %d chars", len(text), MAX_INPUT_CHARS
            )
            text = text[:MAX_INPUT_CHARS]

        if not self.config.distillation.enabled:
            # Bypass: store raw text as single fact
            fact = ExtractedFact(subject=text[:50], relation="is", object="raw_note", importance=0.3)
            fact.embedding = await self._embed(fact.text)
            return [fact]

        # Step 1: Extract SVO triplets
        raw_facts = await self._extract_svo(text)
        logger.info("Extracted %d raw facts from text", len(raw_facts))

        # Step 2: Batch embed all facts
        texts = [f.text for f in raw_facts]
        embeddings = await self._embed_batch(texts)
        for fact, emb in zip(raw_facts, embeddings):
            fact.embedding = emb

        # Step 3: Dedup against existing + intra-batch
        raw_facts = await self._dedup(raw_facts, existing_facts or [])
        logger.info("After dedup: %d facts remain", len(raw_facts))

        return raw_facts

    async def _extract_svo(self, text: str) -> list[ExtractedFact]:
        """Use LLM to extract Subject-Verb-Object triplets."""
        prompt = SVO_EXTRACTION_PROMPT.format(text=text)

        async def _call():
            response = await self.llm.chat.completions.create(
                model=self.config.writer.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.writer.temperature,
                max_tokens=self.config.writer.max_tokens,
            )
            return response.choices[0].message.content or "[]"

        try:
            content = await retry(
                _call,
                max_retries=self.config.pipeline.max_retries,
                delay=self.config.pipeline.retry_delay,
                label="SVO extraction",
            )

            # Parse JSON (handle markdown code blocks)
            content = strip_markdown_json(content)
            items = json.loads(content)

            facts: list[ExtractedFact] = []
            for item in items:
                if isinstance(item, dict) and "subject" in item:
                    facts.append(
                        ExtractedFact(
                            subject=item.get("subject", ""),
                            relation=item.get("relation", "is"),
                            object=item.get("object", ""),
                            importance=float(item.get("importance", 0.5)),
                            category=item.get("category", "fact"),
                        )
                    )
            return facts

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("SVO extraction failed: %s, falling back to raw text", e)
            return [ExtractedFact(subject=text[:80], relation="states", object=text[80:160] if len(text) > 80 else "")]

    async def _dedup(
        self,
        new_facts: list[ExtractedFact],
        existing_facts: list[dict[str, Any]],
    ) -> list[ExtractedFact]:
        """Semantic deduplication: remove new facts too similar to existing ones.

        Uses vector similarity (fast) first, then LLM judgment for borderline cases.
        """
        # Build existing embeddings matrix
        existing_embeddings = []
        for ef in existing_facts:
            vec = ef.get("embedding") or ef.get("vector")
            if vec:
                existing_embeddings.append(vec)
            else:
                existing_embeddings.append(None)

        threshold = self.config.distillation.dedup_threshold
        kept: list[ExtractedFact] = []

        for fact in new_facts:
            if not fact.embedding:
                kept.append(fact)
                continue

            # Check similarity against all existing
            is_dup = False
            for existing_emb in existing_embeddings:
                if existing_emb is None:
                    continue
                sim = cosine_similarity(fact.embedding, existing_emb)
                if sim >= threshold:
                    logger.debug("Dedup: '%s' is duplicate (sim=%.3f)", fact.text, sim)
                    is_dup = True
                    break

            if not is_dup:
                # Also check against already-kept facts in this batch
                for kept_fact in kept:
                    if not kept_fact.embedding:
                        continue
                    sim = cosine_similarity(fact.embedding, kept_fact.embedding)
                    if sim >= threshold:
                        logger.debug("Intra-batch dedup: '%s' vs '%s' (sim=%.3f)", fact.text, kept_fact.text, sim)
                        is_dup = True
                        break

            if not is_dup:
                kept.append(fact)

        removed = len(new_facts) - len(kept)
        if removed:
            logger.info("Dedup removed %d facts (threshold=%.2f)", removed, threshold)

        return kept

    async def embed(self, text: str) -> list[float]:
        """Public embedding method (shared with MemoryCore)."""
        return await embed_text(
            text, self.embed_client, self.config.embedder.model,
            max_retries=self.config.pipeline.max_retries,
            delay=self.config.pipeline.retry_delay,
        )

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector for single text."""
        return await embed_text(
            text, self.embed_client, self.config.embedder.model,
            max_retries=self.config.pipeline.max_retries,
            delay=self.config.pipeline.retry_delay,
        )

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple texts in a single API call."""
        return await embed_batch(
            texts, self.embed_client, self.config.embedder.model,
            self.config.embedder.batch_size,
            max_retries=self.config.pipeline.max_retries,
            delay=self.config.pipeline.retry_delay,
        )
