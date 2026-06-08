"""Write Pipeline — Hy-Memory style auto-distillation.

Responsibilities:
- Ingest raw text (conversation, notes, facts)
- Extract SVO (Subject-Verb-Object) triplets via LLM
- Deduplicate against existing memories
- Store structured facts with metadata
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from openai import OpenAI

from hy_memory_fusion.config import FusionConfig

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

DEDUP_PROMPT = """Given a new fact and a list of existing facts, determine which existing facts
are semantically duplicates of the new one.

New fact: {new_fact}

Existing facts:
{existing_facts}

Return a JSON array of indices (0-based) of duplicate facts. Return [] if no duplicates."""


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

    def __post_init__(self):
        if not self.text:
            self.text = f"{self.subject} {self.relation} {self.object}"
        if not self.fact_id:
            h = hashlib.sha256(self.text.encode()).hexdigest()[:12]
            self.fact_id = f"f_{h}"
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

    def __init__(self, config: FusionConfig, llm_client: OpenAI, embed_client: OpenAI):
        self.config = config
        self.llm = llm_client
        self.embed = embed_client

    async def ingest(self, text: str, user_id: str = "default") -> list[ExtractedFact]:
        """Ingest raw text → extract facts → dedup → return structured facts."""
        if not self.config.distillation.enabled:
            # Bypass: store raw text as single fact
            fact = ExtractedFact(subject=text[:50], relation="is", object="raw_note", importance=0.3)
            fact.embedding = await self._embed(fact.text)
            return [fact]

        # Step 1: Extract SVO triplets
        raw_facts = await self._extract_svo(text)
        logger.info(f"Extracted {len(raw_facts)} raw facts from text")

        # Step 2: Embed each fact
        for fact in raw_facts:
            fact.embedding = await self._embed(fact.text)

        # Step 3: Dedup against existing (if vector store available)
        # For now, return extracted facts — dedup handled by caller with vector store
        return raw_facts

    async def _extract_svo(self, text: str) -> list[ExtractedFact]:
        """Use LLM to extract Subject-Verb-Object triplets."""
        prompt = SVO_EXTRACTION_PROMPT.format(text=text)

        try:
            response = self.llm.chat.completions.create(
                model=self.config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            content = response.choices[0].message.content or "[]"

            # Parse JSON (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                content = content.rsplit("```", 1)[0]
            items = json.loads(content)

            facts = []
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
            logger.warning(f"SVO extraction failed: {e}, falling back to raw text")
            return [ExtractedFact(subject=text[:80], relation="states", object=text[80:160] if len(text) > 80 else "")]

    async def _embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        try:
            resp = self.embed.embeddings.create(
                model=self.config.embedder.model,
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []
