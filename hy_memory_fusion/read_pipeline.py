"""Read Pipeline — Honcho-style dialectic reasoning.

Responsibilities:
- Search vector store for relevant evidence
- Multi-level LLM synthesis (dialectic reasoning)
- Return structured answers with citations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from hy_memory_fusion.config import FusionConfig

logger = logging.getLogger(__name__)

# Depth levels map to prompt complexity
DEPTH_PROMPTS = {
    "minimal": "Answer briefly in 1-2 sentences based on the evidence.",
    "low": "Answer concisely based on the evidence. Cite sources.",
    "medium": (
        "Analyze the evidence carefully. Synthesize a comprehensive answer. "
        "Note any contradictions or gaps. Cite which evidence supports each claim."
    ),
    "high": (
        "Perform deep analysis of the evidence. Consider multiple perspectives. "
        "Identify patterns, contradictions, and temporal relationships. "
        "Synthesize a nuanced answer with explicit confidence levels. "
        "Cite evidence for each claim and note where evidence is missing."
    ),
    "max": (
        "Perform exhaustive analysis. Cross-reference all evidence snippets. "
        "Identify causal chains, temporal sequences, and implicit connections. "
        "Resolve contradictions by weighing recency and source reliability. "
        "Produce a structured report with: findings, confidence levels, "
        "contradictions, gaps, and actionable conclusions. Cite all sources."
    ),
}


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    text: str
    score: float
    fact_id: str = ""
    metadata: dict[str, Any] | None = None


@dataclass
class DialecticResponse:
    """Structured response from dialectic reasoning."""

    answer: str
    depth: str
    evidence_count: int
    confidence: float  # 0.0 - 1.0
    contradictions: list[str]
    citations: list[str]  # fact_ids referenced


class ReadPipeline:
    """Honcho-style read pipeline with dialectic reasoning."""

    def __init__(self, config: FusionConfig, llm_client: OpenAI):
        self.config = config
        self.llm = llm_client

    async def search_and_reason(
        self,
        query: str,
        evidence: list[SearchResult],
        depth: str | None = None,
    ) -> DialecticResponse:
        """Search + dialectic reasoning in one call.

        Args:
            query: The user's question
            evidence: Pre-retrieved evidence snippets
            depth: Reasoning depth (minimal/low/medium/high/max), overrides config

        Returns:
            DialecticResponse with synthesized answer and metadata
        """
        depth = depth or self.config.dialectic.depth

        if not self.config.dialectic.enabled or not evidence:
            # Bypass: return top evidence directly
            return DialecticResponse(
                answer=evidence[0].text if evidence else "No relevant memories found.",
                depth="bypass",
                evidence_count=len(evidence),
                confidence=evidence[0].score if evidence else 0.0,
                contradictions=[],
                citations=[e.fact_id for e in evidence[:3]],
            )

        return await self._dialectic_reason(query, evidence, depth)

    async def _dialectic_reason(
        self,
        query: str,
        evidence: list[SearchResult],
        depth: str,
    ) -> DialecticResponse:
        """Multi-level dialectic reasoning over evidence."""
        # Format evidence
        evidence_text = "\n".join(
            f"[{i}] (score={e.score:.3f}, id={e.fact_id}) {e.text}"
            for i, e in enumerate(evidence[: self.config.dialectic.max_evidence])
        )

        depth_instruction = DEPTH_PROMPTS.get(depth, DEPTH_PROMPTS["medium"])

        prompt = f"""{self.config.dialectic.system_prompt}

Query: {query}

Evidence snippets:
{evidence_text}

Instruction: {depth_instruction}

Respond in this exact JSON format:
{{
  "answer": "your synthesized answer",
  "confidence": 0.0-1.0,
  "contradictions": ["list of any contradictions found"],
  "citations": ["fact_ids referenced in the answer"]
}}"""

        try:
            response = self.llm.chat.completions.create(
                model=self.config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            content = response.choices[0].message.content or "{}"

            # Parse JSON
            import json

            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                content = content.rsplit("```", 1)[0]
            data = json.loads(content)

            return DialecticResponse(
                answer=data.get("answer", ""),
                depth=depth,
                evidence_count=len(evidence),
                confidence=float(data.get("confidence", 0.5)),
                contradictions=data.get("contradictions", []),
                citations=data.get("citations", []),
            )

        except Exception as e:
            logger.error(f"Dialectic reasoning failed: {e}")
            return DialecticResponse(
                answer=f"Reasoning failed: {e}. Top evidence: {evidence[0].text if evidence else 'none'}",
                depth=depth,
                evidence_count=len(evidence),
                confidence=0.0,
                contradictions=[],
                citations=[],
            )
