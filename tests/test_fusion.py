"""Tests for hermes-memory-fusion.

Tests use mock OpenAI/Qdrant clients to avoid real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from hy_memory_fusion.config import FusionConfig, VectorStoreConfig, EmbedderConfig, LLMConfig
from hy_memory_fusion.write_pipeline import WritePipeline, ExtractedFact
from hy_memory_fusion.read_pipeline import ReadPipeline, SearchResult, DialecticResponse


# ── Config Tests ─────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        cfg = FusionConfig()
        assert cfg.vector_store.embedding_dims == 1024
        assert cfg.distillation.enabled is True
        assert cfg.dialectic.depth == "medium"
        assert cfg.recall.top_k == 10

    def test_custom_config(self):
        cfg = FusionConfig(
            vector_store=VectorStoreConfig(url="http://test:6333", collection_name="test_col"),
            dialectic={"depth": "high"},  # type: ignore
        )
        assert cfg.vector_store.url == "http://test:6333"
        assert cfg.vector_store.collection_name == "test_col"


# ── ExtractedFact Tests ─────────────────────────────────────


class TestExtractedFact:
    def test_auto_fields(self):
        fact = ExtractedFact(subject="Ulysses", relation="prefers", object="dark mode")
        assert fact.text == "Ulysses prefers dark mode"
        assert fact.fact_id.startswith("f_")
        assert fact.created_at  # auto-generated

    def test_deterministic_id(self):
        f1 = ExtractedFact(subject="A", relation="is", object="B")
        f2 = ExtractedFact(subject="A", relation="is", object="B")
        assert f1.fact_id == f2.fact_id

    def test_to_dict(self):
        fact = ExtractedFact(subject="X", relation="has", object="Y", importance=0.8)
        d = fact.to_dict()
        assert d["subject"] == "X"
        assert d["importance"] == 0.8
        assert "fact_id" in d
        assert "created_at" in d


# ── Write Pipeline Tests ────────────────────────────────────


class TestWritePipeline:
    def _make_pipeline(self, llm_response: str = "[]"):
        cfg = FusionConfig()
        mock_llm = MagicMock()
        mock_embed = MagicMock()

        # Mock LLM response
        choice = MagicMock()
        choice.message.content = llm_response
        mock_llm.chat.completions.create.return_value = MagicMock(choices=[choice])

        # Mock embedding response
        emb_data = MagicMock()
        emb_data.embedding = [0.1] * 1024
        mock_embed.embeddings.create.return_value = MagicMock(data=[emb_data])

        return WritePipeline(cfg, mock_llm, mock_embed)

    @pytest.mark.asyncio
    async def test_extract_svo_empty(self):
        pipeline = self._make_pipeline("[]")
        facts = await pipeline._extract_svo("hello world")
        # Empty SVO extraction returns empty list (no facts to extract from noise)
        assert isinstance(facts, list)

    @pytest.mark.asyncio
    async def test_extract_svo_valid(self):
        svo_json = json.dumps([
            {"subject": "Ulysses", "relation": "manages", "object": "OCI servers", "importance": 0.9, "category": "fact"},
            {"subject": "Server", "relation": "runs on", "object": "ARM64 Ubuntu", "importance": 0.7, "category": "fact"},
        ])
        pipeline = self._make_pipeline(svo_json)
        facts = await pipeline._extract_svo("Ulysses manages OCI servers running on ARM64 Ubuntu")
        assert len(facts) == 2
        assert facts[0].subject == "Ulysses"
        assert facts[0].importance == 0.9

    @pytest.mark.asyncio
    async def test_ingest_disabled(self):
        cfg = FusionConfig()
        cfg.distillation.enabled = False
        pipeline = self._make_pipeline()
        pipeline.config = cfg
        facts = await pipeline.ingest("test text")
        assert len(facts) == 1
        assert facts[0].subject == "test text"

    @pytest.mark.asyncio
    async def test_embed(self):
        pipeline = self._make_pipeline()
        vec = await pipeline._embed("test")
        assert len(vec) == 1024
        assert all(v == 0.1 for v in vec)


# ── Read Pipeline Tests ─────────────────────────────────────


class TestReadPipeline:
    def _make_pipeline(self, response_json: str | None = None):
        cfg = FusionConfig()
        mock_llm = MagicMock()

        if response_json is None:
            response_json = json.dumps({
                "answer": "Ulysses manages OCI servers in Tokyo.",
                "confidence": 0.9,
                "contradictions": [],
                "citations": ["f_abc123"],
            })

        choice = MagicMock()
        choice.message.content = response_json
        mock_llm.chat.completions.create.return_value = MagicMock(choices=[choice])

        return ReadPipeline(cfg, mock_llm)

    @pytest.mark.asyncio
    async def test_dialectic_basic(self):
        pipeline = self._make_pipeline()
        evidence = [
            SearchResult(text="Ulysses manages OCI servers", score=0.95, fact_id="f_abc123"),
            SearchResult(text="Server runs ARM64 Ubuntu", score=0.85, fact_id="f_def456"),
        ]
        result = await pipeline.search_and_reason("What does Ulysses manage?", evidence)
        assert isinstance(result, DialecticResponse)
        assert "OCI" in result.answer
        assert result.confidence == 0.9
        assert result.evidence_count == 2

    @pytest.mark.asyncio
    async def test_dialectic_bypass(self):
        cfg = FusionConfig()
        cfg.dialectic.enabled = False
        pipeline = self._make_pipeline()
        pipeline.config = cfg
        evidence = [SearchResult(text="test fact", score=0.8, fact_id="f_001")]
        result = await pipeline.search_and_reason("query", evidence)
        assert result.depth == "bypass"
        assert result.answer == "test fact"

    @pytest.mark.asyncio
    async def test_dialectic_no_evidence(self):
        pipeline = self._make_pipeline()
        result = await pipeline.search_and_reason("query", [])
        assert result.depth == "bypass"
        assert "No relevant" in result.answer

    @pytest.mark.asyncio
    async def test_depth_levels(self):
        pipeline = self._make_pipeline()
        evidence = [SearchResult(text="fact", score=0.9, fact_id="f_001")]
        for depth in ["minimal", "low", "medium", "high", "max"]:
            result = await pipeline.search_and_reason("query", evidence, depth=depth)
            assert result.depth == depth


# ── Fact Lifecycle Tests ────────────────────────────────────


class TestFactLifecycle:
    def test_fact_idempotent(self):
        """Same text should produce same fact ID."""
        f1 = ExtractedFact(subject="A", relation="is", object="B")
        f2 = ExtractedFact(subject="A", relation="is", object="B")
        assert f1.fact_id == f2.fact_id

    def test_fact_categories(self):
        for cat in ["preference", "fact", "event", "identity", "intent"]:
            f = ExtractedFact(subject="X", relation="is", object="Y", category=cat)
            assert f.category == cat

    def test_fact_importance_range(self):
        for imp in [0.0, 0.3, 0.5, 0.8, 1.0]:
            f = ExtractedFact(subject="X", relation="is", object="Y", importance=imp)
            assert f.importance == imp
