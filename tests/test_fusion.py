"""Tests for hermes-memory-fusion: Write Pipeline, Read Pipeline, Memory Core.

All tests use mock clients — no live LLM/Qdrant needed.
Run: pytest tests/ -v
"""

from __future__ import annotations

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

from hy_memory_fusion.config import (
    FusionConfig,
    LLMConfig,
    EmbedderConfig,
    QdrantConfig,
    DistillationConfig,
    RecallConfig,
    PipelineConfig,
)
from hy_memory_fusion._utils import cosine_similarity
from hy_memory_fusion.write_pipeline import WritePipeline, ExtractedFact
from hy_memory_fusion.read_pipeline import ReadPipeline, RankedFact
from hy_memory_fusion.memory_core import MemoryCore


# ── Helpers ──────────────────────────────────────────────────────────────


def make_config() -> FusionConfig:
    return FusionConfig(
        llm=LLMConfig(base_url="http://test", api_key="test", model="test-model"),
        embedder=EmbedderConfig(base_url="http://test", api_key="test", model="test-embed", batch_size=2),
        qdrant=QdrantConfig(url="http://test:6333", collection="test_fusion", vector_dim=4),
        distillation=DistillationConfig(enabled=True, importance_threshold=0.3, dedup_threshold=0.92),
        recall=RecallConfig(
            max_results=5,
            min_score=0.1,
            semantic_weight=0.6,
            recency_weight=0.15,
            importance_weight=0.2,
            access_weight=0.05,
        ),
        pipeline=PipelineConfig(timeout=5.0, max_retries=1, retry_delay=0.01),
        reader=LLMConfig(base_url="http://test", api_key="test", model="test-reader"),
        writer=LLMConfig(base_url="http://test", api_key="test", model="test-writer"),
    )


def mock_embed_client(responses: list[list[float]] | None = None):
    """Create a mock AsyncOpenAI embed client."""
    client = AsyncMock()
    call_count = [0]

    def side_effect(**kwargs):
        mock_resp = MagicMock()
        inp = kwargs.get("input", "")
        if isinstance(inp, list):
            # Batch embed
            mock_resp.data = [MagicMock(embedding=responses[i % len(responses)] if responses else [0.1, 0.2, 0.3, 0.4]) for i in range(len(inp))]
        else:
            emb = responses[call_count[0] % len(responses)] if responses else [0.1, 0.2, 0.3, 0.4]
            mock_resp.data = [MagicMock(embedding=emb)]
            call_count[0] += 1
        return mock_resp

    client.embeddings.create = AsyncMock(side_effect=side_effect)
    return client


def mock_llm_client(response_content: str):
    """Create a mock AsyncOpenAI LLM client."""
    client = AsyncMock()
    mock_message = MagicMock()
    mock_message.content = response_content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    client.chat.completions.create = AsyncMock(return_value=mock_resp)
    return client


# ── Config Tests ─────────────────────────────────────────────────────────


class TestFusionConfig:
    def test_default_values(self):
        cfg = FusionConfig()
        assert cfg.llm.model == "nousresearch/hermes-3-llama-3.1-405b"
        assert cfg.qdrant.vector_dim == 1024
        assert cfg.distillation.dedup_threshold == 0.92
        assert cfg.recall.semantic_weight == 0.6
        assert cfg.recall.recency_weight == 0.15
        assert cfg.recall.importance_weight == 0.2
        assert cfg.recall.access_weight == 0.05

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_LLM_MODEL", "env-model")
        monkeypatch.setenv("FUSION_QDRANT_URL", "http://env:6333")
        monkeypatch.setenv("FUSION_DISTILLATION_ENABLED", "false")
        monkeypatch.setenv("FUSION_DEDUP_THRESHOLD", "0.85")
        monkeypatch.setenv("FUSION_READER_MODEL", "env-reader")
        monkeypatch.setenv("FUSION_WRITER_MODEL", "env-writer")
        monkeypatch.setenv("FUSION_EMBEDDER_MODEL", "env-embed")
        monkeypatch.setenv("FUSION_PIPELINE_TIMEOUT", "60")
        monkeypatch.setenv("FUSION_PIPELINE_MAX_RETRIES", "3")
        monkeypatch.setenv("FUSION_RECALL_SEMANTIC_WEIGHT", "0.7")

        cfg = FusionConfig.from_env()
        assert cfg.llm.model == "env-model"
        assert cfg.qdrant.url == "http://env:6333"
        assert cfg.distillation.enabled is False
        assert cfg.distillation.dedup_threshold == 0.85
        assert cfg.reader.model == "env-reader"
        assert cfg.writer.model == "env-writer"
        assert cfg.embedder.model == "env-embed"
        assert cfg.pipeline.timeout == 60.0
        assert cfg.pipeline.max_retries == 3
        assert cfg.recall.semantic_weight == 0.7


# ── Write Pipeline Tests ─────────────────────────────────────────────────


class TestExtractedFact:
    def test_auto_fields(self):
        fact = ExtractedFact(subject="Alice", relation="likes", object="coffee")
        assert fact.text == "Alice likes coffee"
        assert fact.fact_id.startswith("f_")
        assert fact.created_at  # auto-set
        assert fact.importance == 0.5

    def test_to_dict(self):
        fact = ExtractedFact(subject="Bob", relation="is", object="admin", importance=0.9)
        d = fact.to_dict()
        assert d["subject"] == "Bob"
        assert d["importance"] == 0.9
        assert "fact_id" in d
        assert "created_at" in d


class TestWritePipeline:
    @pytest.mark.asyncio
    async def test_ingest_bypass_disabled(self):
        config = make_config()
        config.distillation.enabled = False
        writer = WritePipeline(config, mock_llm_client("[]"), mock_embed_client())

        facts = await writer.ingest("Test text")
        assert len(facts) == 1
        assert facts[0].subject == "Test text"
        assert facts[0].relation == "is"
        assert facts[0].embedding  # should have embedding

    @pytest.mark.asyncio
    async def test_ingest_extracts_svo(self):
        svo_response = json.dumps([
            {"subject": "Alice", "relation": "likes", "object": "coffee", "importance": 0.7, "category": "preference"},
            {"subject": "Bob", "relation": "is", "object": "admin", "importance": 0.9, "category": "identity"},
        ])
        config = make_config()
        writer = WritePipeline(
            config,
            mock_llm_client(svo_response),
            mock_embed_client([[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9]]),
        )

        facts = await writer.ingest("Alice likes coffee. Bob is admin.")
        assert len(facts) == 2
        assert facts[0].subject == "Alice"
        assert facts[0].importance == 0.7
        assert facts[1].subject == "Bob"
        assert facts[0].embedding == [0.9, 0.1, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_ingest_dedup_removes_similar(self):
        svo_response = json.dumps([
            {"subject": "Alice", "relation": "likes", "object": "coffee", "importance": 0.7},
        ])
        config = make_config()
        # New fact and existing fact have same embedding → high similarity
        writer = WritePipeline(
            config,
            mock_llm_client(svo_response),
            mock_embed_client([[0.9, 0.1, 0.0, 0.0]]),
        )

        existing = [{"text": "Alice likes coffee", "embedding": [0.9, 0.1, 0.0, 0.0], "fact_id": "existing_1"}]
        facts = await writer.ingest("Alice likes coffee", existing_facts=existing)
        assert len(facts) == 0  # deduped

    @pytest.mark.asyncio
    async def test_ingest_dedup_keeps_different(self):
        svo_response = json.dumps([
            {"subject": "Bob", "relation": "runs", "object": "marathon", "importance": 0.6},
        ])
        config = make_config()
        writer = WritePipeline(
            config,
            mock_llm_client(svo_response),
            mock_embed_client([[0.0, 0.0, 0.9, 0.1]]),
        )

        existing = [{"text": "Alice likes coffee", "embedding": [0.9, 0.1, 0.0, 0.0], "fact_id": "existing_1"}]
        facts = await writer.ingest("Bob runs marathon", existing_facts=existing)
        assert len(facts) == 1  # not deduped (different embedding)

    @pytest.mark.asyncio
    async def test_ingest_filters_by_importance(self):
        svo_response = json.dumps([
            {"subject": "noise", "relation": "is", "object": "trivial", "importance": 0.1},
        ])
        config = make_config()
        config.distillation.importance_threshold = 0.3
        writer = WritePipeline(
            config,
            mock_llm_client(svo_response),
            mock_embed_client(),
        )

        facts = await writer.ingest("some noise")
        # Facts returned by pipeline, but memory_core would filter them
        assert len(facts) == 1  # pipeline returns all; memory_core filters
        assert facts[0].importance == 0.1

    @pytest.mark.asyncio
    async def test_ingest_handles_invalid_json(self):
        config = make_config()
        writer = WritePipeline(config, mock_llm_client("not json"), mock_embed_client())

        facts = await writer.ingest("test text")
        assert len(facts) == 1  # fallback to raw text
        assert facts[0].relation == "states"

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        config = make_config()
        config.embedder.batch_size = 2
        writer = WritePipeline(
            config,
            mock_llm_client("[]"),
            mock_embed_client([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.8, 0.7, 0.6]]),
        )

        embeddings = await writer._embed_batch(["a", "b", "c"])
        assert len(embeddings) == 3

    @pytest.mark.asyncio
    async def test_embed_public_method(self):
        config = make_config()
        writer = WritePipeline(config, mock_llm_client("[]"), mock_embed_client([[0.1, 0.2, 0.3, 0.4]]))

        result = await writer.embed("test text")
        assert result == [0.1, 0.2, 0.3, 0.4]


class TestCosineSimilarity:
    def test_identical(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_different_lengths(self):
        assert cosine_similarity([1, 0], [1, 0, 0]) == 0.0


# ── Read Pipeline Tests ──────────────────────────────────────────────────


class TestRankedFact:
    def test_to_dict(self):
        fact = RankedFact(
            fact_id="f1", text="test", score=0.85,
            semantic_score=0.9, recency_score=0.8,
            importance_score=0.7, access_score=0.6,
        )
        d = fact.to_dict()
        assert d["score"] == 0.85
        assert d["semantic_score"] == 0.9


class TestReadPipeline:
    def _make_pipeline(self):
        config = make_config()
        return ReadPipeline(config, mock_llm_client("{}"), mock_embed_client())

    def test_rank_applies_all_weights(self):
        pipeline = self._make_pipeline()
        query_embedding = [1.0, 0.0, 0.0, 0.0]

        now = datetime.now(timezone.utc)
        raw = [
            {
                "fact_id": "recent_important",
                "text": "recent and important",
                "embedding": [1.0, 0.0, 0.0, 0.0],  # high semantic
                "importance": 0.9,
                "created_at": now.isoformat(),  # very recent
                "access_count": 10,
            },
            {
                "fact_id": "old_trivial",
                "text": "old and trivial",
                "embedding": [0.0, 1.0, 0.0, 0.0],  # low semantic
                "importance": 0.1,
                "created_at": (now - timedelta(days=365)).isoformat(),  # old
                "access_count": 0,
            },
        ]

        ranked = pipeline.rank(query_embedding, raw)
        assert len(ranked) == 2
        assert ranked[0].fact_id == "recent_important"
        assert ranked[0].score > ranked[1].score

        # Verify all four signals are computed
        assert ranked[0].semantic_score > 0.9  # exact match
        assert ranked[0].recency_score > 0.9   # just created
        assert ranked[0].importance_score == 0.9
        assert ranked[0].access_score > 0.4     # 10 accesses

        assert ranked[1].semantic_score < 0.1   # orthogonal
        assert ranked[1].recency_score < 0.1    # 365 days old
        assert ranked[1].importance_score == 0.1

    def test_rank_respects_weights(self):
        pipeline = self._make_pipeline()
        # Override weights: only importance matters
        pipeline.config.recall.semantic_weight = 0.0
        pipeline.config.recall.recency_weight = 0.0
        pipeline.config.recall.importance_weight = 1.0
        pipeline.config.recall.access_weight = 0.0

        query_embedding = [1.0, 0.0, 0.0, 0.0]
        raw = [
            {"fact_id": "low_imp", "text": "low", "embedding": [1.0, 0.0, 0.0, 0.0], "importance": 0.1, "created_at": "", "access_count": 0},
            {"fact_id": "high_imp", "text": "high", "embedding": [0.0, 1.0, 0.0, 0.0], "importance": 0.9, "created_at": "", "access_count": 0},
        ]

        ranked = pipeline.rank(query_embedding, raw)
        # Even though "low_imp" has better semantic match, importance_weight=1.0 dominates
        assert ranked[0].fact_id == "high_imp"

    def test_rank_empty(self):
        pipeline = self._make_pipeline()
        ranked = pipeline.rank([1, 0, 0, 0], [])
        assert ranked == []

    @pytest.mark.asyncio
    async def test_synthesize_returns_answer(self):
        synthesis_response = json.dumps({
            "answer": "Alice likes coffee",
            "confidence": 0.85,
            "sources": ["f1"],
            "reasoning": "Direct fact match",
        })
        pipeline = ReadPipeline(
            make_config(),
            mock_llm_client(synthesis_response),
            mock_embed_client(),
        )

        facts = [RankedFact(fact_id="f1", text="Alice likes coffee", score=0.9)]
        result = await pipeline.synthesize("What does Alice like?", facts, "low")
        assert result["answer"] == "Alice likes coffee"
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_synthesize_empty_facts(self):
        pipeline = self._make_pipeline()
        result = await pipeline.synthesize("test", [], "low")
        assert "No relevant" in result["answer"]


# ── Memory Core Tests ────────────────────────────────────────────────────


class TestMemoryCore:
    def _make_core(self, llm_response="[]", embed_responses=None):
        config = make_config()
        embed_resp = embed_responses or [[0.1, 0.2, 0.3, 0.4]]

        # Mock Qdrant client
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
        mock_qdrant.create_collection = AsyncMock()
        mock_qdrant.search = AsyncMock(return_value=[])
        mock_qdrant.scroll = AsyncMock(return_value=([], None))
        mock_qdrant.upsert = AsyncMock()
        mock_qdrant.retrieve = AsyncMock(return_value=[])
        mock_qdrant.set_payload = AsyncMock()

        core = MemoryCore(
            config=config,
            qdrant_client=mock_qdrant,
            llm_client=mock_llm_client(llm_response),
            embed_client=mock_embed_client(embed_resp),
            reader_client=mock_llm_client("{}"),
            writer_client=mock_llm_client(llm_response),
        )
        return core

    @pytest.mark.asyncio
    async def test_initialize_creates_collection(self):
        core = self._make_core()
        await core.initialize()
        assert core._initialized
        core._qdrant.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_stores_facts(self):
        svo_response = json.dumps([
            {"subject": "Alice", "relation": "likes", "object": "coffee", "importance": 0.7},
        ])
        core = self._make_core(llm_response=svo_response)
        core.config.distillation.importance_threshold = 0.0

        result = await core.remember("Alice likes coffee")
        assert len(result) == 1
        assert result[0]["subject"] == "Alice"
        core._qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_fetches_existing_for_dedup(self):
        svo_response = json.dumps([
            {"subject": "test", "relation": "is", "object": "fact", "importance": 0.5},
        ])
        core = self._make_core(llm_response=svo_response)
        core.config.distillation.importance_threshold = 0.0

        await core.remember("test fact")
        # Should have called scroll to fetch existing facts
        core._qdrant.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_calls_search_and_synthesize(self):
        core = self._make_core()
        core._reader.search = AsyncMock(return_value=[
            RankedFact(fact_id="f1", text="test fact", score=0.9),
        ])
        core._reader.synthesize = AsyncMock(return_value={
            "answer": "test answer",
            "confidence": 0.8,
            "sources": ["f1"],
            "reasoning": "test",
        })

        result = await core.remember("test")  # init
        core._reader.search.reset_mock()
        core._reader.synthesize.reset_mock()

        result = await core.recall("What is test?")
        assert "answer" in result
        assert "facts" in result
        core._reader.search.assert_called_once()
        core._reader.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_semantic(self):
        core = self._make_core()
        mock_point = MagicMock()
        mock_point.id = "f1"
        mock_point.score = 0.95
        mock_point.payload = {"text": "test", "importance": 0.8}
        core._qdrant.search = AsyncMock(return_value=[mock_point])

        results = await core.hybrid_search("test", mode="semantic")
        assert len(results) == 1
        assert results[0]["fact_id"] == "f1"
        assert results[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_embed_uses_shared_client(self):
        core = self._make_core()
        result = await core.embed("test text")
        assert result == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_get_facts_by_ids(self):
        core = self._make_core()
        mock_point = MagicMock()
        mock_point.id = "f1"
        mock_point.payload = {"text": "test"}
        core._qdrant.retrieve = AsyncMock(return_value=[mock_point])

        results = await core.get_facts_by_ids(["f1"])
        assert len(results) == 1
        assert results[0]["fact_id"] == "f1"

    @pytest.mark.asyncio
    async def test_update_access_increments(self):
        core = self._make_core()
        mock_point = MagicMock()
        mock_point.payload = {"access_count": 3}
        core._qdrant.retrieve = AsyncMock(return_value=[mock_point])

        await core.update_access(["f1"])
        core._qdrant.set_payload.assert_called_once()
        call_args = core._qdrant.set_payload.call_args
        assert call_args.kwargs["payload"]["access_count"] == 4


# ── Retry Utility Tests ─────────────────────────────────────────────────


class TestRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        from hy_memory_fusion._utils import retry

        call_count = 0

        async def success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry(success, max_retries=3, delay=0.01)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        from hy_memory_fusion._utils import retry

        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "done"

        result = await retry(fail_then_succeed, max_retries=3, delay=0.01)
        assert result == "done"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhausted(self):
        from hy_memory_fusion._utils import retry

        async def always_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await retry(always_fail, max_retries=2, delay=0.01)


# ── Intra-batch Dedup Tests ─────────────────────────────────────────────


class TestIntraBatchDedup:
    @pytest.mark.asyncio
    async def test_removes_duplicates_within_batch(self):
        """Two facts with identical embeddings in same batch — second should be deduped."""
        svo_response = json.dumps([
            {"subject": "Alice", "relation": "likes", "object": "coffee", "importance": 0.7},
            {"subject": "Alice", "relation": "enjoys", "object": "coffee", "importance": 0.6},
        ])
        config = make_config()
        writer = WritePipeline(
            config,
            mock_llm_client(svo_response),
            mock_embed_client([[0.9, 0.1, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]),  # same embedding
        )

        facts = await writer.ingest("Alice likes coffee. Alice enjoys coffee.")
        assert len(facts) == 1  # intra-batch dedup removes second

    @pytest.mark.asyncio
    async def test_keeps_different_facts_in_batch(self):
        """Two facts with different embeddings — both should be kept."""
        svo_response = json.dumps([
            {"subject": "Alice", "relation": "likes", "object": "coffee", "importance": 0.7},
            {"subject": "Bob", "relation": "runs", "object": "marathon", "importance": 0.6},
        ])
        config = make_config()
        writer = WritePipeline(
            config,
            mock_llm_client(svo_response),
            mock_embed_client([[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1]]),
        )

        facts = await writer.ingest("Alice likes coffee. Bob runs marathon.")
        assert len(facts) == 2


# ── Hybrid Search Mode Tests ────────────────────────────────────────────


class TestHybridSearchModes:
    @pytest.mark.asyncio
    async def test_hybrid_mode_returns_ranked(self):
        core = TestMemoryCore()._make_core()
        now = datetime.now(timezone.utc).isoformat()
        mock_point = MagicMock()
        mock_point.id = "f1"
        mock_point.score = 0.95
        mock_point.payload = {"text": "test fact", "importance": 0.8, "created_at": now, "access_count": 5}
        core._qdrant.search = AsyncMock(return_value=[mock_point])

        results = await core.hybrid_search("test", mode="hybrid")
        assert len(results) == 1
        assert "semantic_score" in results[0]
        assert "recency_score" in results[0]

    @pytest.mark.asyncio
    async def test_semantic_mode_returns_raw(self):
        core = TestMemoryCore()._make_core()
        mock_point = MagicMock()
        mock_point.id = "f1"
        mock_point.score = 0.95
        mock_point.payload = {"text": "test fact", "importance": 0.8}
        core._qdrant.search = AsyncMock(return_value=[mock_point])

        results = await core.hybrid_search("test", mode="semantic")
        assert len(results) == 1
        assert results[0]["score"] == 0.95
        # semantic mode should NOT have multi-signal fields
        assert "semantic_score" not in results[0]


# ── Config Edge Case Tests ──────────────────────────────────────────────


class TestConfigEdgeCases:
    def test_partial_env_overrides(self):
        """Only some env vars set — others keep defaults."""
        import os
        # Clear any FUSION_ vars
        for key in list(os.environ.keys()):
            if key.startswith("FUSION_"):
                del os.environ[key]

        os.environ["FUSION_LLM_MODEL"] = "custom-model"
        os.environ["FUSION_QDRANT_VECTOR_DIM"] = "512"

        cfg = FusionConfig.from_env()
        assert cfg.llm.model == "custom-model"
        assert cfg.qdrant.vector_dim == 512
        # Defaults preserved
        assert cfg.llm.temperature == 0.1
        assert cfg.distillation.dedup_threshold == 0.92
        assert cfg.recall.semantic_weight == 0.6

        # Cleanup
        del os.environ["FUSION_LLM_MODEL"]
        del os.environ["FUSION_QDRANT_VECTOR_DIM"]

    def test_all_recall_weights_from_env(self):
        """All four recall weights configurable via env."""
        import os
        for key in list(os.environ.keys()):
            if key.startswith("FUSION_"):
                del os.environ[key]

        os.environ["FUSION_RECALL_SEMANTIC_WEIGHT"] = "0.5"
        os.environ["FUSION_RECALL_RECENCY_WEIGHT"] = "0.2"
        os.environ["FUSION_RECALL_IMPORTANCE_WEIGHT"] = "0.25"
        os.environ["FUSION_RECALL_ACCESS_WEIGHT"] = "0.05"

        cfg = FusionConfig.from_env()
        assert cfg.recall.semantic_weight == 0.5
        assert cfg.recall.recency_weight == 0.2
        assert cfg.recall.importance_weight == 0.25
        assert cfg.recall.access_weight == 0.05

        # Cleanup
        for key in ["FUSION_RECALL_SEMANTIC_WEIGHT", "FUSION_RECALL_RECENCY_WEIGHT",
                     "FUSION_RECALL_IMPORTANCE_WEIGHT", "FUSION_RECALL_ACCESS_WEIGHT"]:
            del os.environ[key]


# ── Weight Validation Tests ─────────────────────────────────────────────


class TestRecallConfigValidation:
    def test_default_weights_no_warning(self):
        """Default weights sum to 1.0 — no warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RecallConfig()  # should NOT raise

    def test_custom_weights_no_warning(self):
        """Custom weights that sum to 1.0 — no warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RecallConfig(
                semantic_weight=0.5,
                recency_weight=0.2,
                importance_weight=0.25,
                access_weight=0.05,
            )

    def test_unbalanced_weights_warns(self):
        """Weights that don't sum to 1.0 — should emit RuntimeWarning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RecallConfig(
                semantic_weight=0.8,
                recency_weight=0.3,
                importance_weight=0.3,
                access_weight=0.1,
            )  # sum = 1.5
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "1.500" in str(w[0].message)


# ── Input Truncation Tests ──────────────────────────────────────────────


class TestInputTruncation:
    @pytest.mark.asyncio
    async def test_short_text_not_truncated(self):
        """Normal-length text passes through unchanged."""
        svo_response = json.dumps([
            {"subject": "test", "relation": "is", "object": "fact", "importance": 0.5},
        ])
        config = make_config()
        writer = WritePipeline(
            config,
            mock_llm_client(svo_response),
            mock_embed_client(),
        )
        facts = await writer.ingest("Short text")
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_long_text_truncated(self):
        """Text over 50k chars gets truncated."""
        config = make_config()
        captured_prompts = []

        llm = AsyncMock()
        msg = MagicMock()
        msg.content = json.dumps([
            {"subject": "truncated", "relation": "is", "object": "text", "importance": 0.5},
        ])
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]

        async def capture_call(**kwargs):
            captured_prompts.append(kwargs["messages"][0]["content"])
            return resp

        llm.chat.completions.create = AsyncMock(side_effect=capture_call)

        writer = WritePipeline(config, llm, mock_embed_client())
        long_text = "x" * 60_000

        facts = await writer.ingest(long_text)

        # The prompt should contain truncated text (50k chars + SVO prompt template)
        assert len(captured_prompts) == 1
        assert len(captured_prompts[0]) < 60_000 + 500


# ── Public API Tests ────────────────────────────────────────────────────


class TestPublicAPI:
    def test_cosine_similarity_importable_from_utils(self):
        """cosine_similarity is a public function importable from _utils."""
        from hy_memory_fusion._utils import cosine_similarity
        assert callable(cosine_similarity)
        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_retry_importable_from_utils(self):
        """retry is a public function importable from _utils."""
        from hy_memory_fusion._utils import retry
        assert callable(retry)

    def test_version_exported(self):
        """Package exports version."""
        from hy_memory_fusion import __version__
        assert __version__ == "0.2.0"

    def test_all_public_classes_exported(self):
        """All public classes are exported from __init__."""
        from hy_memory_fusion import (
            FusionConfig,
            WritePipeline,
            ExtractedFact,
            ReadPipeline,
            RankedFact,
            MemoryCore,
        )
        assert FusionConfig is not None
        assert WritePipeline is not None
        assert ExtractedFact is not None
        assert ReadPipeline is not None
        assert RankedFact is not None
        assert MemoryCore is not None
