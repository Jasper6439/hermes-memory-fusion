"""Integration tests — require a real Qdrant instance.

Run locally:
    docker run -d -p 6333:6333 qdrant/qdrant:latest
    FUSION_QDRANT_URL=http://localhost:6333 pytest tests/test_integration.py -v

In CI: Qdrant runs as a service container (see .github/workflows/test.yml).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

import pytest

from hy_memory_fusion.config import FusionConfig, LLMConfig, EmbedderConfig, QdrantConfig
from hy_memory_fusion.memory_core import MemoryCore
from hy_memory_fusion.write_pipeline import WritePipeline, ExtractedFact
from hy_memory_fusion.read_pipeline import ReadPipeline, RankedFact
from hy_memory_fusion._utils import cosine_similarity
from unittest.mock import AsyncMock, MagicMock


# ── Helpers ──────────────────────────────────────────────────────────────

QDRANT_URL = os.getenv("FUSION_QDRANT_URL", "http://localhost:6333")


def _random_collection() -> str:
    return f"test_integration_{uuid.uuid4().hex[:8]}"


def _make_integration_config(collection: str | None = None) -> FusionConfig:
    """Config that uses real Qdrant but mock LLM/embed."""
    return FusionConfig(
        llm=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
        embedder=EmbedderConfig(base_url="http://unused", api_key="unused", model="unused"),
        qdrant=QdrantConfig(
            url=QDRANT_URL,
            collection=collection or _random_collection(),
            vector_dim=4,
        ),
        reader=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
        writer=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
    )


def _mock_embed_client(vectors: dict[str, list[float]] | None = None):
    """Mock embed client that returns deterministic vectors for known texts."""
    client = AsyncMock()

    async def side_effect(**kwargs):
        inp = kwargs.get("input", "")
        mock_resp = MagicMock()
        if isinstance(inp, list):
            mock_resp.data = []
            for text in inp:
                if vectors and text in vectors:
                    mock_resp.data.append(MagicMock(embedding=vectors[text]))
                else:
                    # Deterministic hash-based vector
                    h = hash(text) % 1000
                    mock_resp.data.append(MagicMock(embedding=[h / 1000, (h + 1) % 10 / 10, (h + 2) % 10 / 10, (h + 3) % 10 / 10]))
        else:
            if vectors and inp in vectors:
                mock_resp.data = [MagicMock(embedding=vectors[inp])]
            else:
                h = hash(inp) % 1000
                mock_resp.data = [MagicMock(embedding=[h / 1000, (h + 1) % 10 / 10, (h + 2) % 10 / 10, (h + 3) % 10 / 10])]
        return mock_resp

    client.embeddings.create = AsyncMock(side_effect=side_effect)
    return client


def _mock_llm_client(svo_response: list[dict]):
    """Mock LLM client that returns a fixed SVO extraction result."""
    client = AsyncMock()
    msg = MagicMock()
    msg.content = json.dumps(svo_response)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def _mock_reader_client(answer: str = "test answer"):
    """Mock reader LLM client."""
    client = AsyncMock()
    msg = MagicMock()
    msg.content = json.dumps({
        "answer": answer,
        "confidence": 0.9,
        "sources": [],
        "reasoning": "test",
    })
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


# ── Qdrant Connectivity ─────────────────────────────────────────────────


class TestQdrantConnectivity:
    @pytest.mark.asyncio
    async def test_qdrant_is_reachable(self):
        """Verify Qdrant is running and healthy."""
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(url=QDRANT_URL)
        collections = await client.get_collections()
        assert hasattr(collections, "collections")
        await client.close()

    @pytest.mark.asyncio
    async def test_create_and_delete_collection(self):
        """Create a collection, verify it exists, delete it."""
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        name = _random_collection()
        client = AsyncQdrantClient(url=QDRANT_URL)

        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        assert name in names

        await client.delete_collection(collection_name=name)
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        assert name not in names

        await client.close()


# ── Integration: Write Pipeline ─────────────────────────────────────────


class TestWritePipelineIntegration:
    @pytest.mark.asyncio
    async def test_ingest_stores_in_qdrant(self):
        """Full write pipeline: SVO extract → embed → store in Qdrant."""
        config = _make_integration_config()
        svo = [
            {"subject": "Alice", "relation": "likes", "object": "coffee", "importance": 0.8},
        ]

        writer = WritePipeline(
            config,
            _mock_llm_client(svo),
            _mock_embed_client({"Alice likes coffee": [0.9, 0.1, 0.0, 0.0]}),
        )

        facts = await writer.ingest("Alice likes coffee")
        assert len(facts) == 1
        assert facts[0].subject == "Alice"
        assert len(facts[0].embedding) == 4

        # Cleanup
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=QDRANT_URL)
        try:
            await client.delete_collection(collection_name=config.qdrant.collection)
        except Exception:
            pass
        await client.close()

    @pytest.mark.asyncio
    async def test_ingest_with_real_dedup(self):
        """Ingest same fact twice — second should be deduped via intra-batch."""
        config = _make_integration_config()
        config.distillation.dedup_threshold = 0.95

        svo = [
            {"subject": "Bob", "relation": "runs", "object": "marathon", "importance": 0.7},
        ]
        same_embedding = [0.5, 0.5, 0.3, 0.1]

        writer = WritePipeline(
            config,
            _mock_llm_client(svo),
            _mock_embed_client({"Bob runs marathon": same_embedding}),
        )

        # First ingest — creates one fact
        facts1 = await writer.ingest("Bob runs marathon")
        assert len(facts1) == 1

        # Create a new fact with same embedding and dedup against the first
        new_fact = ExtractedFact(subject="Bob", relation="runs", object="marathon", importance=0.7)
        new_fact.embedding = same_embedding  # same embedding as existing

        existing = [{"text": "Bob runs marathon", "embedding": same_embedding}]

        deduped = await writer._dedup([new_fact], existing)
        assert len(deduped) == 0  # deduped because embeddings match

        # Cleanup
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=QDRANT_URL)
        try:
            await client.delete_collection(collection_name=config.qdrant.collection)
        except Exception:
            pass
        await client.close()


# ── Integration: MemoryCore end-to-end ──────────────────────────────────


class TestMemoryCoreIntegration:
    @pytest.mark.asyncio
    async def test_initialize_creates_collection(self):
        """MemoryCore.initialize() should create collection in real Qdrant."""
        config = _make_integration_config()
        core = MemoryCore(
            config=config,
            llm_client=_mock_llm_client([]),
            embed_client=_mock_embed_client(),
            reader_client=_mock_reader_client(),
            writer_client=_mock_llm_client([]),
        )

        await core.initialize()
        assert core._initialized

        # Verify collection exists
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=QDRANT_URL)
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        assert config.qdrant.collection in names
        await client.close()

        # Cleanup
        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)

    @pytest.mark.asyncio
    async def test_remember_then_search(self):
        """Full cycle: remember → search returns the fact."""
        config = _make_integration_config()
        config.distillation.importance_threshold = 0.0

        svo = [{"subject": "Eve", "relation": "codes", "object": "Python", "importance": 0.9}]
        embed_map = {"Eve codes Python": [0.8, 0.2, 0.1, 0.0]}

        core = MemoryCore(
            config=config,
            llm_client=_mock_llm_client(svo),
            embed_client=_mock_embed_client(embed_map),
            reader_client=_mock_reader_client("Eve codes in Python"),
            writer_client=_mock_llm_client(svo),
        )
        await core.initialize()

        # Remember
        result = await core.remember("Eve codes Python")
        assert len(result) == 1
        assert result[0]["subject"] == "Eve"

        # Wait for Qdrant to index
        await asyncio.sleep(0.5)

        # Search
        results = await core.hybrid_search("Eve codes", mode="semantic")
        assert len(results) >= 1
        assert results[0]["fact_id"] is not None

        # Cleanup
        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)

    @pytest.mark.asyncio
    async def test_remember_then_recall(self):
        """Full cycle: remember → recall returns synthesized answer."""
        config = _make_integration_config()
        config.distillation.importance_threshold = 0.0

        svo = [{"subject": "Carol", "relation": "drinks", "object": "tea", "importance": 0.6}]

        core = MemoryCore(
            config=config,
            llm_client=_mock_llm_client(svo),
            embed_client=_mock_embed_client({"Carol drinks tea": [0.3, 0.7, 0.1, 0.0]}),
            reader_client=_mock_reader_client("Carol drinks tea"),
            writer_client=_mock_llm_client(svo),
        )
        await core.initialize()

        await core.remember("Carol drinks tea")
        await asyncio.sleep(0.5)

        result = await core.recall("What does Carol drink?")
        assert "answer" in result
        assert "facts" in result
        assert len(result["facts"]) >= 1

        # Cleanup
        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self):
        """hybrid_search with user_id filter."""
        config = _make_integration_config()
        config.distillation.importance_threshold = 0.0

        svo = [{"subject": "test", "relation": "is", "object": "fact", "importance": 0.5}]

        core = MemoryCore(
            config=config,
            llm_client=_mock_llm_client(svo),
            embed_client=_mock_embed_client({"test fact": [0.1, 0.2, 0.3, 0.4]}),
            reader_client=_mock_reader_client(),
            writer_client=_mock_llm_client(svo),
        )
        await core.initialize()

        await core.remember("test fact", user_id="user_a")
        await asyncio.sleep(0.5)

        # Search with user filter
        results = await core.hybrid_search("test", mode="semantic", user_id="user_a")
        assert len(results) >= 1

        # Search with wrong user — should find nothing
        results_other = await core.hybrid_search("test", mode="semantic", user_id="user_b")
        assert len(results_other) == 0

        # Cleanup
        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)

    @pytest.mark.asyncio
    async def test_get_facts_by_ids(self):
        """get_facts_by_ids retrieves stored facts by ID."""
        config = _make_integration_config()
        config.distillation.importance_threshold = 0.0

        svo = [{"subject": "Dave", "relation": "reads", "object": "books", "importance": 0.7}]

        core = MemoryCore(
            config=config,
            llm_client=_mock_llm_client(svo),
            embed_client=_mock_embed_client({"Dave reads books": [0.4, 0.6, 0.2, 0.8]}),
            reader_client=_mock_reader_client(),
            writer_client=_mock_llm_client(svo),
        )
        await core.initialize()

        result = await core.remember("Dave reads books")
        fact_id = result[0]["fact_id"]

        retrieved = await core.get_facts_by_ids([fact_id])
        assert len(retrieved) == 1
        assert retrieved[0]["fact_id"] == fact_id
        assert retrieved[0]["text"] == "Dave reads books"

        # Cleanup
        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)

    @pytest.mark.asyncio
    async def test_update_access_increments(self):
        """update_access actually increments the counter in Qdrant."""
        config = _make_integration_config()
        config.distillation.importance_threshold = 0.0

        svo = [{"subject": "test", "relation": "is", "object": "counter", "importance": 0.5}]

        core = MemoryCore(
            config=config,
            llm_client=_mock_llm_client(svo),
            embed_client=_mock_embed_client({"test counter": [0.1, 0.1, 0.1, 0.1]}),
            reader_client=_mock_reader_client(),
            writer_client=_mock_llm_client(svo),
        )
        await core.initialize()

        result = await core.remember("test counter")
        fact_id = result[0]["fact_id"]

        # Access 3 times
        for _ in range(3):
            await core.update_access([fact_id])

        # Verify
        facts = await core.get_facts_by_ids([fact_id])
        assert facts[0]["access_count"] == 3

        # Cleanup
        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)

    @pytest.mark.asyncio
    async def test_empty_recall(self):
        """recall on empty collection returns no results."""
        config = _make_integration_config()

        core = MemoryCore(
            config=config,
            llm_client=_mock_llm_client([]),
            embed_client=_mock_embed_client(),
            reader_client=_mock_reader_client(),
            writer_client=_mock_llm_client([]),
        )
        await core.initialize()

        result = await core.recall("anything")
        assert "No relevant" in result["answer"]

        # Cleanup
        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)
