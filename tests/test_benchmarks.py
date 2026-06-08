"""Performance benchmarks — require a real Qdrant instance.

Run locally:
    docker run -d -p 6333:6333 qdrant/qdrant:latest
    FUSION_QDRANT_URL=http://localhost:6333 pytest tests/test_benchmarks.py -v -s

Tests measure and print timing; they fail only on unreasonable thresholds.
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
from hy_memory_fusion._utils import cosine_similarity, embed_batch
from unittest.mock import AsyncMock, MagicMock


QDRANT_URL = os.getenv("FUSION_QDRANT_URL", "http://localhost:6333")
DIM = 128  # realistic embedding dimension for benchmarks


def _random_collection() -> str:
    return f"bench_{uuid.uuid4().hex[:8]}"


def _make_bench_config(collection: str | None = None) -> FusionConfig:
    return FusionConfig(
        llm=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
        embedder=EmbedderConfig(base_url="http://unused", api_key="unused", model="unused", batch_size=64),
        qdrant=QdrantConfig(url=QDRANT_URL, collection=collection or _random_collection(), vector_dim=DIM),
        reader=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
        writer=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
    )


def _deterministic_vector(text: str, dim: int = DIM) -> list[float]:
    """Generate a deterministic vector from text (not random — reproducible)."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    # Expand to dim floats
    vec = []
    for i in range(dim):
        byte = h[i % len(h)]
        vec.append((byte / 255.0) * 2 - 1)  # range [-1, 1]
    # Normalize
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _mock_embed_client_deterministic():
    """Mock embed client with deterministic vectors."""
    client = AsyncMock()

    async def side_effect(**kwargs):
        inp = kwargs.get("input", "")
        mock_resp = MagicMock()
        if isinstance(inp, list):
            mock_resp.data = [MagicMock(embedding=_deterministic_vector(t)) for t in inp]
        else:
            mock_resp.data = [MagicMock(embedding=_deterministic_vector(inp))]
        return mock_resp

    client.embeddings.create = AsyncMock(side_effect=side_effect)
    return client


def _mock_llm_client_for_n_facts(n: int):
    """Mock LLM that extracts n facts."""
    facts = [
        {"subject": f"entity_{i}", "relation": "has_property", "object": f"value_{i}", "importance": 0.5 + (i % 5) * 0.1}
        for i in range(n)
    ]
    client = AsyncMock()
    msg = MagicMock()
    msg.content = json.dumps(facts)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def _mock_reader_client():
    client = AsyncMock()
    msg = MagicMock()
    msg.content = json.dumps({"answer": "test", "confidence": 0.9, "sources": [], "reasoning": "test"})
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


# ── Benchmarks ──────────────────────────────────────────────────────────


class TestCosineBenchmark:
    """Benchmark cosine_similarity at various dimensions."""

    def test_cosine_4d(self):
        a = _deterministic_vector("a", 4)
        b = _deterministic_vector("b", 4)
        iterations = 100_000

        start = time.perf_counter()
        for _ in range(iterations):
            cosine_similarity(a, b)
        elapsed = time.perf_counter() - start

        print(f"\n  cosine_similarity(4d) × {iterations}: {elapsed:.3f}s ({iterations/elapsed:.0f} ops/s)")
        assert elapsed < 10  # should be way under 10s

    def test_cosine_128d(self):
        a = _deterministic_vector("a", 128)
        b = _deterministic_vector("b", 128)
        iterations = 100_000

        start = time.perf_counter()
        for _ in range(iterations):
            cosine_similarity(a, b)
        elapsed = time.perf_counter() - start

        print(f"\n  cosine_similarity(128d) × {iterations}: {elapsed:.3f}s ({iterations/elapsed:.0f} ops/s)")
        assert elapsed < 30

    def test_cosine_1024d(self):
        a = _deterministic_vector("a", 1024)
        b = _deterministic_vector("b", 1024)
        iterations = 10_000

        start = time.perf_counter()
        for _ in range(iterations):
            cosine_similarity(a, b)
        elapsed = time.perf_counter() - start

        print(f"\n  cosine_similarity(1024d) × {iterations}: {elapsed:.3f}s ({iterations/elapsed:.0f} ops/s)")
        assert elapsed < 30


class TestDedupBenchmark:
    """Benchmark dedup at various scales."""

    @pytest.mark.asyncio
    async def test_dedup_100_facts(self):
        """Dedup 10 new facts against 100 existing (1000 comparisons)."""
        config = _make_bench_config()
        writer = WritePipeline(config, _mock_llm_client_for_n_facts(10), _mock_embed_client_deterministic())

        new_facts = [
            ExtractedFact(subject=f"new_{i}", relation="is", object=f"val_{i}", importance=0.5)
            for i in range(10)
        ]
        for f in new_facts:
            f.embedding = _deterministic_vector(f.text)

        existing = [
            {"text": f"existing_{i}", "embedding": _deterministic_vector(f"existing_{i}")}
            for i in range(100)
        ]

        start = time.perf_counter()
        result = await writer._dedup(new_facts, existing)
        elapsed = time.perf_counter() - start

        print(f"\n  dedup(10 new × 100 existing): {elapsed:.4f}s, {len(result)} kept")
        assert elapsed < 5  # should be fast

    @pytest.mark.asyncio
    async def test_dedup_1000_facts(self):
        """Dedup 50 new facts against 1000 existing (50k comparisons)."""
        config = _make_bench_config()
        writer = WritePipeline(config, _mock_llm_client_for_n_facts(50), _mock_embed_client_deterministic())

        new_facts = [
            ExtractedFact(subject=f"new_{i}", relation="is", object=f"val_{i}", importance=0.5)
            for i in range(50)
        ]
        for f in new_facts:
            f.embedding = _deterministic_vector(f.text)

        existing = [
            {"text": f"existing_{i}", "embedding": _deterministic_vector(f"existing_{i}")}
            for i in range(1000)
        ]

        start = time.perf_counter()
        result = await writer._dedup(new_facts, existing)
        elapsed = time.perf_counter() - start

        print(f"\n  dedup(50 new × 1000 existing): {elapsed:.4f}s, {len(result)} kept")
        assert elapsed < 30  # O(n×m) but should still be reasonable

    @pytest.mark.asyncio
    async def test_dedup_5000_facts(self):
        """Dedup 100 new facts against 5000 existing (500k comparisons)."""
        config = _make_bench_config()
        writer = WritePipeline(config, _mock_llm_client_for_n_facts(100), _mock_embed_client_deterministic())

        new_facts = [
            ExtractedFact(subject=f"new_{i}", relation="is", object=f"val_{i}", importance=0.5)
            for i in range(100)
        ]
        for f in new_facts:
            f.embedding = _deterministic_vector(f.text)

        existing = [
            {"text": f"existing_{i}", "embedding": _deterministic_vector(f"existing_{i}")}
            for i in range(5000)
        ]

        start = time.perf_counter()
        result = await writer._dedup(new_facts, existing)
        elapsed = time.perf_counter() - start

        print(f"\n  dedup(100 new × 5000 existing): {elapsed:.3f}s, {len(result)} kept")
        print(f"  throughput: {100*5000/elapsed:.0f} comparisons/s")
        # This will show the O(n×m) scaling issue
        assert elapsed < 120  # generous timeout


class TestRankBenchmark:
    """Benchmark ranking at various scales."""

    def _make_rank_data(self, n_facts: int):
        """Generate n_facts for ranking."""
        from hy_memory_fusion.read_pipeline import ReadPipeline

        config = _make_bench_config()
        config.recall.max_results = 10
        pipeline = ReadPipeline(config, _mock_reader_client(), _mock_embed_client_deterministic())

        query_embedding = _deterministic_vector("query", DIM)
        raw = [
            {
                "fact_id": f"f_{i}",
                "text": f"fact number {i} about topic {i % 10}",
                "embedding": _deterministic_vector(f"fact_{i}", DIM),
                "importance": 0.3 + (i % 7) * 0.1,
                "created_at": f"2026-01-{(i % 28) + 1:02d}T00:00:00Z",
                "access_count": i % 20,
            }
            for i in range(n_facts)
        ]
        return pipeline, query_embedding, raw

    def test_rank_100_facts(self):
        pipeline, query_emb, raw = self._make_rank_data(100)

        start = time.perf_counter()
        ranked = pipeline.rank(query_emb, raw)
        elapsed = time.perf_counter() - start

        print(f"\n  rank(100 facts, 128d): {elapsed:.4f}s")
        assert len(ranked) == 10  # max_results
        assert elapsed < 1

    def test_rank_1000_facts(self):
        pipeline, query_emb, raw = self._make_rank_data(1000)

        start = time.perf_counter()
        ranked = pipeline.rank(query_emb, raw)
        elapsed = time.perf_counter() - start

        print(f"\n  rank(1000 facts, 128d): {elapsed:.4f}s")
        assert len(ranked) == 10
        assert elapsed < 5

    def test_rank_10000_facts(self):
        pipeline, query_emb, raw = self._make_rank_data(10_000)

        start = time.perf_counter()
        ranked = pipeline.rank(query_emb, raw)
        elapsed = time.perf_counter() - start

        print(f"\n  rank(10000 facts, 128d): {elapsed:.3f}s")
        print(f"  throughput: {10000/elapsed:.0f} facts/s")
        assert len(ranked) == 10
        assert elapsed < 30


class TestQdrantBenchmark:
    """Benchmark real Qdrant operations."""

    @pytest.mark.asyncio
    async def test_bulk_upsert_1000(self):
        """Upsert 1000 facts into Qdrant."""
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        collection = _random_collection()
        client = AsyncQdrantClient(url=QDRANT_URL)

        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=f"f_{i}",
                vector=_deterministic_vector(f"fact_{i}", DIM),
                payload={"text": f"fact {i}", "importance": 0.5, "user_id": "bench"},
            )
            for i in range(1000)
        ]

        start = time.perf_counter()
        await client.upsert(collection_name=collection, points=points)
        elapsed = time.perf_counter() - start

        print(f"\n  qdrant.upsert(1000 points, {DIM}d): {elapsed:.3f}s")
        assert elapsed < 30

        await client.delete_collection(collection_name=collection)
        await client.close()

    @pytest.mark.asyncio
    async def test_search_latency(self):
        """Search latency with 1000 indexed facts."""
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        collection = _random_collection()
        client = AsyncQdrantClient(url=QDRANT_URL)

        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=f"f_{i}",
                vector=_deterministic_vector(f"fact_{i}", DIM),
                payload={"text": f"fact {i}", "importance": 0.5},
            )
            for i in range(1000)
        ]
        await client.upsert(collection_name=collection, points=points)

        query = _deterministic_vector("query", DIM)
        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            await client.query_points(collection_name=collection, query=query, limit=10)
        elapsed = time.perf_counter() - start

        print(f"\n  qdrant.search(1000 facts, {DIM}d) × {iterations}: {elapsed:.3f}s ({iterations/elapsed:.0f} queries/s)")
        assert elapsed < 30

        await client.delete_collection(collection_name=collection)
        await client.close()

    @pytest.mark.asyncio
    async def test_search_with_vectors_latency(self):
        """Search with_vectors=True latency (needed for hybrid_search)."""
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        collection = _random_collection()
        client = AsyncQdrantClient(url=QDRANT_URL)

        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=f"f_{i}",
                vector=_deterministic_vector(f"fact_{i}", DIM),
                payload={"text": f"fact {i}", "importance": 0.5},
            )
            for i in range(1000)
        ]
        await client.upsert(collection_name=collection, points=points)

        query = _deterministic_vector("query", DIM)
        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            await client.query_points(
                collection_name=collection,
                query=query,
                limit=10,
                with_payload=True,
                with_vectors=True,
            )
        elapsed = time.perf_counter() - start

        print(f"\n  qdrant.search(with_vectors, 1000 facts, {DIM}d) × {iterations}: {elapsed:.3f}s ({iterations/elapsed:.0f} queries/s)")
        assert elapsed < 30

        await client.delete_collection(collection_name=collection)
        await client.close()
