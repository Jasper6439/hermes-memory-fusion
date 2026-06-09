1|"""Integration tests — require a real Qdrant instance.
2|
3|Run locally:
4|    docker run -d -p 6333:6333 qdrant/qdrant:latest
5|    FUSION_QDRANT_URL=http://localhost:6333 pytest tests/test_integration.py -v
6|
7|In CI: Qdrant runs as a service container (see .github/workflows/test.yml).
8|"""
9|
10|from __future__ import annotations
11|
12|import asyncio
13|import json
14|import os
15|import time
16|import uuid
17|
18|import pytest
19|
20|from hy_memory_fusion.config import FusionConfig, LLMConfig, EmbedderConfig, QdrantConfig
21|from hy_memory_fusion.memory_core import MemoryCore
22|from hy_memory_fusion.write_pipeline import WritePipeline, ExtractedFact
23|from hy_memory_fusion.read_pipeline import ReadPipeline, RankedFact
24|from hy_memory_fusion._utils import cosine_similarity
25|from unittest.mock import AsyncMock, MagicMock
26|
27|
28|# ── Helpers ──────────────────────────────────────────────────────────────
29|
30|QDRANT_URL = os.getenv("FUSION_QDRANT_URL", "http://localhost:6333")
31|
32|
33|def _random_collection() -> str:
34|    return f"test_integration_{uuid.uuid4().hex[:8]}"
35|
36|
37|def _make_integration_config(collection: str | None = None) -> FusionConfig:
38|    """Config that uses real Qdrant but mock LLM/embed."""
39|    return FusionConfig(
40|        llm=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
41|        embedder=EmbedderConfig(base_url="http://unused", api_key="unused", model="unused"),
42|        qdrant=QdrantConfig(
43|            url=QDRANT_URL,
44|            collection=collection or _random_collection(),
45|            vector_dim=4,
46|        ),
47|        reader=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
48|        writer=LLMConfig(base_url="http://unused", api_key="unused", model="unused"),
49|    )
50|
51|
52|def _mock_embed_client(vectors: dict[str, list[float]] | None = None):
53|    """Mock embed client that returns deterministic vectors for known texts."""
54|    client = AsyncMock()
55|
56|    async def side_effect(**kwargs):
57|        inp = kwargs.get("input", "")
58|        mock_resp = MagicMock()
59|        if isinstance(inp, list):
60|            mock_resp.data = []
61|            for text in inp:
62|                if vectors and text in vectors:
63|                    mock_resp.data.append(MagicMock(embedding=vectors[text]))
64|                else:
65|                    # Deterministic hash-based vector
66|                    h = hash(text) % 1000
67|                    mock_resp.data.append(MagicMock(embedding=[h / 1000, (h + 1) % 10 / 10, (h + 2) % 10 / 10, (h + 3) % 10 / 10]))
68|        else:
69|            if vectors and inp in vectors:
70|                mock_resp.data = [MagicMock(embedding=vectors[inp])]
71|            else:
72|                h = hash(inp) % 1000
73|                mock_resp.data = [MagicMock(embedding=[h / 1000, (h + 1) % 10 / 10, (h + 2) % 10 / 10, (h + 3) % 10 / 10])]
74|        return mock_resp
75|
76|    client.embeddings.create = AsyncMock(side_effect=side_effect)
77|    return client
78|
79|
80|def _mock_llm_client(svo_response: list[dict]):
81|    """Mock LLM client that returns a fixed SVO extraction result."""
82|    client = AsyncMock()
83|    msg = MagicMock()
84|    msg.content = json.dumps(svo_response)
85|    choice = MagicMock()
86|    choice.message = msg
87|    resp = MagicMock()
88|    resp.choices = [choice]
89|    client.chat.completions.create = AsyncMock(return_value=resp)
90|    return client
91|
92|
93|def _mock_reader_client(answer: str = "test answer"):
94|    """Mock reader LLM client."""
95|    client = AsyncMock()
96|    msg = MagicMock()
97|    msg.content = json.dumps({
98|        "answer": answer,
99|        "confidence": 0.9,
100|        "sources": [],
101|        "reasoning": "test",
102|    })
103|    choice = MagicMock()
104|    choice.message = msg
105|    resp = MagicMock()
106|    resp.choices = [choice]
107|    client.chat.completions.create = AsyncMock(return_value=resp)
108|    return client
109|
110|
111|# ── Qdrant Connectivity ─────────────────────────────────────────────────
112|
113|
114|class TestQdrantConnectivity:
115|    @pytest.mark.asyncio
116|    async def test_qdrant_is_reachable(self):
117|        """Verify Qdrant is running and healthy."""
118|        from qdrant_client import AsyncQdrantClient
119|
120|        client = AsyncQdrantClient(url=QDRANT_URL)
121|        collections = await client.get_collections()
122|        assert hasattr(collections, "collections")
123|        await client.close()
124|
125|    @pytest.mark.asyncio
126|    async def test_create_and_delete_collection(self):
127|        """Create a collection, verify it exists, delete it."""
128|        from qdrant_client import AsyncQdrantClient
129|        from qdrant_client.models import Distance, VectorParams
130|
131|        name = _random_collection()
132|        client = AsyncQdrantClient(url=QDRANT_URL)
133|
134|        await client.create_collection(
135|            collection_name=name,
136|            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
137|        )
138|
139|        collections = await client.get_collections()
140|        names = [c.name for c in collections.collections]
141|        assert name in names
142|
143|        await client.delete_collection(collection_name=name)
144|        collections = await client.get_collections()
145|        names = [c.name for c in collections.collections]
146|        assert name not in names
147|
148|        await client.close()
149|
150|
151|# ── Integration: Write Pipeline ─────────────────────────────────────────
152|
153|
154|class TestWritePipelineIntegration:
155|    @pytest.mark.asyncio
156|    async def test_ingest_stores_in_qdrant(self):
157|        """Full write pipeline: SVO extract → embed → store in Qdrant."""
158|        config = _make_integration_config()
159|        svo = [
160|            {"subject": "Alice", "relation": "likes", "object": "coffee", "importance": 0.8},
161|        ]
162|
163|        writer = WritePipeline(
164|            config,
165|            _mock_llm_client(svo),
166|            _mock_embed_client({"Alice likes coffee": [0.9, 0.1, 0.0, 0.0]}),
167|        )
168|
169|        facts = await writer.ingest("Alice likes coffee")
170|        assert len(facts) == 1
171|        assert facts[0].subject == "Alice"
172|        assert len(facts[0].embedding) == 4
173|
174|        # Cleanup
175|        from qdrant_client import AsyncQdrantClient
176|        client = AsyncQdrantClient(url=QDRANT_URL)
177|        try:
178|            await client.delete_collection(collection_name=config.qdrant.collection)
179|        except Exception:
180|            pass
181|        await client.close()
182|
183|    @pytest.mark.asyncio
184|    async def test_ingest_with_real_dedup(self):
185|        """Ingest same fact twice — second should be deduped via intra-batch."""
186|        config = _make_integration_config()
187|        config.distillation.dedup_threshold = 0.95
188|
189|        svo = [
190|            {"subject": "Bob", "relation": "runs", "object": "marathon", "importance": 0.7},
191|        ]
192|        same_embedding = [0.5, 0.5, 0.3, 0.1]
193|
194|        writer = WritePipeline(
195|            config,
196|            _mock_llm_client(svo),
197|            _mock_embed_client({"Bob runs marathon": same_embedding}),
198|        )
199|
200|        # First ingest — creates one fact
201|        facts1 = await writer.ingest("Bob runs marathon")
202|        assert len(facts1) == 1
203|
204|        # Create a new fact with same embedding and dedup against the first
205|        new_fact = ExtractedFact(subject="Bob", relation="runs", object="marathon", importance=0.7)
206|        new_fact.embedding = same_embedding  # same embedding as existing
207|
208|        existing = [{"text": "Bob runs marathon", "embedding": same_embedding}]
209|
210|        deduped = await writer._dedup([new_fact], existing)
211|        assert len(deduped) == 0  # deduped because embeddings match
212|
213|        # Cleanup
214|        from qdrant_client import AsyncQdrantClient
215|        client = AsyncQdrantClient(url=QDRANT_URL)
216|        try:
217|            await client.delete_collection(collection_name=config.qdrant.collection)
218|        except Exception:
219|            pass
220|        await client.close()
221|
222|
223|# ── Integration: MemoryCore end-to-end ──────────────────────────────────
224|
225|
226|class TestMemoryCoreIntegration:
227|    @pytest.mark.asyncio
228|    async def test_initialize_creates_collection(self):
229|        """MemoryCore.initialize() should create collection in real Qdrant."""
230|        config = _make_integration_config()
231|        core = MemoryCore(
232|            config=config,
234|            embed_client=_mock_embed_client(),
235|            reader_client=_mock_reader_client(),
236|            writer_client=_mock_llm_client([]),
237|        )
238|
239|        await core.initialize()
240|        assert core._initialized
241|
242|        # Verify collection exists
243|        from qdrant_client import AsyncQdrantClient
244|        client = AsyncQdrantClient(url=QDRANT_URL)
245|        collections = await client.get_collections()
246|        names = [c.name for c in collections.collections]
247|        assert config.qdrant.collection in names
248|        await client.close()
249|
250|        # Cleanup
251|        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)
252|
253|    @pytest.mark.asyncio
254|    async def test_remember_then_search(self):
255|        """Full cycle: remember → search returns the fact."""
256|        config = _make_integration_config()
257|        config.distillation.importance_threshold = 0.0
258|
259|        svo = [{"subject": "Eve", "relation": "codes", "object": "Python", "importance": 0.9}]
260|        embed_map = {"Eve codes Python": [0.8, 0.2, 0.1, 0.0]}
261|
262|        core = MemoryCore(
263|            config=config,
265|            embed_client=_mock_embed_client(embed_map),
266|            reader_client=_mock_reader_client("Eve codes in Python"),
267|            writer_client=_mock_llm_client(svo),
268|        )
269|        await core.initialize()
270|
271|        # Remember
272|        result = await core.remember("Eve codes Python")
273|        assert len(result) == 1
274|        assert result[0]["subject"] == "Eve"
275|
276|        # Wait for Qdrant to index
277|        await asyncio.sleep(0.5)
278|
279|        # Search
280|        results = await core.hybrid_search("Eve codes", mode="semantic")
281|        assert len(results) >= 1
282|        assert results[0]["fact_id"] is not None
283|
284|        # Cleanup
285|        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)
286|
287|    @pytest.mark.asyncio
288|    async def test_remember_then_recall(self):
289|        """Full cycle: remember → recall returns synthesized answer."""
290|        config = _make_integration_config()
291|        config.distillation.importance_threshold = 0.0
292|
293|        svo = [{"subject": "Carol", "relation": "drinks", "object": "tea", "importance": 0.6}]
294|
295|        core = MemoryCore(
296|            config=config,
298|            embed_client=_mock_embed_client({"Carol drinks tea": [0.3, 0.7, 0.1, 0.0]}),
299|            reader_client=_mock_reader_client("Carol drinks tea"),
300|            writer_client=_mock_llm_client(svo),
301|        )
302|        await core.initialize()
303|
304|        await core.remember("Carol drinks tea")
305|        await asyncio.sleep(0.5)
306|
307|        result = await core.recall("What does Carol drink?")
308|        assert "answer" in result
309|        assert "facts" in result
310|        assert len(result["facts"]) >= 1
311|
312|        # Cleanup
313|        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)
314|
315|    @pytest.mark.asyncio
316|    async def test_hybrid_search_with_filters(self):
317|        """hybrid_search with user_id filter."""
318|        config = _make_integration_config()
319|        config.distillation.importance_threshold = 0.0
320|
321|        svo = [{"subject": "test", "relation": "is", "object": "fact", "importance": 0.5}]
322|
323|        core = MemoryCore(
324|            config=config,
326|            embed_client=_mock_embed_client({"test fact": [0.1, 0.2, 0.3, 0.4]}),
327|            reader_client=_mock_reader_client(),
328|            writer_client=_mock_llm_client(svo),
329|        )
330|        await core.initialize()
331|
332|        await core.remember("test fact", user_id="user_a")
333|        await asyncio.sleep(0.5)
334|
335|        # Search with user filter
336|        results = await core.hybrid_search("test", mode="semantic", user_id="user_a")
337|        assert len(results) >= 1
338|
339|        # Search with wrong user — should find nothing
340|        results_other = await core.hybrid_search("test", mode="semantic", user_id="user_b")
341|        assert len(results_other) == 0
342|
343|        # Cleanup
344|        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)
345|
346|    @pytest.mark.asyncio
347|    async def test_get_facts_by_ids(self):
348|        """get_facts_by_ids retrieves stored facts by ID."""
349|        config = _make_integration_config()
350|        config.distillation.importance_threshold = 0.0
351|
352|        svo = [{"subject": "Dave", "relation": "reads", "object": "books", "importance": 0.7}]
353|
354|        core = MemoryCore(
355|            config=config,
357|            embed_client=_mock_embed_client({"Dave reads books": [0.4, 0.6, 0.2, 0.8]}),
358|            reader_client=_mock_reader_client(),
359|            writer_client=_mock_llm_client(svo),
360|        )
361|        await core.initialize()
362|
363|        result = await core.remember("Dave reads books")
364|        fact_id = result[0]["fact_id"]
365|
366|        retrieved = await core.get_facts_by_ids([fact_id])
367|        assert len(retrieved) == 1
368|        assert retrieved[0]["fact_id"] == fact_id
369|        assert retrieved[0]["text"] == "Dave reads books"
370|
371|        # Cleanup
372|        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)
373|
374|    @pytest.mark.asyncio
375|    async def test_update_access_increments(self):
376|        """update_access actually increments the counter in Qdrant."""
377|        config = _make_integration_config()
378|        config.distillation.importance_threshold = 0.0
379|
380|        svo = [{"subject": "test", "relation": "is", "object": "counter", "importance": 0.5}]
381|
382|        core = MemoryCore(
383|            config=config,
385|            embed_client=_mock_embed_client({"test counter": [0.1, 0.1, 0.1, 0.1]}),
386|            reader_client=_mock_reader_client(),
387|            writer_client=_mock_llm_client(svo),
388|        )
389|        await core.initialize()
390|
391|        result = await core.remember("test counter")
392|        fact_id = result[0]["fact_id"]
393|
394|        # Access 3 times
395|        for _ in range(3):
396|            await core.update_access([fact_id])
397|
398|        # Verify
399|        facts = await core.get_facts_by_ids([fact_id])
400|        assert facts[0]["access_count"] == 3
401|
402|        # Cleanup
403|        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)
404|
405|    @pytest.mark.asyncio
406|    async def test_empty_recall(self):
407|        """recall on empty collection returns no results."""
408|        config = _make_integration_config()
409|
410|        core = MemoryCore(
411|            config=config,
413|            embed_client=_mock_embed_client(),
414|            reader_client=_mock_reader_client(),
415|            writer_client=_mock_llm_client([]),
416|        )
417|        await core.initialize()
418|
419|        result = await core.recall("anything")
420|        assert "No relevant" in result["answer"]
421|
422|        # Cleanup
423|        await core._qdrant.delete_collection(collection_name=config.qdrant.collection)
424|