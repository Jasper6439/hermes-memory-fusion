# hermes-memory-fusion

A hybrid memory system combining **Hy-Memory's write pipeline** (SVO extraction + deduplication) with **Honcho's read pipeline** (multi-signal ranking + dialectic reasoning), unified on a single Qdrant backend.

## Architecture

```
┌─────────────────────────────────────────────┐
│              MemoryCore                      │
│  remember() ←────→ recall()                 │
│       │                   │                  │
│  ┌────▼────┐         ┌───▼────┐             │
│  │  Write   │         │  Read   │             │
│  │ Pipeline │         │ Pipeline│             │
│  │(Hy-Memory)│        │(Honcho) │             │
│  └────┬────┘         └───┬────┘             │
│       │                   │                  │
│  ┌────▼───────────────────▼────┐             │
│  │        Qdrant Vector DB      │             │
│  └──────────────────────────────┘             │
└─────────────────────────────────────────────┘
```

## Write Pipeline (from Hy-Memory)

- **SVO Extraction**: LLM extracts Subject-Verb-Object triplets from raw text
- **Batch Embedding**: Single API call for all extracted facts
- **Semantic Deduplication**: Cosine similarity threshold (default 0.92) removes duplicates against existing memories
- **Importance Filtering**: Only stores facts above configurable threshold (default 0.3)

## Read Pipeline (from Honcho)

- **Multi-Signal Ranking**: Four weighted signals:
  - Semantic similarity (60%) — cosine similarity of query vs fact embeddings
  - Recency (15%) — exponential decay with 30-day half-life
  - Importance (20%) — fact's self-assigned importance score
  - Access frequency (5%) — logarithmic saturation at ~10 accesses
- **Dialectic Reasoning**: 5 levels (minimal → max) with appropriate prompt depth
- **Access Counter Updates**: Fire-and-forget increment on retrieval

## Quick Start

```bash
pip install -e ".[dev]"
```

```python
from hy_memory_fusion import MemoryCore

core = MemoryCore()
await core.initialize()

# Write
await core.remember("Alice likes coffee with oat milk")

# Read
result = await core.recall("What does Alice like to drink?")
print(result["answer"])
```

## Configuration

All config via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FUSION_LLM_BASE_URL` | openrouter | Main LLM endpoint |
| `FUSION_LLM_MODEL` | hermes-3-405b | Main LLM model |
| `FUSION_WRITER_MODEL` | same as LLM | SVO extraction model |
| `FUSION_READER_MODEL` | local llama-server | Synthesis model |
| `FUSION_EMBEDDER_MODEL` | mxbai-embed-large | Embedding model |
| `FUSION_QDRANT_URL` | localhost:6333 | Qdrant endpoint |
| `FUSION_DEDUP_THRESHOLD` | 0.92 | Dedup similarity threshold |
| `FUSION_PIPELINE_MAX_RETRIES` | 2 | LLM call retries |
| `FUSION_PIPELINE_TIMEOUT` | 30.0 | LLM call timeout (seconds) |
| `FUSION_RECALL_SEMANTIC_WEIGHT` | 0.6 | Semantic signal weight |
| `FUSION_RECALL_RECENCY_WEIGHT` | 0.15 | Recency signal weight |
| `FUSION_RECALL_IMPORTANCE_WEIGHT` | 0.2 | Importance signal weight |
| `FUSION_RECALL_ACCESS_WEIGHT` | 0.05 | Access frequency weight |

## Testing

```bash
# Unit tests (no live services needed)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=hy_memory_fusion --cov-report=term-missing
```

## Three-Layer Memory Architecture

This project implements Layer 3 (Qdrant) of the Hermes memory stack:

| Layer | Component | Purpose |
|-------|-----------|---------|
| L1 | MEMORY.md | Hot cache (<50% capacity) |
| L2 | Honcho | Dialectic reasoning + session context |
| L3 | **This project** | Structured facts in Qdrant |

## License

MIT
