# Hermes Memory Fusion

**Hy-Memory auto-distillation + Honcho dialectic reasoning = best of both worlds.**

## Architecture

```
Write (Hy-Memory style)              Read (Honcho style)
┌─────────────────────┐              ┌──────────────────────┐
│ Raw conversation    │              │ User query           │
│        ↓            │              │        ↓             │
│ LLM: SVO extraction │              │ Embed query          │
│        ↓            │              │        ↓             │
│ Embed each fact     │              │ Qdrant vector search │
│        ↓            │              │        ↓             │
│ Store in Qdrant     │              │ LLM: Dialectic       │
│                     │              │ reasoning (5 levels) │
└─────────────────────┘              │        ↓             │
                                     │ Answer + citations   │
         ┌───────────┐               └──────────────────────┘
         │  Qdrant   │
         │  Cloud    │←──────── both read/write
         └───────────┘
```

## What's borrowed from where

| Feature | Source | Description |
|---------|--------|-------------|
| SVO extraction | Hy-Memory | LLM extracts Subject-Verb-Object triplets |
| Auto-distillation | Hy-Memory | Async fact extraction from conversations |
| Deduplication | Hy-Memory | Cosine similarity threshold for duplicates |
| Dialectic reasoning | Honcho | Multi-level LLM synthesis (minimal→max) |
| Evidence citations | Honcho | Answer cites which facts support each claim |
| Contradiction detection | Honcho | Flags conflicting evidence |

## Quick Start

```python
from hy_memory_fusion import MemoryCore, FusionConfig

config = FusionConfig.from_env(".env")
core = MemoryCore(config)
core.connect()

# Write: auto-distill into structured facts
facts = await core.remember("Ulysses prefers dark mode. He manages OCI servers in Tokyo.")
# → Extracts 2 SVO facts, embeds, stores in Qdrant

# Read: search + dialectic reasoning
answer = await core.recall("What does the user prefer?", depth="medium")
# → Searches Qdrant, runs multi-level LLM synthesis
print(answer.answer)       # "Ulysses prefers dark mode UIs."
print(answer.confidence)   # 0.9
print(answer.citations)    # ["f_a1b2c3"]
```

## Configuration

Create a `.env` file:

```env
# Vector Store (Qdrant)
MEMORY_VECTOR_STORE_URL=https://your-qdrant-url
MEMORY_VECTOR_STORE_API_KEY=your-key
MEMORY_VECTOR_STORE_COLLECTION_NAME=hermes_memory
MEMORY_VECTOR_STORE_EMBEDDING_DIMS=1024

# Embedder (Ollama / OpenAI-compatible)
MEMORY_EMBEDDER_MODEL=mxbai-embed-large
MEMORY_EMBEDDER_BASE_URL=http://localhost:11434/v1
MEMORY_EMBEDDER_API_KEY=ollama

# LLM (Gemini / OpenAI-compatible)
MEMORY_LLM_MODEL=gemini-2.5-flash
MEMORY_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MEMORY_LLM_API_KEY=your-gemini-key
```

## Dialectic Depth Levels

| Level | Use case | Token cost |
|-------|----------|-----------|
| `minimal` | Quick lookup, 1-2 sentences | Low |
| `low` | Simple Q&A with citations | Low |
| `medium` | Comprehensive analysis | Medium |
| `high` | Multi-perspective deep analysis | High |
| `max` | Exhaustive audit-level report | Very high |

## Development

```bash
pip install -e ".[dev]"
pytest -v --cov=hy_memory_fusion
```

## License

MIT
