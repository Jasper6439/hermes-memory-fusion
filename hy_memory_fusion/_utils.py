"""Shared utilities for hermes-memory-fusion."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry(
    coro_factory: Callable[..., Any],
    *,
    max_retries: int = 2,
    delay: float = 1.0,
    label: str = "call",
) -> Any:
    """Retry an async coroutine factory with exponential backoff.

    Args:
        coro_factory: Callable that returns an awaitable.
        max_retries: Max retry attempts (0 = no retries).
        delay: Base delay in seconds (doubles each retry).
        label: Label for log messages.

    Returns:
        Result of the coroutine.

    Raises:
        The last exception after all retries exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = delay * (2 ** attempt)
                logger.warning("%s attempt %d failed: %s, retrying in %.1fs", label, attempt + 1, e, wait)
                await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


async def embed_text(
    text: str,
    embed_client: Any,
    model: str,
    *,
    max_retries: int = 2,
    delay: float = 1.0,
) -> list[float]:
    """Shared single-text embedding helper with retry.

    Args:
        text: Text to embed.
        embed_client: AsyncOpenAI-compatible client.
        model: Model name.
        max_retries: Max retry attempts.
        delay: Base retry delay.

    Returns:
        Embedding vector, or empty list on failure.
    """

    async def _call():
        resp = await embed_client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    try:
        return await retry(_call, max_retries=max_retries, delay=delay, label="embed")
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return []


async def embed_batch(
    texts: list[str],
    embed_client: Any,
    model: str,
    batch_size: int = 32,
    *,
    max_retries: int = 2,
    delay: float = 1.0,
) -> list[list[float]]:
    """Shared batch embedding helper with retry.

    Args:
        texts: Texts to embed.
        embed_client: AsyncOpenAI-compatible client.
        model: Model name.
        batch_size: Max texts per API call.
        max_retries: Max retry attempts per batch.
        delay: Base retry delay.

    Returns:
        List of embedding vectors (one per input text).
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        async def _call(b=batch):
            resp = await embed_client.embeddings.create(model=model, input=b)
            return [d.embedding for d in resp.data]

        try:
            embeddings = await retry(
                _call, max_retries=max_retries, delay=delay,
                label=f"embed_batch[{i}:{i+batch_size}]",
            )
            all_embeddings.extend(embeddings)
        except Exception as e:
            logger.error("Batch embedding failed: %s", e)
            all_embeddings.extend([[] for _ in batch])

    return all_embeddings


def strip_markdown_json(text: str) -> str:
    """Strip markdown code block wrappers from LLM JSON responses.

    Handles ```json...``` and ```...``` patterns.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 for mismatched lengths or zero-norm vectors.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
