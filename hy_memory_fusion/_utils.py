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
