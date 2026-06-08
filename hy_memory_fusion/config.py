"""Unified configuration for memory fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorStoreConfig:
    """Qdrant vector store configuration."""

    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection_name: str = "hermes_memory"
    embedding_dims: int = 1024


@dataclass
class EmbedderConfig:
    """Embedding model configuration (OpenAI-compatible)."""

    model: str = "mxbai-embed-large"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"


@dataclass
class LLMConfig:
    """LLM configuration for distillation and dialectic reasoning (OpenAI-compatible)."""

    model: str = "gemini-2.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass
class DistillationConfig:
    """Write-pipeline distillation settings (Hy-Memory style)."""

    enabled: bool = True
    extract_svo: bool = True  # Subject-Verb-Object triplets
    dedup_threshold: float = 0.92  # Cosine similarity for dedup
    auto_capture: bool = True  # Auto-capture from conversation


@dataclass
class DialecticConfig:
    """Read-pipeline dialectic reasoning settings (Honcho style)."""

    enabled: bool = True
    depth: str = "medium"  # minimal / low / medium / high / max
    max_evidence: int = 10  # Max evidence snippets to synthesize
    system_prompt: str = (
        "You are a memory reasoning engine. Given a query and evidence snippets "
        "from the user's memory, synthesize a comprehensive, accurate answer. "
        "Cite which evidence supports each claim. If evidence contradicts, note the conflict."
    )


@dataclass
class RecallConfig:
    """Search/recall configuration."""

    top_k: int = 10
    semantic_weight: float = 0.5
    recency_weight: float = 0.3
    importance_weight: float = 0.15
    access_weight: float = 0.05
    min_score: float = 0.3


@dataclass
class FusionConfig:
    """Top-level configuration for the memory fusion system."""

    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    dialectic: DialecticConfig = field(default_factory=DialecticConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)

    @classmethod
    def from_env(cls, env_path: str = ".env") -> FusionConfig:
        """Load config from a .env file."""
        from dotenv import dotenv_values

        env = dotenv_values(env_path)
        return cls(
            vector_store=VectorStoreConfig(
                url=env.get("MEMORY_VECTOR_STORE_URL", "http://localhost:6333"),
                api_key=env.get("MEMORY_VECTOR_STORE_API_KEY"),
                collection_name=env.get("MEMORY_VECTOR_STORE_COLLECTION_NAME", "hermes_memory"),
                embedding_dims=int(env.get("MEMORY_VECTOR_STORE_EMBEDDING_DIMS", "1024")),
            ),
            embedder=EmbedderConfig(
                model=env.get("MEMORY_EMBEDDER_MODEL", "mxbai-embed-large"),
                base_url=env.get("MEMORY_EMBEDDER_BASE_URL", "http://localhost:11434/v1"),
                api_key=env.get("MEMORY_EMBEDDER_API_KEY", "ollama"),
            ),
            llm=LLMConfig(
                model=env.get("MEMORY_LLM_MODEL", "gemini-2.5-flash"),
                base_url=env.get("MEMORY_LLM_BASE_URL", ""),
                api_key=env.get("MEMORY_LLM_API_KEY", ""),
            ),
        )
