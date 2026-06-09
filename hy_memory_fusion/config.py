"""Configuration classes for hermes-memory-fusion."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM configuration for reasoning/distillation."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = "EMPTY"
    model: str = "nousresearch/hermes-3-llama-3.1-405b"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 30.0


@dataclass
class EmbedderConfig:
    """Embedding model configuration."""

    base_url: str = "http://honcho-ollama:11434/v1"
    api_key: str = "EMPTY"
    model: str = "mxbai-embed-large"
    batch_size: int = 32
    timeout: float = 15.0


@dataclass
class QdrantConfig:
    """Qdrant vector store configuration."""

    url: str = "http://localhost:6333"
    api_key: str = ""
    collection: str = "hermes_fusion"
    vector_dim: int = 1024


@dataclass
class DistillationConfig:
    """Hy-Memory style auto-distillation settings."""

    enabled: bool = True
    importance_threshold: float = 0.3
    dedup_threshold: float = 0.92
    scroll_limit: int = 500


@dataclass
class RecallConfig:
    """Read pipeline retrieval configuration.

    Weights should sum to ~1.0 for interpretable combined scores.
    A RuntimeWarning is emitted if they deviate by more than 0.01.
    """

    max_results: int = 10
    min_score: float = 0.3
    semantic_weight: float = 0.6
    recency_weight: float = 0.15
    importance_weight: float = 0.2
    access_weight: float = 0.05

    def __post_init__(self) -> None:
        total = self.semantic_weight + self.recency_weight + self.importance_weight + self.access_weight
        if abs(total - 1.0) > 0.01:
            warnings.warn(
                f"RecallConfig weights sum to {total:.3f}, expected ~1.0. "
                f"Scores may not be interpretable.",
                RuntimeWarning,
                stacklevel=2,
            )


@dataclass
class PipelineConfig:
    """Shared pipeline configuration."""

    timeout: float = 30.0
    max_retries: int = 2
    retry_delay: float = 1.0


@dataclass
class FusionConfig:
    """Top-level configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    reader: LLMConfig = field(default_factory=lambda: LLMConfig(
        base_url="http://localhost:8081/v1",
        api_key="EMPTY",
        model="local",
        temperature=0.3,
        max_tokens=2048,
    ))
    writer: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_env(cls) -> "FusionConfig":
        """Load config from environment variables."""
        import os

        cfg = cls()

        # LLM (distillation / main reasoning)
        if v := os.getenv("FUSION_LLM_BASE_URL"):
            cfg.llm.base_url = v
        if v := os.getenv("FUSION_LLM_API_KEY"):
            cfg.llm.api_key = v
        if v := os.getenv("FUSION_LLM_MODEL"):
            cfg.llm.model = v
        if v := os.getenv("FUSION_LLM_TIMEOUT"):
            cfg.llm.timeout = float(v)

        # Writer (SVO extraction — defaults to main LLM)
        if v := os.getenv("FUSION_WRITER_BASE_URL"):
            cfg.writer.base_url = v
        if v := os.getenv("FUSION_WRITER_API_KEY"):
            cfg.writer.api_key = v
        if v := os.getenv("FUSION_WRITER_MODEL"):
            cfg.writer.model = v
        else:
            cfg.writer = cfg.llm  # default: same as main LLM

        # Reader (synthesis / Honcho reasoning)
        if v := os.getenv("FUSION_READER_BASE_URL"):
            cfg.reader.base_url = v
        if v := os.getenv("FUSION_READER_API_KEY"):
            cfg.reader.api_key = v
        if v := os.getenv("FUSION_READER_MODEL"):
            cfg.reader.model = v
        if v := os.getenv("FUSION_READER_TEMPERATURE"):
            cfg.reader.temperature = float(v)
        if v := os.getenv("FUSION_READER_TIMEOUT"):
            cfg.reader.timeout = float(v)

        # Embedder
        if v := os.getenv("FUSION_EMBEDDER_BASE_URL"):
            cfg.embedder.base_url = v
        if v := os.getenv("FUSION_EMBEDDER_API_KEY"):
            cfg.embedder.api_key = v
        if v := os.getenv("FUSION_EMBEDDER_MODEL"):
            cfg.embedder.model = v
        if v := os.getenv("FUSION_EMBEDDER_BATCH_SIZE"):
            cfg.embedder.batch_size = int(v)
        if v := os.getenv("FUSION_EMBEDDER_TIMEOUT"):
            cfg.embedder.timeout = float(v)

        # Qdrant
        if v := os.getenv("FUSION_QDRANT_URL"):
            cfg.qdrant.url = v
        if v := os.getenv("FUSION_QDRANT_API_KEY"):
            cfg.qdrant.api_key = v
        if v := os.getenv("FUSION_QDRANT_COLLECTION"):
            cfg.qdrant.collection = v
        if v := os.getenv("FUSION_QDRANT_VECTOR_DIM"):
            cfg.qdrant.vector_dim = int(v)

        # Distillation
        if v := os.getenv("FUSION_DISTILLATION_ENABLED"):
            cfg.distillation.enabled = v.lower() in ("1", "true", "yes")
        if v := os.getenv("FUSION_DEDUP_THRESHOLD"):
            cfg.distillation.dedup_threshold = float(v)
        if v := os.getenv("FUSION_SCROLL_LIMIT"):
            cfg.distillation.scroll_limit = int(v)

        # Recall
        if v := os.getenv("FUSION_RECALL_MAX_RESULTS"):
            cfg.recall.max_results = int(v)
        if v := os.getenv("FUSION_RECALL_MIN_SCORE"):
            cfg.recall.min_score = float(v)
        if v := os.getenv("FUSION_RECALL_SEMANTIC_WEIGHT"):
            cfg.recall.semantic_weight = float(v)
        if v := os.getenv("FUSION_RECALL_RECENCY_WEIGHT"):
            cfg.recall.recency_weight = float(v)
        if v := os.getenv("FUSION_RECALL_IMPORTANCE_WEIGHT"):
            cfg.recall.importance_weight = float(v)
        if v := os.getenv("FUSION_RECALL_ACCESS_WEIGHT"):
            cfg.recall.access_weight = float(v)

        # Pipeline
        if v := os.getenv("FUSION_PIPELINE_TIMEOUT"):
            cfg.pipeline.timeout = float(v)
        if v := os.getenv("FUSION_PIPELINE_MAX_RETRIES"):
            cfg.pipeline.max_retries = int(v)
        if v := os.getenv("FUSION_PIPELINE_RETRY_DELAY"):
            cfg.pipeline.retry_delay = float(v)

        return cfg
