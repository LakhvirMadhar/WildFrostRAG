"""Single source of truth for the embedding-provider identifiers.

Mirrors services.retrieval.retriever_type.RetrieverType - a real type for
provider selection instead of a bare str checked against
settings.embedding.embedding_configs at runtime.
"""

from enum import StrEnum


class EmbedderType(StrEnum):
    """Every embedding provider add_embeddings/card_embeddings knows how to call."""

    HF = "hf"
    OPENAI = "openai"
    GEMMA = "gemma"
