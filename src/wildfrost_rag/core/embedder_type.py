"""Single source of truth for the embedding-provider identifiers.

Mirrors domain.retriever_type.RetrieverType - a real type for provider
selection instead of a bare str checked against
settings.embedding.embedding_configs at runtime. Lives in core/, not
domain/ or services/, specifically so core/config.py can use it to key
embedding_configs without a layers-contract violation (core is the
bottom layer; domain/services sit above it and can't be imported from
core, so the enum has to live at or below the layer that needs it).
"""

from enum import StrEnum


class EmbedderType(StrEnum):
    """Every embedding provider add_embeddings/card_embeddings knows how to call."""

    HF = "hf"
    OPENAI = "openai"
    GEMMA = "gemma"
