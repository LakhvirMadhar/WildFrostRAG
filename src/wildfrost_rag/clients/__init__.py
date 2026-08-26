"""clients layer for WildFrostRAG (scaffolded in T7.1, populated by later Epic 7 tickets)."""

from wildfrost_rag.clients.generator import (
    EmbeddingGenerator,
    load_embedding_model,
    generate_embeddings,
)

__all__ = ["EmbeddingGenerator", "load_embedding_model", "generate_embeddings"]
