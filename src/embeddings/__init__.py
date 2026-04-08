"""Embeddings module for WildFrostRAG."""

from embeddings.generator import (
    EmbeddingGenerator,
    load_embedding_model,
    generate_embeddings,
)

__all__ = ["EmbeddingGenerator", "load_embedding_model", "generate_embeddings"]
