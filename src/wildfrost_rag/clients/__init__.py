"""clients layer for WildFrostRAG - external API/service wrappers (ACLs), no business logic."""

from wildfrost_rag.clients.generator import (
    EmbeddingGenerator,
    load_embedding_model,
    generate_embeddings,
)

__all__ = ["EmbeddingGenerator", "load_embedding_model", "generate_embeddings"]
