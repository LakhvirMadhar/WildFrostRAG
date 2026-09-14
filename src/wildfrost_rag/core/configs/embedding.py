"""Embedding provider metadata, vector/fulltext index names, and retrieval hyperparameters."""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from wildfrost_rag.core.embedder_type import EmbedderType


class EmbedderProviderConfig(BaseModel):
    """Configuration for one embedding provider (e.g. "hf", "openai", "gemma").

    Metadata, not a setting: the vendor/model name, the vector
    dimensionality it produces, and the Neo4j property/index names its
    embeddings are stored and searched under are facts about each provider,
    not tunable choices.

    This is the fixed shape every entry in `EmbeddingSettings.embedding_configs`
    has. Lives here, not in models/, for the same reason EmbedderType lives
    in core/ - core/configs/embedding.py constructs instances of it directly,
    and core can't import from any sibling in its own layer bucket (core |
    domain | models | main), verified via a real lint-imports run.
    """

    model: str
    dimension: int
    property_name: str
    index_name: str


class EmbeddingSettings(BaseSettings):
    """Embedding model, vector index, and retrieval-fusion configuration."""

    # --- Legacy top-level defaults - superseded by embedding_configs below,
    # kept as-is for now (see the follow-up pass planned for this file) ---
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    vector_index_name: str = "document-embeddings"
    similarity_function: str = "cosine"

    # --- Metadata: facts about each embedding provider, keyed by provider ---
    embedding_configs: dict[EmbedderType, EmbedderProviderConfig] = {
        EmbedderType.HF: EmbedderProviderConfig(
            model="all-MiniLM-L6-v2",
            dimension=384,
            property_name="hf_embedding",
            index_name="document-embeddings-hf",
        ),
        EmbedderType.OPENAI: EmbedderProviderConfig(
            model="text-embedding-3-small",
            dimension=1536,
            property_name="openai_embedding",
            index_name="document-embeddings-openai",
        ),
        EmbedderType.GEMMA: EmbedderProviderConfig(
            model="embeddinggemma",
            dimension=768,
            property_name="gemma_embedding",
            index_name="document-embeddings-gemma",
        ),
    }

    # --- Metadata: fixed Neo4j index names ---
    fulltext_index_name: str = "document-fulltext"
    fulltext_index_name_sw: str = "document-fulltext-sw"  # With stop word removal
    bm25_index_name: str = "Document"

    # --- Hyperparameters: tunable, affect retrieval quality ---
    rrf_k1: int = 60  # Smoothing parameter for Reciprocal Rank Fusion
    default_k: int = 5  # Default number of chunks to retrieve

    # --- Operational tuning ---
    default_batch_size: int = 25  # Default batch size for processing

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
