"""Centralized configuration management for WildFrostRAG.

This module uses Pydantic Settings to manage all configuration values,
eliminating magic strings and providing type safety.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jSettings(BaseSettings):
    """Neo4j connection configuration."""

    uri: SecretStr
    username: str
    password: SecretStr

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class OpenAISettings(BaseSettings):
    """OpenAI API and per-use-case LLM configuration."""

    api_key: SecretStr | None = None
    model_name: str = "gpt-4.1-nano"  # Default model for generation
    temperature: float = 0.0  # For deterministic responses
    seed: int = 42  # Random seed for reproducibility

    llm_semaphore_limit: int = 50  # Max concurrent LLM calls

    text2cypher_temperature: float = 1.0  # gpt-5-mini only supports temperature=1
    text2cypher_model: str = "gpt-5-mini"

    taxonomy_temperature: float = 0.3  # Slightly creative for categorization
    taxonomy_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class EmbeddingSettings(BaseSettings):
    """Embedding model, vector index, and retrieval-fusion configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    vector_index_name: str = "document-embeddings"
    similarity_function: str = "cosine"

    # Multi-embedder support: maps embedder name -> configuration
    embedding_configs: dict[str, dict[str, Any]] = {
        "hf": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "property_name": "hf_embedding",
            "index_name": "document-embeddings-hf",
        },
        "openai": {
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "property_name": "openai_embedding",
            "index_name": "document-embeddings-openai",
        },
        "gemma": {
            "model": "embeddinggemma",
            "dimension": 768,
            "property_name": "gemma_embedding",
            "index_name": "document-embeddings-gemma",
        },
    }

    fulltext_index_name: str = "document-fulltext"
    fulltext_index_name_sw: str = "document-fulltext-sw"  # With stop word removal
    bm25_index_name: str = "Document"
    rrf_k1: int = 60  # Smoothing parameter for Reciprocal Rank Fusion

    default_k: int = 5  # Default number of chunks to retrieve
    default_batch_size: int = 25  # Default batch size for processing

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class ScrapingSettings(BaseSettings):
    """Web scraping target configuration."""

    wildfrost_wiki_base_url: str = "https://wildfrostwiki.com"
    sitemap_url: str = "https://wildfrostwiki.com/sitemap.xml"
    max_concurrent_requests: int = 50

    # Special pages for tribe enrichment
    companions_page_url: str = "https://wildfrostwiki.com/Companions"
    items_page_url: str = "https://wildfrostwiki.com/Items"

    model_config = SettingsConfigDict(
        env_prefix="SCRAPING_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class PathSettings(BaseSettings):
    """Directory paths for data, outputs, and schemas.

    data_dir/outputs_dir and their subdirectories are computed properties
    derived from project_root, not independently-defaulted fields - a
    Pydantic field default referencing another field is baked in at class
    definition time, so it would silently ignore an overridden
    PATH_PROJECT_ROOT instead of tracking it.
    """

    project_root: Path = Path(__file__).parent.parent.parent

    model_config = SettingsConfigDict(
        env_prefix="PATH_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def data_dir(self) -> Path:
        """Directory for scraped/processed data."""
        return self.project_root / "data"

    @property
    def outputs_dir(self) -> Path:
        """Directory for experiment outputs."""
        return self.project_root / "outputs"

    @property
    def structured_outputs_dir(self) -> Path:
        """Directory for structured card data by CardType."""
        return self.data_dir / "structured_outputs"

    @property
    def raw_htmls_dir(self) -> Path:
        """Directory for raw scraped HTML files."""
        return self.data_dir / "raw_htmls"

    @property
    def schemas_dir(self) -> Path:
        """Directory for JSON card type schemas."""
        return self.data_dir / "schemas"


class Settings:
    """Composed application settings.

    Holds each concern as a typed sub-settings instance. Each sub-settings
    loads its own env vars independently (see each class's env_prefix).
    """

    def __init__(self) -> None:
        """Instantiate each sub-settings, each loading its own env vars."""
        self.neo4j = Neo4jSettings()  # type: ignore[call-arg]
        self.openai = OpenAISettings()
        self.embedding = EmbeddingSettings()
        self.scraping = ScrapingSettings()
        self.paths = PathSettings()

    def create_directories(self) -> None:
        """Create all necessary directories if they don't exist.

        This is useful for initial setup or ensuring the directory
        structure is correct before running the pipeline.
        """
        directories = [
            self.paths.data_dir,
            self.paths.structured_outputs_dir,
            self.paths.raw_htmls_dir,
            self.paths.schemas_dir,
            self.paths.outputs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_retrieval_output_dir(
        self, run_num: int, retriever_type: str, experiment_id: str
    ) -> Path:
        """Get path for retrieval experiment output directory.

        Args:
            run_num: Run number
            retriever_type: Type of retriever (bm25, vector, text2cypher, etc.)
            experiment_id: Experiment ID (e.g., "001")

        Returns:
            Path to retrieval experiment directory
            (e.g., outputs/run_1/retrievals/bm25/001)
        """
        return (
            self.paths.outputs_dir
            / f"run_{run_num}"
            / "retrievals"
            / retriever_type
            / experiment_id
        )

    def get_generation_output_dir(self, run_num: int, experiment_id: str) -> Path:
        """Get path for generation experiment output directory.

        Args:
            run_num: Run number
            experiment_id: Experiment ID (e.g., "001")

        Returns:
            Path to generation experiment directory
            (e.g., outputs/run_1/generation/001)
        """
        return self.paths.outputs_dir / f"run_{run_num}" / "generation" / experiment_id


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance, building it on first call.

    Importing this module never requires real credentials to be present;
    only calling this function does.

    Returns:
        The cached Settings instance (built once, reused via lru_cache).
    """
    return Settings()
