"""Centralized configuration management for WildFrostRAG.

This module uses Pydantic Settings to manage all configuration values,
eliminating magic strings and providing type safety. Each concern's fields
live in core/configs/ (one file per external system) - this module is only
the composition root, building one Settings instance holding all of them.
"""

from functools import lru_cache

from wildfrost_rag.core.configs.embedding import EmbeddingSettings
from wildfrost_rag.core.configs.mlflow import MlflowSettings
from wildfrost_rag.core.configs.neo4j import Neo4jSettings
from wildfrost_rag.core.configs.openai import OpenAISettings
from wildfrost_rag.core.configs.paths import PathSettings
from wildfrost_rag.core.configs.scraping import ScrapingSettings

DEFAULT_QUERIES_FILE = "queries/simple_reference_based_queries.csv"
"""Relative path to the project's default query dataset.

A fixed project convention, not environment-specific configuration, so it's
a plain constant rather than a Settings field - it never needs to vary by
deployment the way Neo4j credentials or the OpenAI key do.
"""


class Settings:
    """Composed application settings - one instance per typed concern.

    Holds each concern as a typed sub-settings instance. Each sub-settings
    loads its own env vars independently (see each class's env_prefix, in
    core/configs/). Construction only - no other behavior belongs here.
    """

    def __init__(self) -> None:
        """Instantiate each sub-settings, each loading its own env vars."""
        self.neo4j = Neo4jSettings()
        self.openai = OpenAISettings()
        self.mlflow = MlflowSettings()
        self.embedding = EmbeddingSettings()
        self.scraping = ScrapingSettings()
        self.paths = PathSettings()


def create_settings_directories(settings: Settings) -> None:
    """Create all necessary directories if they don't exist.

    Useful for initial setup or ensuring the directory structure is correct
    before running the pipeline.
    """
    directories = [
        settings.paths.data_dir,
        settings.paths.structured_outputs_dir,
        settings.paths.raw_htmls_dir,
        settings.paths.schemas_dir,
        settings.paths.outputs_dir,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance, building it on first call.

    Importing this module never requires real credentials to be present;
    only calling this function does.

    Returns:
        The cached Settings instance (built once, reused via lru_cache).
    """
    return Settings()
