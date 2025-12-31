"""
Centralized configuration management for WildFrostRAG.

This module uses Pydantic Settings to manage all configuration values,
eliminating magic strings and providing type safety.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    All settings have sensible defaults where possible. Sensitive values
    (passwords, API keys) must be provided via environment variables.
    """

    # ===== Neo4j Configuration =====
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str
    neo4j_password: str

    # ===== Embedding Configuration =====
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    vector_index_name: str = "document-embeddings"
    similarity_function: str = "cosine"

    # ===== Web Scraping Configuration =====
    wildfrost_wiki_base_url: str = "https://wildfrostwiki.com"
    sitemap_url: str = "https://wildfrostwiki.com/sitemap.xml"
    max_concurrent_requests: int = 50

    # Special pages for tribe enrichment
    companions_page_url: str = "https://wildfrostwiki.com/Companions"
    items_page_url: str = "https://wildfrostwiki.com/Items"

    # ===== Directory Paths =====
    # Base directories
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"

    # Data subdirectories
    structured_outputs_dir: Path = data_dir / "structured_outputs"
    raw_htmls_dir: Path = data_dir / "raw_htmls"
    schemas_dir: Path = data_dir / "schemas"

    # Config directory
    configs_dir: Path = project_root / "configs"

    # ===== OpenAI Configuration (for evaluation) =====
    openai_api_key: Optional[str] = None
    openai_model_name: str = "gpt-4o-mini"  # Default model for generation
    openai_temperature: float = 0.0  # For deterministic responses
    openai_seed: int = 42  # Random seed for reproducibility

    # ===== Generation Pipeline Configuration =====
    default_k: int = 5  # Default number of chunks to retrieve
    default_batch_size: int = 25  # Default batch size for processing
    default_overwrite: bool = False  # Whether to overwrite existing responses by default

    model_config = SettingsConfigDict(
        env_file="configs/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def create_directories(self) -> None:
        """
        Create all necessary directories if they don't exist.

        This is useful for initial setup or ensuring the directory
        structure is correct before running the pipeline.
        """
        directories = [
            self.data_dir,
            self.structured_outputs_dir,
            self.raw_htmls_dir,
            self.schemas_dir,
            self.configs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
# Import this in other modules: from src.utils.config import settings
settings = Settings()
