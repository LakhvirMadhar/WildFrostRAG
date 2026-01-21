"""
Centralized configuration management for WildFrostRAG.

This module uses Pydantic Settings to manage all configuration values,
eliminating magic strings and providing type safety.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    All settings have sensible defaults where possible. Sensitive values
    (passwords, API keys) must be provided via environment variables.
    """

    # ===== Neo4j Configuration =====
    neo4j_uri: SecretStr
    neo4j_username: str
    neo4j_password: SecretStr

    # ===== Embedding Configuration =====
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    vector_index_name: str = "document-embeddings"
    similarity_function: str = "cosine"

    # Multi-embedder support: maps embedder name -> configuration
    embedding_configs: Dict[str, Dict[str, Any]] = {
        "hf": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "property_name": "hf_embedding",
            "index_name": "document-embeddings-hf"
        },
        "openai": {
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "property_name": "openai_embedding",
            "index_name": "document-embeddings-openai"
        },
        "gemma": {
            "model": "embeddinggemma",
            "dimension": 768,
            "property_name": "gemma_embedding",
            "index_name": "document-embeddings-gemma"
        }
    }

    # ===== Retrieval Configuration =====
    fulltext_index_name: str = "document-fulltext"
    bm25_index_name: str = "Document"
    rrf_k1: int = 60  # Smoothing parameter for Reciprocal Rank Fusion

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
    outputs_dir: Path = project_root / "outputs"

    # Data subdirectories
    structured_outputs_dir: Path = data_dir / "structured_outputs"
    raw_htmls_dir: Path = data_dir / "raw_htmls"
    schemas_dir: Path = data_dir / "schemas"


    # ===== OpenAI Configuration (for evaluation) =====
    openai_api_key: Optional[SecretStr] = None
    openai_model_name: str = "gpt-4.1-nano"     # Default model for generation
    openai_temperature: float = 0.0             # For deterministic responses
    openai_seed: int = 42                       # Random seed for reproducibility

    # ===== Generation Pipeline Configuration =====
    default_k: int = 5  # Default number of chunks to retrieve
    default_batch_size: int = 25  # Default batch size for processing

    model_config = SettingsConfigDict(
        env_file=".env",
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
            self.outputs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_retrieval_output_dir(self, run_num: int, retriever_type: str, experiment_id: str) -> Path:
        """
        Get path for retrieval experiment output directory.

        Args:
            run_num: Run number
            retriever_type: Type of retriever (bm25, vector, text2cypher, etc.)
            experiment_id: Experiment ID (e.g., "001")

        Returns:
            Path to retrieval experiment directory
            (e.g., outputs/run_1/retrievals/bm25/001)
        """
        return self.outputs_dir / f"run_{run_num}" / "retrievals" / retriever_type / experiment_id

    def get_generation_output_dir(self, run_num: int, experiment_id: str) -> Path:
        """
        Get path for generation experiment output directory.

        Args:
            run_num: Run number
            experiment_id: Experiment ID (e.g., "001")

        Returns:
            Path to generation experiment directory
            (e.g., outputs/run_1/generation/001)
        """
        return self.outputs_dir / f"run_{run_num}" / "generation" / experiment_id


# Global settings instance
# Import this in other modules: from src.utils.config import settings
settings = Settings()
