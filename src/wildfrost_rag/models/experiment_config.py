"""Typed schemas for experiment config.json files.

These models define what create_retrieval_config()/create_generation_config()
build and save_config() writes to config.json - the metadata saved alongside
each experiment's results. Composed from smaller sub-models rather than one
flat class, matching the Settings decomposition (see utils/config.py).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryStats(BaseModel):
    """Query counts for an experiment run."""

    total: int
    successful: int
    failed: int = 0


class EmbeddingConfig(BaseModel):
    """Embedding provider details for a vector-based retrieval experiment."""

    model: str
    provider: str | None = None
    vector_index_name: str | None = None


class Text2CypherConfig(BaseModel):
    """Text2Cypher-specific LLM settings for a retrieval experiment."""

    prompt_version: str = "V1"
    llm_model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    notes: str | None = None


class RetrievalConfig(BaseModel):
    """config.json schema for a retrieval experiment."""

    experiment_type: Literal["retrieval"] = "retrieval"
    retrieval_id: str
    run_number: int
    timestamp: str
    retriever_type: str
    chunking: bool
    k: int = 10
    description: str = ""
    dataset: str = "simple_reference_based_queries.csv"
    query_stats: QueryStats
    embedding: EmbeddingConfig
    text2cypher: Text2CypherConfig | None = None
    additional_metadata: dict[str, Any] = Field(default_factory=dict)


class PromptVersions(BaseModel):
    """Prompt versions used by a generation experiment."""

    system_prompt_version: str
    rag_prompt_version: str | None = None


class GenerationConfig(BaseModel):
    """config.json schema for a generation experiment."""

    experiment_type: Literal["generation"] = "generation"
    generation_id: str
    run_number: int
    timestamp: str
    retrieval_reference: str
    llm_model: str
    temperature: float = 0.0
    seed: int = 42
    prompts: PromptVersions
    is_zero_shot: bool = False
    description: str = ""
    dataset: str = "simple_reference_based_queries.csv"
    query_stats: QueryStats
    additional_metadata: dict[str, Any] = Field(default_factory=dict)
