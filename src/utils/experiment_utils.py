"""Experiment tracking utilities for metadata-driven experiment management.

This module provides utilities for creating, managing, and querying experiments
in a metadata-driven approach inspired by MLflow and Weights & Biases.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from models.experiment_config import (
    EmbeddingConfig,
    GenerationConfig,
    PromptVersions,
    QueryStats,
    RetrievalConfig,
    Text2CypherConfig,
)
from utils.config import get_settings
from utils.logger import logger


def get_next_experiment_id(base_path: Path) -> str:
    """Generate next sequential experiment ID with zero-padding.

    Args:
        base_path: Directory containing existing experiments

    Returns:
        Next experiment ID (e.g., "001", "002", "003")
    """
    if not base_path.exists():
        return "001"

    existing_ids = [int(d.name) for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]

    if not existing_ids:
        return "001"

    next_id = max(existing_ids) + 1
    return f"{next_id:03d}"  # Zero-pad to 3 digits


def create_retrieval_config(
    run_num: int,
    retriever_type: str,
    experiment_id: str,
    chunking: bool,
    total_queries: int,
    successful_queries: int,
    failed_queries: int = 0,
    description: str = "",
    k: int = 10,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    vector_index_name: str | None = None,
    text2cypher_prompt_version: str | None = None,
    text2cypher_llm_model: str | None = None,
    text2cypher_temperature: float | None = None,
    text2cypher_seed: int | None = None,
    notes: str | None = None,
    **extra_metadata: str | int | float | bool | None,
) -> RetrievalConfig:
    """Create config.json for retrieval experiment.

    Args:
        run_num: Run number
        retriever_type: Type of retriever (bm25, vector, text2cypher, etc.)
        experiment_id: Experiment ID (e.g., "001")
        chunking: Whether chunking was used
        total_queries: Total number of queries processed
        successful_queries: Number of successful queries
        failed_queries: Number of failed queries
        description: Human-readable description
        k: Number of chunks retrieved per query
        embedding_model: Embedding model name (vector-based retrievers only)
        embedding_provider: Embedding provider name (vector-based retrievers only)
        vector_index_name: Vector index name (vector-based retrievers only)
        text2cypher_prompt_version: Text2Cypher prompt version (text2cypher only)
        text2cypher_llm_model: Text2Cypher LLM model (text2cypher only)
        text2cypher_temperature: Text2Cypher LLM temperature (text2cypher only)
        text2cypher_seed: Text2Cypher LLM seed (text2cypher only)
        notes: Free-form notes (text2cypher only)
        **extra_metadata: Anything else worth recording (e.g. sw_query, sw_docs)

    Returns:
        Typed retrieval config ready to save
    """
    settings = get_settings()

    text2cypher = None
    if retriever_type == "text2cypher":
        text2cypher = Text2CypherConfig(
            prompt_version=text2cypher_prompt_version or "V1",
            llm_model=text2cypher_llm_model or settings.openai.text2cypher_model,
            temperature=(
                text2cypher_temperature
                if text2cypher_temperature is not None
                else settings.openai.text2cypher_temperature
            ),
            seed=text2cypher_seed if text2cypher_seed is not None else settings.openai.seed,
            notes=notes,
        )

    return RetrievalConfig(
        retrieval_id=f"{retriever_type}/{experiment_id}",
        run_number=run_num,
        timestamp=datetime.now().isoformat(),
        retriever_type=retriever_type,
        chunking=chunking,
        k=k,
        description=description,
        query_stats=QueryStats(
            total=total_queries, successful=successful_queries, failed=failed_queries
        ),
        embedding=EmbeddingConfig(
            model=embedding_model or settings.embedding.model_name,
            provider=embedding_provider,
            vector_index_name=vector_index_name,
        ),
        text2cypher=text2cypher,
        additional_metadata=dict(extra_metadata),
    )


def create_generation_config(
    run_num: int,
    generation_id: str,
    retrieval_reference: str,
    system_prompt_version: str,
    rag_prompt_version: str | None,
    total_queries: int,
    successful_queries: int,
    failed_queries: int = 0,
    description: str = "",
    is_zero_shot: bool = False,
    llm_model: str | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    **extra_metadata: str | int | float | bool | None,
) -> GenerationConfig:
    """Create config.json for generation experiment.

    Args:
        run_num: Run number
        generation_id: Generation experiment ID (e.g., "gen/001")
        retrieval_reference: Reference to retrieval experiment (e.g., "bm25/001")
        system_prompt_version: System prompt version
        rag_prompt_version: RAG prompt version
        total_queries: Total number of queries processed
        successful_queries: Number of successful queries
        failed_queries: Number of failed queries
        description: Human-readable description
        is_zero_shot: Whether this generation used no retrieval context
        llm_model: LLM model name (defaults to settings.openai.model_name)
        temperature: LLM temperature
        seed: LLM seed
        **extra_metadata: Anything else worth recording (e.g. batch_size)

    Returns:
        Typed generation config ready to save
    """
    return GenerationConfig(
        generation_id=generation_id,
        run_number=run_num,
        timestamp=datetime.now().isoformat(),
        retrieval_reference=retrieval_reference,
        llm_model=llm_model or get_settings().openai.model_name,
        temperature=temperature,
        seed=seed,
        prompts=PromptVersions(
            system_prompt_version=system_prompt_version,
            rag_prompt_version=rag_prompt_version,
        ),
        is_zero_shot=is_zero_shot,
        description=description,
        query_stats=QueryStats(
            total=total_queries, successful=successful_queries, failed=failed_queries
        ),
        additional_metadata=dict(extra_metadata),
    )


def save_config(config: RetrievalConfig | GenerationConfig, output_dir: Path) -> None:
    """Save config.json to experiment directory.

    Args:
        config: Typed experiment config
        output_dir: Directory to save config in
    """
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved config to {config_path}")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load config.json from experiment directory.

    Args:
        config_path: Path to config.json file

    Returns:
        Config dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        logger.error(f"Expected dict in {config_path}, got {type(data)}")
        raise TypeError(f"Expected dict in {config_path}, got {type(data)}")
    return data


def validate_retrieval_reference(run_num: int, retrieval_reference: str) -> bool:
    """Validate that retrieval reference exists.

    Args:
        run_num: Run number
        retrieval_reference: Reference like "bm25/001"

    Returns:
        True if reference exists, False otherwise
    """
    retrieval_path = (
        get_settings().paths.outputs_dir / f"run_{run_num}" / "retrievals" / retrieval_reference
    )
    return retrieval_path.exists() and (retrieval_path / "config.json").exists()


def list_available_retrievals(run_num: int) -> list[str]:
    """List all available retrieval experiments for a run.

    Args:
        run_num: Run number

    Returns:
        List of retrieval references (e.g., ["bm25/001", "vector/001"])
    """
    retrievals_dir = get_settings().paths.outputs_dir / f"run_{run_num}" / "retrievals"

    if not retrievals_dir.exists():
        return []

    available = []
    for retriever_type_dir in retrievals_dir.iterdir():
        if not retriever_type_dir.is_dir():
            continue

        for exp_dir in retriever_type_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / "config.json").exists():
                retrieval_ref = f"{retriever_type_dir.name}/{exp_dir.name}"
                available.append(retrieval_ref)

    return sorted(available)


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path to save results
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} results to {output_path}")


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load results from JSON file.

    Args:
        results_path: Path to results file

    Returns:
        List of result dictionaries
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.error(f"Expected list in {results_path}, got {type(data)}")
        raise TypeError(f"Expected list in {results_path}, got {type(data)}")
    return data


def save_cypher_queries(
    queries: list[dict[str, Any]],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save generated Cypher queries to JSON file (for text2cypher).

    Args:
        queries: List of query dictionaries with generated Cypher
        output_path: Path to save queries
        metadata: Optional metadata about the queries
    """
    output_data = {"metadata": metadata or {}, "queries": queries}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(queries)} Cypher queries to {output_path}")


def load_cypher_queries(queries_path: Path) -> dict[str, Any]:
    """Load Cypher queries from JSON file.

    Args:
        queries_path: Path to cypher_queries.json

    Returns:
        Dictionary with metadata and queries
    """
    if not queries_path.exists():
        raise FileNotFoundError(f"Cypher queries file not found: {queries_path}")

    with open(queries_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        logger.error(f"Expected dict in {queries_path}, got {type(data)}")
        raise TypeError(f"Expected dict in {queries_path}, got {type(data)}")
    return data


def save_individual_results(
    individual_results_per_query: list[dict[str, Any]],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save individual retriever results to JSON file (for hybrid retrievers).

    Args:
        individual_results_per_query: List of dicts with query_id and individual results
        output_path: Path to save individual results
        metadata: Optional metadata about the retrievers
    """
    output_data = {"metadata": metadata or {}, "queries": individual_results_per_query}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Saved individual results for {len(individual_results_per_query)} queries to {output_path}"
    )
