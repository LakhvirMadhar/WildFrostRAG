#!/usr/bin/env python3
"""
Retriever Pipeline for WildFrostRAG.

This script runs different retrieval strategies and saves raw results for manual evaluation:
1. Loads query data from CSV
2. Tests various retrieval methods (vector, fulltext, BM25, hybrid combinations, text2cypher)
3. Saves raw retrieval results to structured output directories with run numbers
4. Results can be manually evaluated later using a GUI

Usage:
    python -m scripts.evaluate_retrievers --run-num 1 --retriever vector --chunking yes
    python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25 --chunking no
    python -m scripts.evaluate_retrievers --run-num 1 --retriever text2cypher --chunking no
"""

import asyncio
import argparse
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from neo4j import GraphDatabase, Driver

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.retrievers import (
    Neo4jVectorSearch,
    Neo4jFullTextSearch,
    BM25Retriever,
    BM25VectorHybridRetriever,
    FulltextVectorHybridRetriever,
    BM25FulltextVectorHybridRetriever,
    Text2CypherRetriever,
    Text2CypherVectorHybridRetriever,
    VectorThenCypherRetriever,
)
from src.rag.retrievers.hybrid_retrievers import HybridRetriever
from src.embeddings.query_embedders import get_query_embed_fn
from src.utils.logger import logger
from src.utils.config import settings
from src.utils.experiment_utils import (
    get_next_experiment_id,
    create_retrieval_config,
    save_config,
    save_results,
    save_cypher_queries,
    save_individual_results
)
from src.experiment_tracker import ExperimentRegistry
from src.gui.auto_annotator import run_auto_annotation


# Retriever types that use vector embeddings
VECTOR_BASED_RETRIEVERS = ['vector', 'bm25_vector', 'fulltext_vector', 'bm25_fulltext_vector', 'vector_then_cypher', 'text2cypher_vector']


def get_retriever(retriever_type: str, driver: Driver, embedder: str = "hf", **kwargs) -> Any:
    """
    Factory function to create the appropriate retriever based on type.

    Args:
        retriever_type: Type of retriever to create
        driver: Neo4j driver instance
        embedder: Embedding provider name (for vector-based retrievers)
        **kwargs: Additional arguments passed to the retriever constructor

    Returns:
        An instance of the specified retriever
    """
    index_name = _get_vector_index_name(retriever_type, embedder)

    # Build embed_fn only for vector-based retrievers
    embed_fn = get_query_embed_fn(embedder) if retriever_type in VECTOR_BASED_RETRIEVERS else None

    retriever_factory = {
        'vector': lambda: Neo4jVectorSearch(driver, embed_fn, index_name=index_name),
        'fulltext': lambda: Neo4jFullTextSearch(driver),
        'bm25': lambda: BM25Retriever(driver),
        'bm25_vector': lambda: BM25VectorHybridRetriever(driver, embed_fn, index_name=index_name),
        'fulltext_vector': lambda: FulltextVectorHybridRetriever(driver, embed_fn, index_name=index_name),
        'bm25_fulltext_vector': lambda: BM25FulltextVectorHybridRetriever(driver, embed_fn, index_name=index_name),
        'text2cypher': lambda: Text2CypherRetriever(driver, **kwargs),
        'text2cypher_vector': lambda: Text2CypherVectorHybridRetriever(driver, embed_fn, index_name=index_name, **kwargs),
        'vector_then_cypher': lambda: VectorThenCypherRetriever(driver, embed_fn, index_name=index_name, **kwargs),
    }

    if retriever_type not in retriever_factory:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    return retriever_factory[retriever_type]()


def _get_vector_index_name(retriever_type: str, embedder: str) -> str | None:
    """Get vector index name for vector-based retrievers."""
    if retriever_type not in VECTOR_BASED_RETRIEVERS:
        return None

    if embedder not in settings.embedding_configs:
        raise ValueError(f"Unknown embedder: {embedder}. Available: {list(settings.embedding_configs.keys())}")

    embedder_config = settings.embedding_configs[embedder]
    index_name = embedder_config['index_name']
    logger.info(f"Using embedder '{embedder}' with index '{index_name}'")
    return index_name


def _clean_chunks(chunks: list[dict]) -> list[dict]:
    """Remove embedding arrays from chunks (they bloat files and aren't needed for evaluation)."""
    return [
        {k: v for k, v in chunk.items() if not k.endswith('_embedding') and k != 'embedding'}
        for chunk in chunks
    ]


async def _process_single_query(
    retriever: Any,
    retriever_type: str,
    query: str,
    query_id: int,
    k: int
) -> tuple[dict, dict | None, dict | None]:
    """
    Process a single query and return results.

    Returns:
        Tuple of (result_entry, cypher_query_entry, individual_results_entry)
    """
    # Handle both sync and async retrievers
    result = retriever.search(query, k=k)
    if asyncio.iscoroutine(result):
        retrieved_chunks = await result
    else:
        retrieved_chunks = result
    cleaned_chunks = _clean_chunks(retrieved_chunks)

    result_entry = {
        'query_id': query_id,
        'query': query,
        'retrieved_chunks': cleaned_chunks,
        'relevance_annotations': []
    }

    # Capture text2cypher LLM response
    cypher_entry = None
    if retriever_type == "text2cypher" and hasattr(retriever, 'llm_response'):
        cypher_entry = {
            "query_id": query_id,
            "query": query,
            "llm_response": retriever.llm_response,
            "execution_status": "success" if retrieved_chunks else "failed",
            "execution_time_ms": None,
            "error_message": None
        }

    # Capture hybrid retriever individual results
    individual_entry = None
    if isinstance(retriever, HybridRetriever) and hasattr(retriever, 'last_individual_results'):
        individual_entry = {
            "query_id": query_id,
            "query": query,
            "individual_results": retriever.last_individual_results
        }

    return result_entry, cypher_entry, individual_entry


async def _process_all_queries(
    df: pd.DataFrame,
    retriever: Any,
    retriever_type: str,
    k: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Process all queries in the dataframe through the retriever.

    Returns:
        Tuple of (results, cypher_queries, individual_results)
    """
    results = []
    cypher_queries = []
    individual_results_list = []

    for idx, row in df.iterrows():
        query = row['query']
        if pd.isna(query) or query == '':
            continue

        logger.info(f"Processing query {idx + 1}/{len(df)}: '{query}'")
        query_id = row.get('query_id', idx)

        result, cypher_entry, individual_entry = await _process_single_query(
            retriever, retriever_type, query, query_id, k
        )
        results.append(result)

        if cypher_entry:
            cypher_queries.append(cypher_entry)
        if individual_entry:
            individual_results_list.append(individual_entry)

    return results, cypher_queries, individual_results_list


def _get_embedder_config_kwargs(retriever_type: str, embedder: str) -> dict:
    """Get embedder configuration for config file (only for vector-based retrievers)."""
    if retriever_type not in VECTOR_BASED_RETRIEVERS:
        return {}

    embedder_cfg = settings.embedding_configs[embedder]
    return {
        "embedding_provider": embedder,
        "embedding_model": embedder_cfg['model'],
        "vector_index_name": embedder_cfg['index_name']
    }


def _save_experiment_artifacts(
    experiment_dir: Path,
    config: dict,
    results: list[dict],
    cypher_queries: list[dict],
    individual_results: list[dict],
    retriever: Any,
    retriever_type: str,
    experiment_id: str,
    run_num: int,
    kwargs: dict
) -> None:
    """Save all experiment artifacts to disk."""
    # Save config
    save_config(config, experiment_dir)

    # Register in experiment registry
    registry = ExperimentRegistry()
    registry.register_retrieval(run_num, retriever_type, experiment_id, config)

    # Save results
    save_results(results, experiment_dir / "results.json")

    # Save text2cypher queries if applicable
    if retriever_type == "text2cypher" and cypher_queries:
        metadata = {
            "retrieval_id": f"{retriever_type}/{experiment_id}",
            "text2cypher_prompt_version": kwargs.get("text2cypher_prompt_version", "V1"),
            "timestamp": datetime.now().isoformat()
        }
        save_cypher_queries(cypher_queries, experiment_dir / "cypher_queries.json", metadata=metadata)

    # Save hybrid individual results if applicable
    if isinstance(retriever, HybridRetriever) and individual_results:
        metadata = {
            "retrieval_id": f"{retriever_type}/{experiment_id}",
            "retriever_names": retriever.retriever_names,
            "timestamp": datetime.now().isoformat()
        }
        save_individual_results(individual_results, experiment_dir / "individual_results.json", metadata=metadata)


async def run_retriever(
    df: pd.DataFrame,
    retriever: Any,
    retriever_type: str,
    run_num: int,
    chunking: bool,
    k: int = 10,
    description: str = "",
    embedder: str = "hf",
    queries_json_path: Path | None = None,
    **kwargs
) -> list[dict]:
    """
    Run a specific retriever on the provided dataset and save raw results.

    Args:
        df: DataFrame containing queries
        retriever: Retriever instance to run
        retriever_type: Type of retriever being used
        run_num: Experiment run number
        chunking: Whether chunking was used (boolean)
        k: Number of chunks to retrieve per query
        description: Human-readable description of this experiment
        embedder: Embedding provider (for vector-based retrievers)
        **kwargs: Additional metadata (e.g., text2cypher_prompt_version)

    Returns:
        List of retrieval results
    """
    # Setup experiment directory
    retriever_dir_name = retriever_type
    if retriever_type in VECTOR_BASED_RETRIEVERS:
        retriever_dir_name = f"{retriever_type}_{embedder}"

    base_path = settings.outputs_dir / f"run_{run_num}" / "retrievals" / retriever_dir_name
    experiment_id = get_next_experiment_id(base_path)
    experiment_dir = base_path / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running {retriever_type} retriever in {experiment_dir}")
    logger.info(f"Experiment ID: {retriever_type}/{experiment_id}")

    # Process queries
    results, cypher_queries, individual_results_list = await _process_all_queries(
        df, retriever, retriever_type, k
    )

    # Build and save config
    total_queries = len([r for _, r in df.iterrows() if not pd.isna(r.get('query', '')) and r.get('query', '') != ''])
    embedder_kwargs = _get_embedder_config_kwargs(retriever_type, embedder)

    config = create_retrieval_config(
        run_num=run_num,
        retriever_type=retriever_type,
        experiment_id=experiment_id,
        chunking=chunking,
        total_queries=total_queries,
        successful_queries=len(results),
        failed_queries=total_queries - len(results),
        description=description,
        k=k,
        **embedder_kwargs,
        **kwargs
    )

    _save_experiment_artifacts(
        experiment_dir, config, results, cypher_queries, individual_results_list,
        retriever, retriever_type, experiment_id, run_num, kwargs
    )

    # Auto-annotate relevance based on URL matching with ground truth
    annotation_summary = run_auto_annotation(experiment_dir, queries_json_path)
    logger.info(f"Auto-annotation: {annotation_summary['auto_annotated']} chunks annotated")

    logger.info("Experiment completed successfully!")
    logger.info(f"Retrieval ID: {retriever_type}/{experiment_id}")
    logger.info(f"Results saved to {experiment_dir}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run different retrievers and save raw results")
    parser.add_argument("--run-num", type=int, required=True, help="Experiment run number")
    parser.add_argument("--retriever", type=str,
                        choices=["vector", "fulltext", "bm25", "bm25_vector", "fulltext_vector",
                                 "bm25_fulltext_vector", "text2cypher", "text2cypher_vector", "vector_then_cypher"],
                        required=True, help="Retriever to run")
    parser.add_argument("--chunking", type=str, choices=["yes", "no"], default="no",
                        help="Whether chunking was used during ingestion")
    parser.add_argument("--description", type=str, default="",
                        help="Human-readable description of this experiment")
    parser.add_argument("--text2cypher-prompt", type=str, default="TEXT2CYPHER_PROMPT_V1",
                        help="Text2cypher prompt name (e.g., TEXT2CYPHER_PROMPT_V1)")
    parser.add_argument("--query-ids", type=str,
                        help="Comma-separated query IDs to include (e.g., '1,5,10')")
    parser.add_argument("--exclude-query-ids", type=str,
                        help="Comma-separated query IDs to exclude (e.g., '2,3,4')")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of chunks to retrieve per query (default: 10)")
    parser.add_argument("--file", type=str,
                        default="queries/simple_reference_based_queries.csv",
                        help="Path to input CSV file with queries")
    parser.add_argument("--embedder", type=str, default="hf",
                        help="Embedding provider (e.g., 'hf', 'openai')")
    parser.add_argument("--queries-json", type=str, default=None,
                        help="Path to queries JSON with doc_references for auto-annotation")
    return parser.parse_args()


def load_and_filter_queries(file_path: str, query_ids: str | None, exclude_ids: str | None) -> pd.DataFrame:
    """Load queries from CSV and apply filters."""
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found.")
        sys.exit(1)

    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows.")

    if query_ids:
        ids = [int(qid.strip()) for qid in query_ids.split(',')]
        df = df[df['query_id'].isin(ids)]
        logger.info(f"Filtered to {len(df)} queries with IDs: {ids}")

    if exclude_ids:
        ids = [int(qid.strip()) for qid in exclude_ids.split(',')]
        df = df[~df['query_id'].isin(ids)]
        logger.info(f"Excluded {len(ids)} queries. Remaining: {len(df)} queries")

    if len(df) == 0:
        logger.error("No queries to process after filtering!")
        sys.exit(1)

    return df


def load_text2cypher_prompt(prompt_name: str):
    """Load text2cypher prompt from prompts module."""
    try:
        prompts_module = importlib.import_module("prompts.text2cypher_prompts")
        return getattr(prompts_module, prompt_name)
    except AttributeError:
        logger.error(f"Prompt '{prompt_name}' not found in prompts.text2cypher_prompts")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Failed to import prompts.text2cypher_prompts: {e}")
        sys.exit(1)


async def main():
    args = parse_args()
    settings.create_directories()

    df = load_and_filter_queries(args.file, args.query_ids, args.exclude_query_ids)
    chunking = args.chunking == "yes"

    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
    )

    try:
        retriever_kwargs = {}
        config_kwargs = {}

        if args.retriever in ("text2cypher", "text2cypher_vector"):
            prompt = load_text2cypher_prompt(args.text2cypher_prompt)
            retriever_kwargs["text2cypher_prompt"] = prompt
            config_kwargs["text2cypher_prompt_version"] = prompt.prompt_version_name

        retriever = get_retriever(args.retriever, driver, embedder=args.embedder, **retriever_kwargs)
        logger.info(f"Using {args.retriever} retriever")

        queries_json = Path(args.queries_json) if args.queries_json else None

        results = await run_retriever(
            df=df,
            retriever=retriever,
            retriever_type=args.retriever,
            run_num=args.run_num,
            chunking=chunking,
            k=args.k,
            description=args.description,
            embedder=args.embedder,
            queries_json_path=queries_json,
            **config_kwargs
        )

        if results:
            logger.info("Retriever run completed successfully! Results saved for manual evaluation.")
        else:
            logger.error("Retriever run failed.")

    finally:
        driver.close()
        logger.info("Neo4j driver closed")


if __name__ == "__main__":
    asyncio.run(main())
