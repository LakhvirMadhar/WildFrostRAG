#!/usr/bin/env python3
"""Retriever Pipeline for WildFrostRAG.

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
import inspect
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Callable

from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

import pandas as pd
from neo4j import GraphDatabase, Driver

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from prompts import get_prompt
from prompts.prompt_utils import VersionedPrompt
from rag.retrievers import (
    Neo4jVectorSearch,
    Neo4jFullTextSearch,
    BM25Retriever,
    BM25VectorHybridRetriever,
    FulltextVectorHybridRetriever,
    BM25FulltextVectorHybridRetriever,
    Text2CypherRetriever,
    Text2CypherVectorHybridRetriever,
    VectorThenCypherRetriever,
    FulltextThenCypherRetriever,
)
from rag.retrievers.hybrid_retrievers import HybridRetriever
from core.exceptions import CypherExecutionError
from embeddings.query_embedders import get_query_embed_fn
from models.experiment_config import RetrievalConfig
from models.retrieval import QueryResult, CypherExecution
from utils.logger import logger
from utils.config import get_settings
from utils.experiment_utils import (
    get_next_experiment_id,
    create_retrieval_config,
    save_config,
    save_results,
    save_individual_results,
)
from experiment_tracker import ExperimentRegistry
from gui.auto_annotator import run_auto_annotation

# Retriever types that support stop word removal
SW_QUERY_RETRIEVERS = {
    "bm25",
    "fulltext",
    "bm25_vector",
    "fulltext_vector",
    "bm25_fulltext_vector",
    "fulltext_then_cypher",
}
SW_DOCS_RETRIEVERS = {"bm25", "bm25_vector", "bm25_fulltext_vector"}


# Retriever types that use vector embeddings
VECTOR_BASED_RETRIEVERS = [
    "vector",
    "bm25_vector",
    "fulltext_vector",
    "bm25_fulltext_vector",
    "vector_then_cypher",
    "text2cypher_vector",
]


def get_retriever(
    retriever_type: str,
    driver: Driver,
    embedder: str = "hf",
    sw_query: bool = True,
    sw_docs: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Factory function to create the appropriate retriever based on type.

    Args:
        retriever_type: Type of retriever to create
        driver: Neo4j driver instance
        embedder: Embedding provider name (for vector-based retrievers)
        sw_query: Remove stop words from queries (BM25 + fulltext)
        sw_docs: Remove stop words from documents (BM25 only)
        **kwargs: Additional arguments passed to the retriever constructor

    Returns:
        An instance of the specified retriever
    """
    index_name = _get_vector_index_name(retriever_type, embedder)

    # Build embed_fn only for vector-based retrievers
    embed_fn = get_query_embed_fn(embedder) if retriever_type in VECTOR_BASED_RETRIEVERS else None

    # Non-vector retrievers (embed_fn not needed)
    non_vector_factory: dict[str, Callable[[], Any]] = {
        "fulltext": lambda: Neo4jFullTextSearch(driver, remove_stopwords=sw_query),
        "bm25": lambda: BM25Retriever(
            driver, remove_stopwords_query=sw_query, remove_stopwords_docs=sw_docs
        ),
        "text2cypher": lambda: Text2CypherRetriever(driver, **kwargs),
        "fulltext_then_cypher": lambda: FulltextThenCypherRetriever(
            driver, remove_stopwords=sw_query
        ),
    }

    if retriever_type in non_vector_factory:
        return non_vector_factory[retriever_type]()

    # Vector-based retrievers (embed_fn is guaranteed non-None here)
    if embed_fn is None:
        raise ValueError(f"embed_fn is required for vector-based retriever: {retriever_type}")

    vector_factory: dict[str, Callable[[], Any]] = {
        "vector": lambda: Neo4jVectorSearch(driver, embed_fn, index_name=index_name),
        "bm25_vector": lambda: BM25VectorHybridRetriever(driver, embed_fn, index_name=index_name),
        "fulltext_vector": lambda: FulltextVectorHybridRetriever(
            driver, embed_fn, index_name=index_name, remove_stopwords=sw_query
        ),
        "bm25_fulltext_vector": lambda: BM25FulltextVectorHybridRetriever(
            driver, embed_fn, index_name=index_name
        ),
        "text2cypher_vector": lambda: Text2CypherVectorHybridRetriever(
            driver, embed_fn, index_name=index_name, **kwargs
        ),
        "vector_then_cypher": lambda: VectorThenCypherRetriever(
            driver, embed_fn, index_name=index_name, **kwargs
        ),
    }

    if retriever_type not in vector_factory:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    return vector_factory[retriever_type]()


def _get_vector_index_name(retriever_type: str, embedder: str) -> str | None:
    """Get vector index name for vector-based retrievers."""
    if retriever_type not in VECTOR_BASED_RETRIEVERS:
        return None

    settings = get_settings()
    if embedder not in settings.embedding.embedding_configs:
        raise ValueError(
            f"Unknown embedder: {embedder}. Available: {list(settings.embedding.embedding_configs.keys())}"
        )

    embedder_config = settings.embedding.embedding_configs[embedder]
    index_name: str = embedder_config["index_name"]
    logger.info(f"Using embedder '{embedder}' with index '{index_name}'")
    return index_name


async def _process_single_query(
    retriever: Any,  # noqa: ANN401
    query: str,
    query_id: int,
    k: int,
) -> tuple[QueryResult, dict[str, Any] | None]:
    """Process a single query and return typed results.

    Returns:
        Tuple of (QueryResult, individual_results_entry or None)
    """
    # Handle both sync and async retrievers
    try:
        result = retriever.search(query, k=k)
        if asyncio.iscoroutine(result):
            retrieved_chunks = await result
        else:
            retrieved_chunks = result
    except CypherExecutionError as e:
        # A single bad generated Cypher query shouldn't kill the whole batch -
        # record it as a failed query and let the run continue.
        failed_result = QueryResult(
            query_id=query_id,
            query=query,
            cypher_execution=CypherExecution(
                cypher_query=e.cypher_query,
                cypher_execution_status="failed",
                cypher_error_message=e.reason,
            ),
            retrieved_chunks=[],
            relevance_annotations=[],
        )
        return failed_result, None

    # Retrievers return typed RetrievedChunk objects and record the Cypher query
    # they ran (if any) on last_cypher_query - None for BM25/hybrid retrievers.
    cypher_query = getattr(retriever, "last_cypher_query", None)
    cypher_execution = CypherExecution(
        cypher_query=cypher_query,
        cypher_execution_status="success",
        cypher_error_message=None,
    )

    query_result = QueryResult(
        query_id=query_id,
        query=query,
        cypher_execution=cypher_execution,
        retrieved_chunks=retrieved_chunks,
        relevance_annotations=[],
    )

    # Capture hybrid retriever individual results
    individual_entry = None
    if isinstance(retriever, HybridRetriever) and retriever.last_individual_results is not None:
        individual_entry = {
            "query_id": query_id,
            "query": query,
            "individual_results": {
                name: [chunk.to_dict() for chunk in chunks]
                for name, chunks in retriever.last_individual_results.items()
            },
        }

    return query_result, individual_entry


async def _process_all_queries(
    df: pd.DataFrame,
    retriever: Any,  # noqa: ANN401
    retriever_type: str,
    k: int,
) -> tuple[list[QueryResult], list[dict[str, Any]]]:
    """Process all queries in the dataframe through the retriever.

    Async retrievers (e.g., text2cypher) run concurrently via asyncio.gather().
    Sync retrievers run sequentially.

    Returns:
        Tuple of (results, individual_results)
    """
    # Build query list, filtering empty rows
    query_rows = [
        (row.get("query_id", idx), row["query"])
        for idx, row in df.iterrows()
        if not pd.isna(row["query"]) and row["query"] != ""
    ]

    # Check if retriever is async by testing a dummy call
    is_async = inspect.iscoroutinefunction(getattr(retriever, "search", None))

    if is_async:
        logger.info(f"Running {len(query_rows)} queries concurrently")
        tasks = [
            _process_single_query(retriever, query, query_id, k) for query_id, query in query_rows
        ]
        all_results = await tqdm_asyncio.gather(
            *tasks, desc=f"{retriever_type} queries", unit="query"
        )
    else:
        logger.info(f"Running {len(query_rows)} queries sequentially")
        all_results = []
        for query_id, query in tqdm(query_rows, desc=f"{retriever_type} queries", unit="query"):
            result = await _process_single_query(retriever, query, query_id, k)
            all_results.append(result)

    results: list[QueryResult] = []
    individual_results_list: list[dict[str, Any]] = []

    for query_result, individual_entry in all_results:
        results.append(query_result)
        if individual_entry:
            individual_results_list.append(individual_entry)

    return results, individual_results_list


def _get_embedder_config(
    retriever_type: str, embedder: str
) -> tuple[str | None, str | None, str | None]:
    """Get (provider, model, vector_index_name) for config file.

    Returns (None, None, None) for non-vector-based retrievers.
    """
    if retriever_type not in VECTOR_BASED_RETRIEVERS:
        return None, None, None

    embedder_cfg = get_settings().embedding.embedding_configs[embedder]
    return embedder, embedder_cfg["model"], embedder_cfg["index_name"]


def _save_experiment_artifacts(
    experiment_dir: Path,
    config: RetrievalConfig,
    results: list[QueryResult],
    individual_results: list[dict[str, Any]],
    retriever: Any,  # noqa: ANN401
    retriever_type: str,
    experiment_id: str,
    run_num: int,
) -> None:
    """Save all experiment artifacts to disk."""
    # Save config
    save_config(config, experiment_dir)

    # Register in experiment registry
    registry = ExperimentRegistry()
    registry.register_retrieval(run_num, retriever_type, experiment_id, config)

    # Serialize typed objects to dicts for JSON storage
    # cypher_execution is now embedded inside each QueryResult
    results_dicts = [r.to_dict() for r in results]
    save_results(results_dicts, experiment_dir / "results.json")

    # Save hybrid individual results if applicable
    if isinstance(retriever, HybridRetriever) and individual_results:
        metadata = {
            "retrieval_id": f"{retriever_type}/{experiment_id}",
            "retriever_names": retriever.retriever_names,
            "timestamp": datetime.now().isoformat(),
        }
        save_individual_results(
            individual_results,
            experiment_dir / "individual_results.json",
            metadata=metadata,
        )


async def run_retriever(
    df: pd.DataFrame,
    retriever: Any,  # noqa: ANN401
    retriever_type: str,
    run_num: int,
    chunking: bool,
    k: int = 10,
    description: str = "",
    embedder: str = "hf",
    queries_json_path: Path | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> list[QueryResult]:
    """Run a specific retriever on the provided dataset and save raw results.

    Args:
        df: DataFrame containing queries
        retriever: Retriever instance to run
        retriever_type: Type of retriever being used
        run_num: Experiment run number
        chunking: Whether chunking was used (boolean)
        k: Number of chunks to retrieve per query
        description: Human-readable description of this experiment
        embedder: Embedding provider (for vector-based retrievers)
        queries_json_path: Path to queries JSON with doc_references for auto-annotation
        **kwargs: Additional metadata (e.g., text2cypher_prompt_version)

    Returns:
        List of retrieval results
    """
    # Setup experiment directory
    retriever_dir_name = retriever_type
    if retriever_type in VECTOR_BASED_RETRIEVERS:
        retriever_dir_name = f"{retriever_type}_{embedder}"

    base_path = (
        get_settings().paths.outputs_dir / f"run_{run_num}" / "retrievals" / retriever_dir_name
    )
    experiment_id = get_next_experiment_id(base_path)
    experiment_dir = base_path / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running {retriever_type} retriever in {experiment_dir}")
    logger.info(f"Experiment ID: {retriever_type}/{experiment_id}")

    # Process queries
    results, individual_results_list = await _process_all_queries(df, retriever, retriever_type, k)

    # Build and save config
    total_queries = len(
        [
            r
            for _, r in df.iterrows()
            if not pd.isna(r.get("query", "")) and r.get("query", "") != ""
        ]
    )
    embedding_provider, embedding_model, vector_index_name = _get_embedder_config(
        retriever_type, embedder
    )

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
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        vector_index_name=vector_index_name,
        **kwargs,
    )

    _save_experiment_artifacts(
        experiment_dir,
        config,
        results,
        individual_results_list,
        retriever,
        retriever_type,
        experiment_id,
        run_num,
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
    parser.add_argument(
        "--retriever",
        type=str,
        choices=[
            "vector",
            "fulltext",
            "bm25",
            "bm25_vector",
            "fulltext_vector",
            "bm25_fulltext_vector",
            "text2cypher",
            "text2cypher_vector",
            "vector_then_cypher",
            "fulltext_then_cypher",
        ],
        required=True,
        help="Retriever to run",
    )
    parser.add_argument(
        "--chunking",
        type=str,
        choices=["yes", "no"],
        default="no",
        help="Whether chunking was used during ingestion",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Human-readable description of this experiment",
    )
    parser.add_argument(
        "--text2cypher-prompt",
        type=str,
        default="TEXT2CYPHER_PROMPT_V1",
        help="Text2cypher prompt name (e.g., TEXT2CYPHER_PROMPT_V1)",
    )
    parser.add_argument(
        "--query-ids",
        type=str,
        help="Comma-separated query IDs to include (e.g., '1,5,10')",
    )
    parser.add_argument(
        "--exclude-query-ids",
        type=str,
        help="Comma-separated query IDs to exclude (e.g., '2,3,4')",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="queries/simple_reference_based_queries.csv",
        help="Path to input CSV file with queries",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="hf",
        help="Embedding provider (e.g., 'hf', 'openai')",
    )
    parser.add_argument(
        "--queries-json",
        type=str,
        default=None,
        help="Path to queries JSON with doc_references for auto-annotation",
    )
    parser.add_argument(
        "--sw-query",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Remove stop words from queries (BM25 + fulltext)",
    )
    parser.add_argument(
        "--sw-docs",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Remove stop words from documents (BM25 only)",
    )
    return parser.parse_args()


def load_and_filter_queries(
    file_path: str, query_ids: str | None, exclude_ids: str | None
) -> pd.DataFrame:
    """Load queries from CSV and apply filters."""
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} not found.")
        sys.exit(1)

    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows.")

    if query_ids:
        ids = [int(qid.strip()) for qid in query_ids.split(",")]
        df = df[df["query_id"].isin(ids)]
        logger.info(f"Filtered to {len(df)} queries with IDs: {ids}")

    if exclude_ids:
        ids = [int(qid.strip()) for qid in exclude_ids.split(",")]
        df = df[~df["query_id"].isin(ids)]
        logger.info(f"Excluded {len(ids)} queries. Remaining: {len(df)} queries")

    if len(df) == 0:
        logger.error("No queries to process after filtering!")
        sys.exit(1)

    return df


def load_text2cypher_prompt(prompt_name: str) -> VersionedPrompt:
    """Load text2cypher prompt by name from the registry."""
    return get_prompt(prompt_name)


async def run(args: argparse.Namespace) -> None:
    """Run a retrieval experiment from a parsed Namespace."""
    settings = get_settings()
    settings.create_directories()

    df = load_and_filter_queries(args.file, args.query_ids, args.exclude_query_ids)
    chunking = args.chunking == "yes"

    driver = GraphDatabase.driver(
        settings.neo4j.uri.get_secret_value(),
        auth=(settings.neo4j.username, settings.neo4j.password.get_secret_value()),
    )

    try:
        retriever_kwargs = {}
        config_kwargs = {}

        if args.retriever in ("text2cypher", "text2cypher_vector"):
            prompt = load_text2cypher_prompt(args.text2cypher_prompt)
            retriever_kwargs["text2cypher_prompt"] = prompt
            config_kwargs["text2cypher_prompt_version"] = prompt.prompt_version_name

        sw_query = args.sw_query == "yes"
        sw_docs = args.sw_docs == "yes"
        retriever = get_retriever(
            args.retriever,
            driver,
            embedder=args.embedder,
            sw_query=sw_query,
            sw_docs=sw_docs,
            **retriever_kwargs,
        )

        if args.retriever in SW_QUERY_RETRIEVERS:
            config_kwargs["sw_query"] = sw_query
        if args.retriever in SW_DOCS_RETRIEVERS:
            config_kwargs["sw_docs"] = sw_docs
        logger.info(f"Using {args.retriever} retriever (sw_query={sw_query}, sw_docs={sw_docs})")

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
            **config_kwargs,
        )

        if results:
            logger.info(
                "Retriever run completed successfully! Results saved for manual evaluation."
            )
        else:
            logger.error("Retriever run failed.")

    finally:
        driver.close()
        logger.info("Neo4j driver closed")


async def main() -> None:
    """CLI entry point — parse args and run."""
    args = parse_args()
    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
