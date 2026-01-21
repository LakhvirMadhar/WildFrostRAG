#!/usr/bin/env python3
"""
Retriever Pipeline for WildFrostRAG.

This script runs different retrieval strategies and saves raw results for manual evaluation:
1. Loads query data from CSV
2. Tests various retrieval methods (vector, fulltext, BM25, hybrid combinations, text2cypher)
3. Saves raw retrieval results to structured output directories with run numbers
4. Results can be manually evaluated later using a GUI

Usage:
    python -m scripts.evaluate_retrievers --run-num 1 --retriever vector --chunking yes    # Run vector search with chunking
    python -m scripts.evaluate_retrievers --run-num 1 --retriever vector --chunking no     # Run vector search without chunking
    python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext --chunking yes  # Run full-text search
    python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25 --chunking no       # Run BM25 without chunking
    python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25_vector --chunking yes    # Run BM25+Vector hybrid search
    python -m scripts.evaluate_retrievers --run-num 1 --retriever fulltext_vector --chunking yes # Run Fulltext+Vector hybrid search
    python -m scripts.evaluate_retrievers --run-num 1 --retriever bm25_fulltext_vector --chunking yes # Run BM25+Fulltext+Vector hybrid search
    python -m scripts.evaluate_retrievers --run-num 1 --retriever text2cypher --chunking no # Run Text2Cypher
"""

import asyncio
import argparse
import os
import pandas as pd
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.retrievers import (
    Neo4jVectorSearch,
    Neo4jFullTextSearch,
    BM25Retriever,
    BM25VectorHybridRetriever,
    FulltextVectorHybridRetriever,
    BM25FulltextVectorHybridRetriever,
    Text2CypherRetriever
)
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
from src.rag.retrievers.hybrid_retriever import HybridRetriever
from neo4j import GraphDatabase
import importlib
from prompts.text2cypher_prompts import TEXT2CYPHER_PROMPT_V1


def get_retriever(retriever_type: str, driver, embedder: str = "hf", **kwargs):
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
    # List of vector-based retrievers that need embedder config
    vector_based_retrievers = ['vector', 'bm25_vector', 'fulltext_vector', 'bm25_fulltext_vector']

    # Get embedder config for vector-based retrievers
    index_name = None
    if retriever_type in vector_based_retrievers:
        if embedder not in settings.embedding_configs:
            raise ValueError(f"Unknown embedder: {embedder}. Available: {list(settings.embedding_configs.keys())}")

        embedder_config = settings.embedding_configs[embedder]
        index_name = embedder_config['index_name']
        logger.info(f"Using embedder '{embedder}' with index '{index_name}'")

    # Create retrievers
    if retriever_type == 'vector':
        return Neo4jVectorSearch(driver, index_name=index_name)

    elif retriever_type == 'fulltext':
        return Neo4jFullTextSearch(driver)

    elif retriever_type == 'bm25':
        return BM25Retriever(driver)

    elif retriever_type == 'bm25_vector':
        return BM25VectorHybridRetriever(driver, index_name=index_name)

    elif retriever_type == 'fulltext_vector':
        return FulltextVectorHybridRetriever(driver, index_name=index_name)

    elif retriever_type == 'bm25_fulltext_vector':
        return BM25FulltextVectorHybridRetriever(driver, index_name=index_name)

    elif retriever_type == 'text2cypher':
        return Text2CypherRetriever(driver, **kwargs)

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


async def run_retriever(
    df: pd.DataFrame,
    retriever,
    retriever_type: str,
    run_num: int,
    chunking: bool,
    k: int = 10,
    description: str = "",
    embedder: str = "hf",
    **kwargs
):
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
        Dictionary containing raw retrieval results
    """
    # Determine retriever directory name
    vector_based_retrievers = ['vector', 'bm25_vector', 'fulltext_vector', 'bm25_fulltext_vector']
    retriever_dir_name = retriever_type

    # Add embedder suffix for vector-based retrievers
    if retriever_type in vector_based_retrievers:
        retriever_dir_name = f"{retriever_type}_{embedder}"

    # Generate experiment ID
    base_path = settings.outputs_dir / f"run_{run_num}" / "retrievals" / retriever_dir_name
    experiment_id = get_next_experiment_id(base_path)
    experiment_dir = base_path / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running {retriever_type} retriever in {experiment_dir}")
    logger.info(f"Experiment ID: {retriever_type}/{experiment_id}")

    # Process each query in the dataset
    results = []
    cypher_queries = []  # For text2cypher
    individual_results_list = []  # For hybrid retrievers

    # Check if this is a hybrid retriever
    is_hybrid = isinstance(retriever, HybridRetriever)

    for idx, row in df.iterrows():
        query = row['query']

        if pd.isna(query) or query == '':
            continue

        logger.info(f"Processing query {idx+1}/{len(df)}: '{query}'")

        # Retrieve chunks using the retriever
        retrieved_chunks = retriever.search(query, k=k)

        # For text2cypher: capture LLM responses
        if retriever_type == "text2cypher" and hasattr(retriever, 'llm_response'):
            cypher_queries.append({
                "query_id": row.get('query_id', idx),
                "query": query,
                "llm_response": retriever.llm_response,
                "execution_status": "success" if retrieved_chunks else "failed",
                "execution_time_ms": None,  # Could add timing if needed
                "error_message": None
            })

        # For hybrid retrievers: capture individual results
        if is_hybrid and hasattr(retriever, 'last_individual_results'):
            individual_results_list.append({
                "query_id": row.get('query_id', idx),
                "query": query,
                "individual_results": retriever.last_individual_results
            })

        # Store results with all available node properties
        result_entry = {
            'query_id': row.get('query_id', idx),
            'query': query,
            'retrieved_chunks': [
                {
                    key: value for key, value in chunk.items()
                    if key != 'text' or len(str(value)) <= 500  # Limit text length in results
                } for chunk in retrieved_chunks
            ],
            # Initialize relevance annotations as empty - to be filled by manual evaluation
            'relevance_annotations': []
        }

        results.append(result_entry)

    # Count successful and failed queries
    total_queries = len([r for _, r in df.iterrows() if not pd.isna(r.get('query', '')) and r.get('query', '') != ''])
    successful_queries = len(results)
    failed_queries = total_queries - successful_queries

    # Prepare embedder info for config (only for vector-based retrievers)
    embedder_config_kwargs = {}
    if retriever_type in vector_based_retrievers:
        embedder_cfg = settings.embedding_configs[embedder]
        embedder_config_kwargs = {
            "embedding_provider": embedder,
            "embedding_model": embedder_cfg['model'],
            "vector_index_name": embedder_cfg['index_name']
        }

    # Build config with all metadata
    config = create_retrieval_config(
        run_num=run_num,
        retriever_type=retriever_type,
        experiment_id=experiment_id,
        chunking=chunking,
        total_queries=total_queries,
        successful_queries=successful_queries,
        failed_queries=failed_queries,
        description=description,
        k=k,
        **embedder_config_kwargs,
        **kwargs
    )

    # Save config
    save_config(config, experiment_dir)

    # Register in experiment registry
    registry = ExperimentRegistry()
    registry.register_retrieval(run_num, retriever_type, experiment_id, config)

    # Save results
    save_results(results, experiment_dir / "results.json")

    # For text2cypher: Save cypher_queries.json
    if retriever_type == "text2cypher" and cypher_queries:
        metadata = {
            "retrieval_id": f"{retriever_type}/{experiment_id}",
            "text2cypher_prompt_version": kwargs.get("text2cypher_prompt_version", "V1"),
            "timestamp": datetime.now().isoformat()
        }
        save_cypher_queries(cypher_queries, experiment_dir / "cypher_queries.json", metadata=metadata)

    # For hybrid retrievers: Save individual_results.json
    if is_hybrid and individual_results_list:
        metadata = {
            "retrieval_id": f"{retriever_type}/{experiment_id}",
            "retriever_names": retriever.retriever_names,
            "timestamp": datetime.now().isoformat()
        }
        save_individual_results(individual_results_list, experiment_dir / "individual_results.json", metadata=metadata)

    logger.info(f"Experiment completed successfully!")
    logger.info(f"Retrieval ID: {retriever_type}/{experiment_id}")
    logger.info(f"Results saved to {experiment_dir}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Run different retrievers and save raw results")
    parser.add_argument("--run-num", type=int, required=True, help="Experiment run number")
    parser.add_argument("--retriever", type=str,
                       choices=["vector", "fulltext", "bm25", "bm25_vector", "fulltext_vector", "bm25_fulltext_vector", "text2cypher"],
                       required=True, help="Retriever to run")
    parser.add_argument("--chunking", type=str, choices=["yes", "no"], default="no",
                       help="Whether chunking was used during ingestion")
    parser.add_argument("--description", type=str, default="",
                       help="Human-readable description of this experiment")
    parser.add_argument("--text2cypher-prompt", type=str, default="TEXT2CYPHER_PROMPT_V1",
                       help="Text2cypher prompt name (e.g., TEXT2CYPHER_PROMPT_V1, TEXT2CYPHER_PROMPT_V2)")
    parser.add_argument("--query-ids", type=str, help="Comma-separated query IDs to include (e.g., '1,5,10')")
    parser.add_argument("--exclude-query-ids", type=str, help="Comma-separated query IDs to exclude (e.g., '2,3,4')")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve per query (default: 10)")
    parser.add_argument("--file", type=str,
                       default="queries/simple_reference_based_queries.csv",
                       help="Path to input CSV file with queries")
    parser.add_argument("--embedder", type=str, default="hf",
                       help="Embedding provider (e.g., 'hf', 'openai'). Only used for vector-based retrievers.")

    args = parser.parse_args()

    # Ensure directories exist
    settings.create_directories()

    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File {args.file} not found.")
        exit(1)

    logger.info(f"Loading data from {args.file}...")
    df = pd.read_csv(args.file)
    logger.info(f"Loaded {len(df)} rows.")

    # Filter by query IDs if specified
    if args.query_ids:
        query_ids = [int(qid.strip()) for qid in args.query_ids.split(',')]
        df = df[df['query_id'].isin(query_ids)]
        logger.info(f"Filtered to {len(df)} queries with IDs: {query_ids}")

    # Exclude query IDs if specified
    if args.exclude_query_ids:
        exclude_ids = [int(qid.strip()) for qid in args.exclude_query_ids.split(',')]
        df = df[~df['query_id'].isin(exclude_ids)]
        logger.info(f"Excluded {len(exclude_ids)} queries. Remaining: {len(df)} queries")

    if len(df) == 0:
        logger.error("No queries to process after filtering!")
        exit(1)

    # Convert chunking to boolean
    chunking = args.chunking == "yes"

    # Create Neo4j driver (created once, passed to retriever)
    uri = settings.neo4j_uri.get_secret_value()
    username = settings.neo4j_username
    password = settings.neo4j_password.get_secret_value()
    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        # Prepare kwargs for retriever creation and config
        retriever_kwargs = {}
        config_kwargs = {}

        if args.retriever == "text2cypher":
            # Dynamically load the prompt from prompts.text2cypher_prompts
            prompt_name = args.text2cypher_prompt
            try:
                prompts_module = importlib.import_module("prompts.text2cypher_prompts")
                text2cypher_prompt = getattr(prompts_module, prompt_name)
            except AttributeError:
                logger.error(f"Prompt '{prompt_name}' not found in prompts.text2cypher_prompts")
                exit(1)
            except ImportError as e:
                logger.error(f"Failed to import prompts.text2cypher_prompts: {e}")
                exit(1)

            retriever_kwargs["text2cypher_prompt"] = text2cypher_prompt
            config_kwargs["text2cypher_prompt_version"] = text2cypher_prompt.prompt_version_name

        # Get the retriever instance (pass driver and embedder)
        retriever = get_retriever(args.retriever, driver, embedder=args.embedder, **retriever_kwargs)
        logger.info(f"Using {args.retriever} retriever")

        # Run the retriever
        results = await run_retriever(
            df=df,
            retriever=retriever,
            retriever_type=args.retriever,
            run_num=args.run_num,
            chunking=chunking,
            k=args.k,
            description=args.description,
            embedder=args.embedder,
            **config_kwargs
        )

        if results is not None:
            logger.info("Retriever run completed successfully! Results saved for manual evaluation.")
        else:
            logger.error("Retriever run failed.")

    finally:
        # Close driver
        driver.close()
        logger.info("Neo4j driver closed")


if __name__ == "__main__":
    asyncio.run(main())