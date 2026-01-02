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


def get_retriever(retriever_type: str):
    """
    Factory function to create the appropriate retriever based on type.

    Args:
        retriever_type: Type of retriever to create

    Returns:
        An instance of the specified retriever
    """
    retrievers = {
        'vector': Neo4jVectorSearch,
        'fulltext': Neo4jFullTextSearch,
        'bm25': BM25Retriever,
        'bm25_vector': BM25VectorHybridRetriever,
        'fulltext_vector': FulltextVectorHybridRetriever,
        'bm25_fulltext_vector': BM25FulltextVectorHybridRetriever,
        'text2cypher': Text2CypherRetriever,
    }

    if retriever_type not in retrievers:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Available types: {list(retrievers.keys())}")

    return retrievers[retriever_type]()


def create_retriever_experiment_directory(outputs_dir: Path, run_num: int, retriever_type: str, chunking: str) -> Path:
    """
    Create the directory structure for a retriever experiment run.

    Args:
        outputs_dir: The base outputs directory
        run_num: The run number
        retriever_type: The type of retriever being used
        chunking: Whether chunking was used ("yes" or "no")

    Returns:
        Path to the created experiment directory
    """
    # Create directory structure: outputs/retrievers/{run_num}/{retriever_type}_{chunking}
    run_dir = outputs_dir / f"run_{run_num}"
    experiment_dir = run_dir / f"{retriever_type}_{chunking}"

    experiment_dir.mkdir(parents=True, exist_ok=True)

    return experiment_dir


async def run_retriever(df: pd.DataFrame, retriever, retriever_type: str, run_num: int, chunking: str):
    """
    Run a specific retriever on the provided dataset and save raw results.

    Args:
        df: DataFrame containing queries
        retriever: Retriever instance to run
        retriever_type: Type of retriever being used
        run_num: Experiment run number
        chunking: Whether chunking was used ("yes" or "no")

    Returns:
        Dictionary containing raw retrieval results
    """
    results = []

    # Prepare experiment directory
    experiment_dir = create_retriever_experiment_directory(
        settings.retriever_outputs_dir,
        run_num,
        retriever_type,
        chunking
    )

    logger.info(f"Running {retriever_type} retriever (chunking: {chunking}) in {experiment_dir}")

    # Check if results already exist
    results_file = experiment_dir / "results.json"
    if results_file.exists():
        logger.error(f"Results already exist at {results_file}. Please delete the directory manually or use a different run number.")
        return None

    # Process each query in the dataset
    for idx, row in df.iterrows():
        query = row['query']

        if pd.isna(query) or query == '':
            continue

        logger.info(f"Processing query {idx+1}/{len(df)}: '{query}'")

        # Retrieve chunks using the retriever
        retrieved_chunks = retriever.search(query, k=10)  # Retrieve top 10

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

    # Prepare experiment metadata
    experiment_metadata = {
        "run_number": run_num,
        "timestamp": datetime.now().isoformat(),
        "retriever_type": retriever_type,
        "chunking": chunking,
        "total_queries": len([r for _, r in df.iterrows() if not pd.isna(r.get('query', '')) and r.get('query', '') != '']),
        "successful_queries": len(results)
    }

    # Save results to the experiment directory
    results_data = {
        "metadata": experiment_metadata,
        "results": results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, default=str)

    logger.info(f"Raw retrieval results saved to {results_file}")

    return results_data


async def main():
    parser = argparse.ArgumentParser(description="Run different retrievers and save raw results")
    parser.add_argument("--run-num", type=int, required=True, help="Experiment run number")
    parser.add_argument("--retriever", type=str,
                       choices=["vector", "fulltext", "bm25", "bm25_vector", "fulltext_vector", "bm25_fulltext_vector", "text2cypher"],
                       required=True, help="Retriever to run")
    parser.add_argument("--chunking", type=str, choices=["yes", "no"], default="no",
                       help="Whether chunking was used during ingestion (affects directory naming)")
    parser.add_argument("--file", type=str,
                       default="queries/simple_reference_based_queries.csv",
                       help="Path to input CSV file with queries")

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

    # Get the retriever instance
    retriever = get_retriever(args.retriever)
    logger.info(f"Using {args.retriever} retriever")

    # Run the retriever
    results = await run_retriever(df, retriever, args.retriever, args.run_num, args.chunking)

    if results is not None:
        logger.info("Retriever run completed successfully! Results saved for manual evaluation.")
    else:
        logger.error("Retriever run failed due to existing results.")


if __name__ == "__main__":
    asyncio.run(main())