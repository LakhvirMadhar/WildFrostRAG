#!/usr/bin/env python3
"""Test script for Neo4j retrieval in WildFrostRAG.

This script tests different retrieval strategies:
1. Takes a query string as input
2. Performs retrieval using specified method in Neo4j
3. Returns and displays top-k retrieved chunks
4. Shows metadata and scores for each chunk

Usage:
    python -m scripts.test_neo4j_retrieval --retriever vector "What is the Azul Candle?"    # Vector search with default k=5
    python -m scripts.test_neo4j_retrieval --retriever fulltext "Your query here" --k 10   # Full-text search with custom k
    python -m scripts.test_neo4j_retrieval --retriever bm25 "Your query here"              # BM25 search
    python -m scripts.test_neo4j_retrieval --retriever bm25_vector "Your query here"       # BM25+Vector hybrid search
    python -m scripts.test_neo4j_retrieval --retriever fulltext_vector "Your query here"   # Fulltext+Vector hybrid search
    python -m scripts.test_neo4j_retrieval --retriever bm25_fulltext_vector "Your query here" # BM25+Fulltext+Vector hybrid search
    python -m scripts.test_neo4j_retrieval --retriever text2cypher "Your query here"       # Text2Cypher search
"""

import sys
from pathlib import Path
import argparse
from typing import Any
from neo4j import Driver

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from wildfrost_rag.rag.retrievers import (
    Neo4jVectorSearch,
    Neo4jFullTextSearch,
    BM25Retriever,
    BM25VectorHybridRetriever,
    FulltextVectorHybridRetriever,
    BM25FulltextVectorHybridRetriever,
    Text2CypherRetriever,
)
from wildfrost_rag.neo4j_kg.driver import neo4j_driver
from wildfrost_rag.utils.logger import logger


def get_retriever(retriever_type: str, driver: Driver) -> Any:  # noqa: ANN401
    """Factory function to create the appropriate retriever based on type.

    Args:
        retriever_type: Type of retriever to create
        driver: Neo4j driver instance to pass to the retriever

    Returns:
        An instance of the specified retriever
    """
    retrievers = {
        "vector": Neo4jVectorSearch,
        "fulltext": Neo4jFullTextSearch,
        "bm25": BM25Retriever,
        "bm25_vector": BM25VectorHybridRetriever,
        "fulltext_vector": FulltextVectorHybridRetriever,
        "bm25_fulltext_vector": BM25FulltextVectorHybridRetriever,
        "text2cypher": Text2CypherRetriever,
    }

    if retriever_type not in retrievers:
        raise ValueError(
            f"Unknown retriever type: {retriever_type}. Available types: {list(retrievers.keys())}"
        )

    return retrievers[retriever_type](driver)


def main() -> None:
    """Test retrieval for a single query against Neo4j."""
    parser = argparse.ArgumentParser(description="Test retrieval for a single query")
    parser.add_argument("query", type=str, help="The search query")
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
        ],
        default="vector",
        help="Retrieval method to use (default: vector)",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")

    args = parser.parse_args()

    logger.info(f"Searching for: '{args.query}' using {args.retriever} retriever (k={args.k})")

    with neo4j_driver() as driver:
        _run_search(args, driver)

    logger.info("Neo4j driver closed")


def _run_search(args: argparse.Namespace, driver: Driver) -> None:
    """Run the retriever search and print the results."""
    try:
        retriever = get_retriever(args.retriever, driver)
        results = retriever.search(args.query, k=args.k)

        # Check if this is a hybrid result and display individual retriever results first
        is_hybrid = results and "individual_results" in results[0]

        if is_hybrid:
            logger.info("Individual retriever results before fusion:\n")
            individual_results = results[0]["individual_results"]
            for retriever_name, retriever_results in individual_results.items():
                logger.info(f"--- {retriever_name.upper()} RETRIEVER RESULTS ---")
                for j, chunk in enumerate(
                    retriever_results[: args.k]
                ):  # Show top k from each retriever
                    score = chunk.get("score", 0)
                    source = chunk.get("source_file", "unknown")
                    search_type = chunk.get("search_type", "unknown")
                    text = chunk.get("text", "")[:200].replace("\n", " ") + "..."
                    logger.info(
                        f"  [{j + 1}] Score: {score:.4f} | Source: {source} | Search Type: {search_type}"
                    )
                    logger.info(f"      Text: {text}")
                logger.info("")  # Empty line after each retriever's results

        # Display results differently for hybrid vs non-hybrid
        if is_hybrid:
            logger.info(f"Final fused hybrid results ({len(results)} total):\n")
        else:
            logger.info(f"Retrieval results ({len(results)} total):\n")

        for i, chunk in enumerate(results):
            score = chunk.get("score", 0)
            source = chunk.get("source_file", "unknown")
            search_type = chunk.get("search_type", "unknown")
            text = chunk.get("text", "")[:200].replace("\n", " ") + "..."

            if is_hybrid:
                rrf_score = chunk.get("rrf_score", 0)
                source_retriever = chunk.get("source_retriever", "")
                retriever_scores = chunk.get("retriever_scores", {})
                logger.info(
                    f"[{i + 1}] RRF Score: {rrf_score:.4f} | Original Scores: {retriever_scores} | Source: {source} | Search Type: {search_type}"
                )
                if source_retriever:
                    logger.info(f"    Source Retriever: {source_retriever}")
            else:
                logger.info(
                    f"[{i + 1}] Score: {score:.4f} | Source: {source} | Search Type: {search_type}"
                )

            logger.info(f"    Text: {text}\n")

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
