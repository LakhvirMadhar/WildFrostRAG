#!/usr/bin/env python3
"""
Test script for Neo4j vector retrieval in WildFrostRAG.

This script tests the vector search functionality:
1. Takes a query string as input
2. Performs vector similarity search in Neo4j
3. Returns and displays top-k retrieved chunks
4. Shows metadata and scores for each chunk

Usage:
    python -m scripts.test_neo4j_retrieval "What is the Azul Candle?"    # Search with default k=5
    python -m scripts.test_neo4j_retrieval "Your query here" --k 10      # Search with custom k
"""

import sys
from pathlib import Path
import argparse

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.retrievers.neo4j_vector_search import Neo4jVectorSearch
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Test retrieval for a single query")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")

    args = parser.parse_args()

    logger.info(f"Searching for: '{args.query}' (k={args.k})")

    try:
        retriever = Neo4jVectorSearch()
        results = retriever.search(args.query, k=args.k)

        logger.info(f"Found {len(results)} chunks:\n")

        for i, chunk in enumerate(results):
            score = chunk.get('score', 0)
            source = chunk.get('source_file', 'unknown')
            headers = [chunk.get(f'header{j}', '') for j in range(1, 4)]
            breadcrumb = " > ".join([h for h in headers if h])
            text = chunk.get('text', '')[:200].replace('\n', ' ') + "..."

            logger.info(f"[{i+1}] Score: {score:.4f} | Source: {source}")
            if breadcrumb:
                logger.info(f"    Path: {breadcrumb}")
            logger.info(f"    Text: {text}\n")

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
