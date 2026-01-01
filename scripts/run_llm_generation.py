#!/usr/bin/env python3
"""
LLM Generation Pipeline for WildFrostRAG.

This script orchestrates the LLM generation process:
1. Loads query data from CSV
2. Processes queries in batch mode
3. Generates responses using OpenAI API
4. Supports both zero-shot and RAG modes with various retrieval strategies
5. Saves results back to CSV

Usage:
    python -m scripts.run_llm_generation --mode zero_shot     # Run zero-shot generation
    python -m scripts.run_llm_generation --mode rag --retriever vector    # Run RAG with vector search
    python -m scripts.run_llm_generation --mode rag --retriever fulltext  # Run RAG with full-text search
    python -m scripts.run_llm_generation --mode rag --retriever bm25      # Run RAG with BM25
    python -m scripts.run_llm_generation --mode rag --retriever hybrid    # Run RAG with hybrid search
    python -m scripts.run_llm_generation --mode rag --retriever text2cypher # Run RAG with Text2Cypher
    python -m scripts.run_llm_generation --overwrite          # Overwrite existing responses
    python -m scripts.run_llm_generation --batch-size 20      # Custom batch size
"""

import asyncio
import argparse
import os
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.augmented_generation import LLMGenerator, GenerationPipeline
from src.rag.retrievers import (
    Neo4jVectorSearch,
    Neo4jFullTextSearch,
    BM25Retriever,
    BM25VectorHybridRetriever,
    Text2CypherRetriever
)
from src.utils.logger import logger

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
        'hybrid': BM25VectorHybridRetriever,
        'text2cypher': Text2CypherRetriever,
    }

    if retriever_type not in retrievers:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Available types: {list(retrievers.keys())}")

    return retrievers[retriever_type]()

async def main():
    parser = argparse.ArgumentParser(description="Run LLM Generation Pipeline")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "rag"], default="zero_shot", help="Mode to run: zero_shot or rag")
    parser.add_argument("--retriever", type=str, choices=["vector", "fulltext", "bm25", "hybrid", "text2cypher"],
                       default="vector", help="Retriever to use in RAG mode (default: vector)")
    parser.add_argument("--file", type=str, default="queries/simple_reference_based_queries.csv", help="Path to input CSV file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing responses")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File {args.file} not found.")
        exit(1)

    logger.info(f"Loading data from {args.file}...")
    df = pd.read_csv(args.file)
    logger.info(f"Loaded {len(df)} rows.")

    # Initialize the LLM generator and pipeline
    llm_generator = LLMGenerator()
    pipeline = GenerationPipeline(llm_generator)

    logger.info(f"Running {args.mode} pipeline with {args.retriever} retriever (Overwrite={args.overwrite})...")

    # Determine target column and retriever based on mode
    target_column = 'openAI_RAG_response' if args.mode == 'rag' else 'openAI_zero_shot'

    if args.mode == 'rag':
        retriever = get_retriever(args.retriever)
        logger.info(f"Using {args.retriever} retriever")
    else:
        retriever = None

    # Run pipeline
    new_df = await pipeline.process_batch(
        df,
        target_column=target_column,
        retriever=retriever,
        batch_size=args.batch_size,
        overwrite=args.overwrite
    )

    logger.info(f"Saving results to {args.file}...")
    new_df.to_csv(args.file, index=False)
    logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())
