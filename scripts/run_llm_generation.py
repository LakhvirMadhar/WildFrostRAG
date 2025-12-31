#!/usr/bin/env python3
"""
LLM Generation Pipeline for WildFrostRAG.

This script orchestrates the LLM generation process:
1. Loads query data from CSV
2. Processes queries in batch mode
3. Generates responses using OpenAI API
4. Supports both zero-shot and RAG modes
5. Saves results back to CSV

Usage:
    python -m scripts.run_llm_generation --mode zero_shot     # Run zero-shot generation
    python -m scripts.run_llm_generation --mode rag           # Run RAG generation
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

from src.rag.call_llm_generation import LLMGenerator, GenerationPipeline
from src.rag.retrievers.neo4j_vector_search import Neo4jVectorSearch
from src.utils.logger import logger

async def main():
    parser = argparse.ArgumentParser(description="Run LLM Generation Pipeline")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "rag"], default="zero_shot", help="Mode to run: zero_shot or rag")
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

    logger.info(f"Running {args.mode} pipeline (Overwrite={args.overwrite})...")

    # Determine target column and retriever based on mode
    target_column = 'openAI_RAG_response' if args.mode == 'rag' else 'openAI_zero_shot'
    retriever = Neo4jVectorSearch() if args.mode == 'rag' else None

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
