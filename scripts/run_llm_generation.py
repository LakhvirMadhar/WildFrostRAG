#!/usr/bin/env python3
"""
LLM Generation Pipeline for WildFrostRAG.

This script orchestrates the LLM generation process:
1. Loads query data from CSV
2. Processes queries in batch mode
3. Generates responses using OpenAI API
4. Supports both zero-shot and RAG modes with various retrieval strategies
5. Saves results to structured output directories with run numbers
6. Prevents overwriting of existing experiment results

Usage:
    python -m scripts.run_llm_generation --run-num 1 --mode zero_shot     # Run zero-shot generation with run number 1
    python -m scripts.run_llm_generation --run-num 1 --mode rag --retriever vector    # Run RAG with vector search
    python -m scripts.run_llm_generation --run-num 1 --mode rag --retriever fulltext  # Run RAG with full-text search
    python -m scripts.run_llm_generation --run-num 1 --mode rag --retriever bm25      # Run RAG with BM25
    python -m scripts.run_llm_generation --run-num 1 --mode rag --retriever hybrid    # Run RAG with hybrid search
    python -m scripts.run_llm_generation --run-num 1 --mode rag --retriever text2cypher # Run RAG with Text2Cypher
    python -m scripts.run_llm_generation --run-num 1 --batch-size 20      # Custom batch size
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

from src.rag.augmented_generation import LLMGenerator, GenerationPipeline
from src.rag.retrievers import (
    Neo4jVectorSearch,
    Neo4jFullTextSearch,
    BM25Retriever,
    BM25VectorHybridRetriever,
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
        'hybrid': BM25VectorHybridRetriever,
        'text2cypher': Text2CypherRetriever,
    }

    if retriever_type not in retrievers:
        raise ValueError(f"Unknown retriever type: {retriever_type}. Available types: {list(retrievers.keys())}")

    return retrievers[retriever_type]()

def create_generation_experiment_directory(outputs_dir: Path, run_num: int, retriever_type: str, model_name: str) -> Path:
    """
    Create the directory structure for a generation experiment run.

    Args:
        outputs_dir: The base generation outputs directory
        run_num: The run number
        retriever_type: The type of retriever being used
        model_name: The model name being used

    Returns:
        Path to the created experiment directory
    """
    # Create directory structure: outputs/generation/{run_num}/{retriever_type}_{model_name}
    run_dir = outputs_dir / f"run_{run_num}"
    experiment_dir = run_dir / f"{retriever_type}_{model_name}"

    experiment_dir.mkdir(parents=True, exist_ok=True)

    return experiment_dir

async def main():
    parser = argparse.ArgumentParser(description="Run LLM Generation Pipeline")
    parser.add_argument("--run-num", type=int, required=True, help="Experiment run number")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "rag"], default="zero_shot", help="Mode to run: zero_shot or rag")
    parser.add_argument("--retriever", type=str, choices=["vector", "fulltext", "bm25", "hybrid", "text2cypher"],
                       default="vector", help="Retriever to use in RAG mode (default: vector)")
    parser.add_argument("--file", type=str, default="queries/simple_reference_based_queries.csv", help="Path to input CSV file")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")

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

    # Determine model name and retriever type for directory naming
    model_name = settings.openai_model_name.replace("-", "_")
    retriever_type = args.retriever if args.mode == 'rag' else 'zero_shot'

    # Create experiment directory
    experiment_dir = create_generation_experiment_directory(
        settings.generation_outputs_dir,
        args.run_num,
        retriever_type,
        model_name
    )

    logger.info(f"Experiment directory: {experiment_dir}")

    # Check if experiment directory already contains results and prevent overwriting
    results_file = experiment_dir / "results.json"
    if results_file.exists():
        logger.error(f"Results already exist at {results_file}. Please delete the directory manually or use a different run number.")
        exit(1)

    # Initialize the LLM generator and pipeline
    llm_generator = LLMGenerator()
    pipeline = GenerationPipeline(llm_generator)

    logger.info(f"Running {args.mode} pipeline with {args.retriever} retriever (Run #{args.run_num})...")

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
        batch_size=args.batch_size
    )

    # Prepare experiment metadata
    experiment_metadata = {
        "run_number": args.run_num,
        "timestamp": datetime.now().isoformat(),
        "model_name": settings.openai_model_name,
        "mode": args.mode,
        "retriever": args.retriever if args.mode == 'rag' else 'zero_shot',
        "batch_size": args.batch_size,
        "input_file": args.file,
        "total_queries": len(df),
        "successful_queries": len(new_df[new_df[target_column].notna()]),
    }

    # Save results to the experiment directory
    results_data = {
        "metadata": experiment_metadata,
        "results": new_df.to_dict('records')
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, default=str)

    logger.info(f"Results saved to {results_file}")
    logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())
