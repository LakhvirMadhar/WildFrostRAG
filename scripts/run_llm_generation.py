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

from src.utils.logger import logger
from src.utils.config import settings
from src.utils.experiment_utils import (
    get_next_experiment_id,
    create_generation_config,
    save_config,
    save_results,
    load_config,
    load_results,
    validate_retrieval_reference,
    list_available_retrievals
)
from src.experiment_tracker import ExperimentRegistry
from src.rag.augmented_generation.call_llm_generation import LLMGenerator
from src.utils.prompt_utils import format_prompt_tuple
import importlib


async def main():
    parser = argparse.ArgumentParser(description="Run LLM Generation Pipeline")
    parser.add_argument("--run-num", type=int, default=1, help="Experiment run number (default: 1)")
    parser.add_argument("--retrieval-reference", type=str, required=True,
                       help="Retrieval experiment to use (e.g., 'bm25/001', 'vector/002')")
    parser.add_argument("--system-prompt", type=str, required=True,
                       help="System prompt name (e.g., SYSTEM_PROMPT_V1)")
    parser.add_argument("--rag-prompt", type=str, required=True,
                       help="RAG prompt name for user message formatting (e.g., RAG_PROMPT_V1)")
    parser.add_argument("--description", type=str, default="",
                       help="Human-readable description of this experiment")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--query-ids", type=str, help="Comma-separated query IDs to include (e.g., '1,5,10')")
    parser.add_argument("--exclude-query-ids", type=str, help="Comma-separated query IDs to exclude")

    args = parser.parse_args()

    # Ensure directories exist
    settings.create_directories()

    # Validate retrieval reference exists
    if not validate_retrieval_reference(args.run_num, args.retrieval_reference):
        logger.error(f"Retrieval reference not found: {args.retrieval_reference}")
        logger.info(f"Available retrievals for run {args.run_num}:")
        available = list_available_retrievals(args.run_num)
        if available:
            for ref in available:
                logger.info(f"  - {ref}")
        else:
            logger.info(f"  No retrievals found for run {args.run_num}")
        exit(1)

    # Load retrieval config and results
    retrieval_path = settings.outputs_dir / f"run_{args.run_num}" / "retrievals" / args.retrieval_reference
    retrieval_config = load_config(retrieval_path / "config.json")
    retrieval_results = load_results(retrieval_path / "results.json")

    logger.info(f"Loaded retrieval results from: {args.retrieval_reference}")
    logger.info(f"Retriever type: {retrieval_config['retriever_type']}")
    logger.info(f"Total retrieved queries: {len(retrieval_results)}")

    # Filter by query IDs if specified
    if args.query_ids:
        query_ids = [int(qid.strip()) for qid in args.query_ids.split(',')]
        retrieval_results = [r for r in retrieval_results if r['query_id'] in query_ids]
        logger.info(f"Filtered to {len(retrieval_results)} queries with IDs: {query_ids}")

    # Exclude query IDs if specified
    if args.exclude_query_ids:
        exclude_ids = [int(qid.strip()) for qid in args.exclude_query_ids.split(',')]
        retrieval_results = [r for r in retrieval_results if r['query_id'] not in exclude_ids]
        logger.info(f"Excluded {len(exclude_ids)} queries. Remaining: {len(retrieval_results)} queries")

    if len(retrieval_results) == 0:
        logger.error("No queries to process after filtering!")
        exit(1)

    # Dynamically load system and RAG prompts
    try:
        prompts_module = importlib.import_module("prompts.system_prompts")
        system_prompt = getattr(prompts_module, args.system_prompt)
        rag_prompt = getattr(prompts_module, args.rag_prompt)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load prompts: {e}")
        exit(1)

    # Generate generation experiment ID
    base_path = settings.outputs_dir / f"run_{args.run_num}" / "generation"
    generation_id = get_next_experiment_id(base_path)
    experiment_dir = base_path / generation_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generation experiment directory: {experiment_dir}")
    logger.info(f"Generation ID: gen/{generation_id}")

    # Initialize LLM generator with system prompt
    llm_generator = LLMGenerator(system_prompt=system_prompt)

    logger.info(f"Running generation pipeline with {args.retrieval_reference}...")
    logger.info(f"Using system prompt: {system_prompt.prompt_version_name}")
    logger.info(f"Using RAG prompt: {rag_prompt.prompt_version_name}")

    # Run generation with retrieval results
    results = []
    successful_queries = 0
    failed_queries = 0

    for retrieval_result in retrieval_results:
        query = retrieval_result['query']
        retrieved_chunks = retrieval_result['retrieved_chunks']

        logger.info(f"Generating response for query: {query}")

        # Generate response using LLM with retrieved context
        try:
            # Build context from retrieved chunks
            if not retrieved_chunks:
                response = "ERROR: No retrieved chunks available"
                failed_queries += 1
            else:
                # Extract text from chunks
                context_texts = []
                for chunk in retrieved_chunks:
                    if 'text' in chunk:
                        context_texts.append(chunk['text'])

                context = "\n\n".join(context_texts)

                # Generate response using LLMGenerator with RAG prompt
                response = await llm_generator.generate_rag_response(
                    query=query,
                    context=context,
                    rag_prompt=rag_prompt
                )
                successful_queries += 1

        except Exception as e:
            logger.error(f"Failed to generate response for query {query}: {e}")
            response = f"ERROR: {str(e)}"
            failed_queries += 1

        results.append({
            'query_id': retrieval_result['query_id'],
            'query': query,
            'response': response,
            'retrieved_chunks': retrieved_chunks
        })

    # Build config
    config = create_generation_config(
        run_num=args.run_num,
        generation_id=f"gen/{generation_id}",
        retrieval_reference=args.retrieval_reference,
        system_prompt_version=system_prompt.prompt_version_name,
        rag_prompt_version=rag_prompt.prompt_version_name,
        total_queries=len(retrieval_results),
        successful_queries=successful_queries,
        failed_queries=failed_queries,
        description=args.description,
        batch_size=args.batch_size
    )

    # Save config
    save_config(config, experiment_dir)

    # Register in experiment registry
    registry = ExperimentRegistry()
    registry.register_generation(args.run_num, generation_id, config)

    # Save results
    save_results(results, experiment_dir / "results.json")

    logger.info(f"Experiment completed successfully!")
    logger.info(f"Generation ID: gen/{generation_id}")
    logger.info(f"Results saved to {experiment_dir}")
    logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())
