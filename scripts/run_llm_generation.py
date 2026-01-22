#!/usr/bin/env python3
"""
LLM Generation Pipeline for WildFrostRAG.

This script orchestrates the LLM generation process:
1. Loads query data from CSV
2. Processes queries in batch mode
3. Generates responses using OpenAI API
4. Supports both zero-shot and RAG modes with various retrieval strategies
5. Saves results to structured output directories with run numbers

Usage:
    python -m scripts.run_llm_generation --run-num 1 --retrieval-reference bm25/001 \
        --system-prompt SYSTEM_PROMPT_V1 --rag-prompt RAG_PROMPT_V1
"""

import asyncio
import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM Generation Pipeline")
    parser.add_argument("--run-num", type=int, default=1,
                        help="Experiment run number (default: 1)")
    parser.add_argument("--retrieval-reference", type=str, required=True,
                        help="Retrieval experiment to use (e.g., 'bm25/001', 'vector/002')")
    parser.add_argument("--system-prompt", type=str, required=True,
                        help="System prompt name (e.g., SYSTEM_PROMPT_V1)")
    parser.add_argument("--rag-prompt", type=str, required=True,
                        help="RAG prompt name for user message formatting (e.g., RAG_PROMPT_V1)")
    parser.add_argument("--description", type=str, default="",
                        help="Human-readable description of this experiment")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for processing")
    parser.add_argument("--query-ids", type=str,
                        help="Comma-separated query IDs to include (e.g., '1,5,10')")
    parser.add_argument("--exclude-query-ids", type=str,
                        help="Comma-separated query IDs to exclude")
    return parser.parse_args()


def load_retrieval_data(run_num: int, retrieval_reference: str) -> tuple[dict, list[dict]]:
    """Load and validate retrieval config and results."""
    if not validate_retrieval_reference(run_num, retrieval_reference):
        logger.error(f"Retrieval reference not found: {retrieval_reference}")
        logger.info(f"Available retrievals for run {run_num}:")
        available = list_available_retrievals(run_num)
        for ref in available:
            logger.info(f"  - {ref}")
        if not available:
            logger.info(f"  No retrievals found for run {run_num}")
        sys.exit(1)

    retrieval_path = settings.outputs_dir / f"run_{run_num}" / "retrievals" / retrieval_reference
    config = load_config(retrieval_path / "config.json")
    results = load_results(retrieval_path / "results.json")

    logger.info(f"Loaded retrieval results from: {retrieval_reference}")
    logger.info(f"Retriever type: {config['retriever_type']}")
    logger.info(f"Total retrieved queries: {len(results)}")

    return config, results


def filter_results_by_query_ids(
    results: list[dict],
    include_ids: str | None,
    exclude_ids: str | None
) -> list[dict]:
    """Filter retrieval results by query IDs."""
    if include_ids:
        ids = [int(qid.strip()) for qid in include_ids.split(',')]
        results = [r for r in results if r['query_id'] in ids]
        logger.info(f"Filtered to {len(results)} queries with IDs: {ids}")

    if exclude_ids:
        ids = [int(qid.strip()) for qid in exclude_ids.split(',')]
        results = [r for r in results if r['query_id'] not in ids]
        logger.info(f"Excluded {len(ids)} queries. Remaining: {len(results)} queries")

    if len(results) == 0:
        logger.error("No queries to process after filtering!")
        sys.exit(1)

    return results


def load_prompts(system_prompt_name: str, rag_prompt_name: str) -> tuple[Any, Any]:
    """Load system and RAG prompts from prompts module."""
    try:
        prompts_module = importlib.import_module("prompts.system_prompts")
        system_prompt = getattr(prompts_module, system_prompt_name)
        rag_prompt = getattr(prompts_module, rag_prompt_name)
        return system_prompt, rag_prompt
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load prompts: {e}")
        sys.exit(1)


def extract_context_from_chunks(chunks: list[dict]) -> str:
    """Extract text content from retrieved chunks."""
    context_texts = [chunk['text'] for chunk in chunks if 'text' in chunk]
    return "\n\n".join(context_texts)


async def generate_response_for_query(
    llm_generator: LLMGenerator,
    query: str,
    chunks: list[dict],
    rag_prompt: Any
) -> tuple[str, bool]:
    """
    Generate LLM response for a single query.

    Returns:
        Tuple of (response, success)
    """
    if not chunks:
        return "ERROR: No retrieved chunks available", False

    context = extract_context_from_chunks(chunks)
    try:
        response = await llm_generator.generate_rag_response(
            query=query,
            context=context,
            rag_prompt=rag_prompt
        )
        return response, True
    except Exception as e:
        logger.error(f"Failed to generate response for query {query}: {e}")
        return f"ERROR: {str(e)}", False


async def run_generation(
    retrieval_results: list[dict],
    llm_generator: LLMGenerator,
    rag_prompt: Any
) -> tuple[list[dict], int, int]:
    """
    Run generation for all queries.

    Returns:
        Tuple of (results, successful_count, failed_count)
    """
    results = []
    successful = 0
    failed = 0

    for retrieval_result in retrieval_results:
        query = retrieval_result['query']
        chunks = retrieval_result['retrieved_chunks']

        logger.info(f"Generating response for query: {query}")
        response, success = await generate_response_for_query(
            llm_generator, query, chunks, rag_prompt
        )

        if success:
            successful += 1
        else:
            failed += 1

        results.append({
            'query_id': retrieval_result['query_id'],
            'query': query,
            'response': response,
            'retrieved_chunks': chunks
        })

    return results, successful, failed


def save_experiment(
    experiment_dir: Path,
    run_num: int,
    generation_id: str,
    retrieval_reference: str,
    system_prompt: Any,
    rag_prompt: Any,
    results: list[dict],
    successful: int,
    failed: int,
    description: str,
    batch_size: int
) -> None:
    """Save experiment config and results."""
    config = create_generation_config(
        run_num=run_num,
        generation_id=f"gen/{generation_id}",
        retrieval_reference=retrieval_reference,
        system_prompt_version=system_prompt.prompt_version_name,
        rag_prompt_version=rag_prompt.prompt_version_name,
        total_queries=len(results),
        successful_queries=successful,
        failed_queries=failed,
        description=description,
        batch_size=batch_size
    )

    save_config(config, experiment_dir)

    registry = ExperimentRegistry()
    registry.register_generation(run_num, generation_id, config)

    save_results(results, experiment_dir / "results.json")

    logger.info("Experiment completed successfully!")
    logger.info(f"Generation ID: gen/{generation_id}")
    logger.info(f"Results saved to {experiment_dir}")


async def main():
    args = parse_args()
    settings.create_directories()

    # Load retrieval data
    _, retrieval_results = load_retrieval_data(args.run_num, args.retrieval_reference)

    # Filter queries
    retrieval_results = filter_results_by_query_ids(
        retrieval_results, args.query_ids, args.exclude_query_ids
    )

    # Load prompts
    system_prompt, rag_prompt = load_prompts(args.system_prompt, args.rag_prompt)

    # Setup experiment directory
    base_path = settings.outputs_dir / f"run_{args.run_num}" / "generation"
    generation_id = get_next_experiment_id(base_path)
    experiment_dir = base_path / generation_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generation experiment directory: {experiment_dir}")
    logger.info(f"Generation ID: gen/{generation_id}")
    logger.info(f"Using system prompt: {system_prompt.prompt_version_name}")
    logger.info(f"Using RAG prompt: {rag_prompt.prompt_version_name}")

    # Run generation
    llm_generator = LLMGenerator(system_prompt=system_prompt)
    results, successful, failed = await run_generation(
        retrieval_results, llm_generator, rag_prompt
    )

    # Save experiment
    save_experiment(
        experiment_dir, args.run_num, generation_id, args.retrieval_reference,
        system_prompt, rag_prompt, results, successful, failed,
        args.description, args.batch_size
    )

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
