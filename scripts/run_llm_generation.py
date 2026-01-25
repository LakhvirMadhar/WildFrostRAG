#!/usr/bin/env python3
"""
LLM Generation Pipeline for WildFrostRAG.

This script orchestrates the LLM generation process:
1. Loads query data from CSV or retrieval results
2. Generates responses using OpenAI API
3. Supports both zero-shot (baseline) and RAG modes
4. Saves results to structured output directories with run numbers

Usage:
    # Zero-shot mode (baseline - no retrieval)
    python -m scripts.run_llm_generation --run-num 1 --zero-shot --system-prompt SYSTEM_PROMPT_V1

    # RAG mode (with retrieval)
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
from src.rag.augmented_generation.openai_client import generate_zero_shot, generate_rag
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM Generation Pipeline")
    parser.add_argument("--run-num", type=int, default=1,
                        help="Experiment run number (default: 1)")
    parser.add_argument("--system-prompt", type=str, required=True,
                        help="System prompt name (e.g., SYSTEM_PROMPT_V1)")
    parser.add_argument("--description", type=str, default="",
                        help="Human-readable description of this experiment")
    parser.add_argument("--query-ids", type=str,
                        help="Comma-separated query IDs to include (e.g., '1,5,10')")
    parser.add_argument("--exclude-query-ids", type=str,
                        help="Comma-separated query IDs to exclude")

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--zero-shot", action="store_true",
                            help="Run zero-shot generation (no retrieval)")
    mode_group.add_argument("--retrieval-reference", type=str,
                            help="Retrieval experiment to use for RAG (e.g., 'bm25/001')")

    # RAG-specific arguments
    parser.add_argument("--rag-prompt", type=str,
                        help="RAG prompt name (required for RAG mode, e.g., RAG_PROMPT_V1)")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations."""
    if args.retrieval_reference and not args.rag_prompt:
        logger.error("--rag-prompt is required when using --retrieval-reference")
        sys.exit(1)


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


def load_queries_for_zero_shot() -> list[dict]:
    """Load queries from the base query CSV for zero-shot mode."""
    import pandas as pd

    queries_path = settings.project_root / "queries" / "simple_reference_based_queries.csv"
    if not queries_path.exists():
        logger.error(f"Queries file not found: {queries_path}")
        sys.exit(1)

    df = pd.read_csv(queries_path)
    logger.info(f"Loaded {len(df)} queries from {queries_path}")

    results = []
    for _, row in df.iterrows():
        results.append({
            'query_id': row.get('query_id', row.name),
            'query': row['query'],
            'retrieved_chunks': []
        })

    return results


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


def load_prompt(prompt_name: str, module_name: str = "prompts.system_prompts") -> Any:
    """Load a prompt from the prompts module."""
    try:
        prompts_module = importlib.import_module(module_name)
        return getattr(prompts_module, prompt_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load prompt '{prompt_name}' from {module_name}: {e}")
        sys.exit(1)


def extract_context_from_chunks(chunks: list[dict]) -> str:
    """Extract text content from retrieved chunks."""
    context_texts = [chunk['text'] for chunk in chunks if 'text' in chunk]
    return "\n\n".join(context_texts)


async def run_generation(
    query_results: list[dict],
    system_prompt: Any,
    rag_prompt: Any | None,
    is_zero_shot: bool
) -> tuple[list[dict], int, int]:
    """
    Run generation for all queries.

    Returns:
        Tuple of (results, successful_count, failed_count)
    """
    results = []
    successful = 0
    failed = 0

    mode_name = "zero-shot" if is_zero_shot else "RAG"
    logger.info(f"Running {mode_name} generation for {len(query_results)} queries...")

    for query_result in query_results:
        query = query_result['query']
        chunks = query_result.get('retrieved_chunks', [])

        logger.info(f"Generating response for query: {query[:50]}...")

        try:
            if is_zero_shot:
                response = await generate_zero_shot(query, system_prompt)
            else:
                context = extract_context_from_chunks(chunks)
                if not context:
                    response = "ERROR: No retrieved chunks available"
                    raise ValueError("No context available")
                response = await generate_rag(query, context, system_prompt, rag_prompt)
            successful += 1
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            response = f"ERROR: {str(e)}"
            failed += 1

        results.append({
            'query_id': query_result['query_id'],
            'query': query,
            'response': response,
            'retrieved_chunks': chunks
        })

    return results, successful, failed


def save_experiment(
    experiment_dir: Path,
    run_num: int,
    generation_id: str,
    retrieval_reference: str | None,
    system_prompt: Any,
    rag_prompt: Any | None,
    results: list[dict],
    successful: int,
    failed: int,
    description: str,
    is_zero_shot: bool
) -> None:
    """Save experiment config and results."""
    config = create_generation_config(
        run_num=run_num,
        generation_id=f"gen/{generation_id}",
        retrieval_reference=retrieval_reference or "zero-shot",
        system_prompt_version=system_prompt.prompt_version_name,
        rag_prompt_version=rag_prompt.prompt_version_name if rag_prompt else None,
        total_queries=len(results),
        successful_queries=successful,
        failed_queries=failed,
        description=description,
        batch_size=1
    )

    config['is_zero_shot'] = is_zero_shot

    save_config(config, experiment_dir)

    registry = ExperimentRegistry()
    registry.register_generation(run_num, generation_id, config)

    save_results(results, experiment_dir / "results.json")

    logger.info("Experiment completed successfully!")
    logger.info(f"Generation ID: gen/{generation_id}")
    logger.info(f"Results saved to {experiment_dir}")


async def main():
    args = parse_args()
    validate_args(args)
    settings.create_directories()

    is_zero_shot = args.zero_shot

    # Load query data
    if is_zero_shot:
        query_results = load_queries_for_zero_shot()
        logger.info("Running in ZERO-SHOT mode (no retrieval)")
    else:
        _, query_results = load_retrieval_data(args.run_num, args.retrieval_reference)
        logger.info("Running in RAG mode")

    # Filter queries
    query_results = filter_results_by_query_ids(
        query_results, args.query_ids, args.exclude_query_ids
    )

    # Load prompts
    system_prompt = load_prompt(args.system_prompt)
    rag_prompt = load_prompt(args.rag_prompt) if args.rag_prompt else None

    # Setup experiment directory
    base_path = settings.outputs_dir / f"run_{args.run_num}" / "generation"
    generation_id = get_next_experiment_id(base_path)
    experiment_dir = base_path / generation_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generation experiment directory: {experiment_dir}")
    logger.info(f"Generation ID: gen/{generation_id}")
    logger.info(f"Using system prompt: {system_prompt.prompt_version_name}")
    if rag_prompt:
        logger.info(f"Using RAG prompt: {rag_prompt.prompt_version_name}")

    # Run generation
    results, successful, failed = await run_generation(
        query_results, system_prompt, rag_prompt, is_zero_shot
    )

    # Save experiment
    save_experiment(
        experiment_dir, args.run_num, generation_id,
        args.retrieval_reference, system_prompt, rag_prompt,
        results, successful, failed, args.description, is_zero_shot
    )

    logger.info(f"Completed: {successful} successful, {failed} failed")
    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
