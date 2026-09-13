#!/usr/bin/env python3
"""Unified experiment CLI for WildFrostRAG.

Convenience wrapper around the retrieval/generation scripts that:
- Resolves shortcuts like "latest/bm25"
- Lists and searches experiments tracked in MLflow

Usage:
    # Run retrieval
    python -m wildfrost_rag.cli.experiment retrieval --run 1 --retriever bm25 --description "Baseline BM25"

    # Run generation with shortcuts
    python -m wildfrost_rag.cli.experiment generation --run 1 --retrieval latest/bm25 --prompt SYSTEM_PROMPT_V1

    # List experiments
    python -m wildfrost_rag.cli.experiment list --run 1
    python -m wildfrost_rag.cli.experiment list --run 1 --type retrieval

    # Search experiments
    python -m wildfrost_rag.cli.experiment search --retriever-type bm25
    python -m wildfrost_rag.cli.experiment search --chunking no

"""

import argparse
import asyncio
import sys

import pandas as pd

from wildfrost_rag.experiment_tracker.experiment_utils import (
    list_available_retrievals,
    resolve_retrieval_reference,
)
from wildfrost_rag.domain.experiment_type import ExperimentType
from wildfrost_rag.domain.retriever_type import RetrieverType
from wildfrost_rag.services.evaluation.mlflow_tracking import search_experiments
from wildfrost_rag.cli.evaluate_retrievers import run as run_retrieval
from wildfrost_rag.cli.run_llm_generation import run as run_generation
from wildfrost_rag.core.logger import logger


def cmd_retrieval(args: argparse.Namespace) -> None:
    """Run a retrieval experiment."""
    retrieval_args = argparse.Namespace(
        run_num=args.run,
        retriever=args.retriever,
        chunking=args.chunking,
        description=args.description or "",
        text2cypher_prompt=args.text2cypher_prompt,
        query_ids=getattr(args, "query_ids", None),
        exclude_query_ids=getattr(args, "exclude_query_ids", None),
        k=args.k,
        file="queries/simple_reference_based_queries.csv",
        embedder="hf",
        queries_json=None,
        sw_query="yes",
        sw_docs="yes",
    )

    logger.info(f"Running retrieval: {args.retriever} (run {args.run})")
    asyncio.run(run_retrieval(retrieval_args))


def cmd_generation(args: argparse.Namespace) -> None:
    """Run a generation experiment."""
    run_num = args.run

    retrieval_ref = resolve_retrieval_reference(run_num, args.retrieval)
    if not retrieval_ref:
        logger.error(f"Could not resolve retrieval reference: {args.retrieval}")
        logger.info(f"Available retrievals for run {run_num}:")
        for ref in list_available_retrievals(run_num):
            logger.info(f"  - {ref}")
        sys.exit(1)

    generation_args = argparse.Namespace(
        run_num=run_num,
        retrieval_reference=retrieval_ref,
        system_prompt=args.prompt,
        description=args.description or "",
        zero_shot=False,
        rag_prompt=getattr(args, "rag_prompt", "RAG_PROMPT_V1"),
        query_ids=getattr(args, "query_ids", None),
        exclude_query_ids=getattr(args, "exclude_query_ids", None),
    )

    logger.info(f"Running generation: {args.prompt} with {retrieval_ref} (run {run_num})")
    asyncio.run(run_generation(generation_args))


def _print_run_row(row: pd.Series) -> None:
    """Print one MLflow run row in the shape both cmd_list and cmd_search share."""
    run_name = row.get("tags.mlflow.runName", "?")
    print(f"  {run_name:<20} {row.get('params.description', '')}")
    if pd.notna(row.get("params.retriever_type")):
        print(
            f"    Retriever: {row['params.retriever_type']}, Chunking: {row.get('params.chunking')}"
        )
    elif pd.notna(row.get("params.retrieval_reference")):
        print(f"    Retrieval: {row['params.retrieval_reference']}")
        print(f"    Prompt: {row.get('params.system_prompt_version')}")
    successful = row.get("metrics.successful_queries")
    total = row.get("metrics.total_queries")
    if pd.notna(successful) and pd.notna(total):
        print(f"    Queries: {int(successful)}/{int(total)}")
    print()


def cmd_list(args: argparse.Namespace) -> None:
    """List experiments for one run number, from MLflow."""
    experiment_type = ExperimentType(args.type) if args.type else None

    runs = search_experiments(experiment_type=experiment_type, run_number=args.run)

    print(f"\n{'=' * 80}")
    print(f"Experiments for Run {args.run}")
    print(f"{'=' * 80}\n")

    if runs.empty:
        print("  No experiments found.\n")
        return

    for _, row in runs.iterrows():
        _print_run_row(row)


def cmd_search(args: argparse.Namespace) -> None:
    """Search for experiments across all runs, from MLflow."""
    experiment_type = ExperimentType(args.type) if args.type else None
    chunking = args.chunking == "yes" if args.chunking else None

    runs = search_experiments(
        experiment_type=experiment_type,
        retriever_type=args.retriever_type,
        chunking=chunking,
    )

    print(f"\n{'=' * 80}")
    print(f"Search Results ({len(runs)} matches)")
    print(f"{'=' * 80}\n")

    if runs.empty:
        print("  No matching experiments found.\n")
        return

    for _, row in runs.iterrows():
        run_num = row.get("tags.run_number", "?")
        print(f"  Run {run_num}")
        _print_run_row(row)


def main() -> None:
    """Unified experiment CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified experiment CLI for WildFrostRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Retrieval command
    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval experiment")
    retrieval_parser.add_argument("--run", type=int, required=True, help="Run number")
    retrieval_parser.add_argument(
        "--retriever",
        required=True,
        choices=[member.value for member in RetrieverType],
        help="Retriever type",
    )
    retrieval_parser.add_argument(
        "--chunking", choices=["yes", "no"], default="no", help="Chunking enabled"
    )
    retrieval_parser.add_argument("--description", default="", help="Experiment description")
    retrieval_parser.add_argument(
        "--text2cypher-prompt",
        default="TEXT2CYPHER_PROMPT_V1",
        help="Text2Cypher prompt (for text2cypher retriever)",
    )
    retrieval_parser.add_argument(
        "--query-ids", help="Comma-separated query IDs to include (e.g., '1,5,10')"
    )
    retrieval_parser.add_argument(
        "--exclude-query-ids", help="Comma-separated query IDs to exclude"
    )
    retrieval_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10)",
    )

    # Generation command
    generation_parser = subparsers.add_parser("generation", help="Run generation experiment")
    generation_parser.add_argument("--run", type=int, required=True, help="Run number")
    generation_parser.add_argument(
        "--retrieval",
        required=True,
        help="Retrieval reference (e.g., 'bm25/001' or 'latest/bm25')",
    )
    generation_parser.add_argument("--prompt", required=True, help="System prompt name")
    generation_parser.add_argument("--description", default="", help="Experiment description")
    generation_parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    generation_parser.add_argument("--query-ids", help="Comma-separated query IDs to include")
    generation_parser.add_argument(
        "--exclude-query-ids", help="Comma-separated query IDs to exclude"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--run", type=int, required=True, help="Run number")
    list_parser.add_argument(
        "--type", choices=[member.value for member in ExperimentType], help="Filter by type"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search experiments")
    search_parser.add_argument(
        "--type", choices=[member.value for member in ExperimentType], help="Filter by type"
    )
    search_parser.add_argument("--retriever-type", help="Filter by retriever type")
    search_parser.add_argument("--chunking", choices=["yes", "no"], help="Filter by chunking")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    command_handlers = {
        "retrieval": cmd_retrieval,
        "generation": cmd_generation,
        "list": cmd_list,
        "search": cmd_search,
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
