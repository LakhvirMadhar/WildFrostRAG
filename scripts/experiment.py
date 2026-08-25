#!/usr/bin/env python3
"""Unified experiment CLI - MLflow-like interface for WildFrostRAG.

This provides convenient shortcuts and automation for running experiments:
- Auto-manages run numbers
- Resolves shortcuts like "latest/bm25"
- Lists and searches experiments
- Wraps the underlying scripts with convenience features

Usage:
    # Run retrieval
    python -m scripts.experiment retrieval --retriever bm25 --description "Baseline BM25"

    # Run generation with shortcuts
    python -m scripts.experiment generation --retrieval latest/bm25 --prompt SYSTEM_PROMPT_V1

    # List experiments
    python -m scripts.experiment list
    python -m scripts.experiment list --type retrieval
    python -m scripts.experiment list --run 1

    # Search experiments
    python -m scripts.experiment search --retriever-type bm25
    python -m scripts.experiment search --chunking no

    # Manage runs
    python -m scripts.experiment new-run  # Increment to next run number
    python -m scripts.experiment current  # Show current run number
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from experiment_tracker import ExperimentRegistry
from models.experiment import GenerationRecord, RetrievalRecord
from scripts.evaluate_retrievers import run as run_retrieval
from scripts.run_llm_generation import run as run_generation
from utils.logger import logger


def cmd_retrieval(args: argparse.Namespace) -> None:
    """Run a retrieval experiment."""
    registry = ExperimentRegistry()
    run_num = args.run if args.run != "current" else registry.get_current_run()

    retrieval_args = argparse.Namespace(
        run_num=run_num,
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

    logger.info(f"Running retrieval: {args.retriever} (run {run_num})")
    asyncio.run(run_retrieval(retrieval_args))


def cmd_generation(args: argparse.Namespace) -> None:
    """Run a generation experiment."""
    registry = ExperimentRegistry()
    run_num = args.run if args.run != "current" else registry.get_current_run()

    # Resolve retrieval reference
    retrieval_ref = registry.resolve_retrieval_reference(run_num, args.retrieval)
    if not retrieval_ref:
        logger.error(f"Could not resolve retrieval reference: {args.retrieval}")
        logger.info(f"Available retrievals for run {run_num}:")
        for ret in registry.list_retrievals(run_num):
            logger.info(f"  - {ret.reference}: {ret.description}")
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


def cmd_list(args: argparse.Namespace) -> None:
    """List experiments."""
    registry = ExperimentRegistry()
    run_num = args.run if args.run != "current" else registry.get_current_run()

    print(f"\n{'=' * 80}")
    print(f"Experiments for Run {run_num}")
    print(f"{'=' * 80}\n")

    if args.type in [None, "retrieval"]:
        print("RETRIEVAL EXPERIMENTS")
        print("-" * 80)
        retrievals = registry.list_retrievals(run_num)
        if retrievals:
            for ret in retrievals:
                print(f"  {ret.reference:<20} {ret.description}")
                print(f"    Retriever: {ret.retriever_type}, Chunking: {ret.chunking}")
                print(f"    Queries: {ret.successful_queries}/{ret.total_queries}")
                print(f"    Time: {ret.timestamp}")
                print()
        else:
            print("  No retrieval experiments found.\n")

    if args.type in [None, "generation"]:
        print("GENERATION EXPERIMENTS")
        print("-" * 80)
        generations = registry.list_generations(run_num)
        if generations:
            for gen in generations:
                print(f"  {gen.reference:<20} {gen.description}")
                print(f"    Retrieval: {gen.retrieval_reference}")
                print(f"    Prompt: {gen.system_prompt_version}")
                print(f"    Queries: {gen.successful_queries}/{gen.total_queries}")
                print(f"    Time: {gen.timestamp}")
                print()
        else:
            print("  No generation experiments found.\n")


def cmd_search(args: argparse.Namespace) -> None:
    """Search for experiments."""
    registry = ExperimentRegistry()

    # Build filter dict
    filters = {}
    if args.retriever_type:
        filters["retriever_type"] = args.retriever_type
    if args.chunking:
        filters["chunking"] = args.chunking == "yes"

    results = registry.search_experiments(
        run_num=None,  # Search all runs
        experiment_type=args.type,
        **filters,
    )

    print(f"\n{'=' * 80}")
    print(f"Search Results ({len(results)} matches)")
    print(f"{'=' * 80}\n")

    if results:
        for exp in results:
            print(f"  Run {exp.run_num} - {exp.reference}")
            print(f"    Type: {exp.type}")
            print(f"    Description: {exp.description}")
            if isinstance(exp, RetrievalRecord):
                print(f"    Retriever: {exp.retriever_type}, Chunking: {exp.chunking}")
            elif isinstance(exp, GenerationRecord):
                print(f"    Retrieval: {exp.retrieval_reference}")
                print(f"    Prompt: {exp.system_prompt_version}")
            print()
    else:
        print("  No matching experiments found.\n")


def cmd_new_run(args: argparse.Namespace) -> None:
    """Increment to new run number."""
    registry = ExperimentRegistry()
    new_run = registry.increment_run()
    print(f"\nIncremented to run {new_run}\n")


def cmd_current(args: argparse.Namespace) -> None:
    """Show current run number."""
    registry = ExperimentRegistry()
    current = registry.get_current_run()
    print(f"\nCurrent run: {current}\n")


def main() -> None:
    """Unified experiment CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified experiment CLI for WildFrostRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Retrieval command
    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval experiment")
    retrieval_parser.add_argument("--run", default="current", help="Run number (default: current)")
    retrieval_parser.add_argument(
        "--retriever",
        required=True,
        choices=[
            "vector",
            "fulltext",
            "bm25",
            "bm25_vector",
            "fulltext_vector",
            "bm25_fulltext_vector",
            "text2cypher",
        ],
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
    generation_parser.add_argument("--run", default="current", help="Run number (default: current)")
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
    list_parser.add_argument("--run", default="current", help="Run number (default: current)")
    list_parser.add_argument("--type", choices=["retrieval", "generation"], help="Filter by type")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search experiments")
    search_parser.add_argument("--type", choices=["retrieval", "generation"], help="Filter by type")
    search_parser.add_argument("--retriever-type", help="Filter by retriever type")
    search_parser.add_argument("--chunking", choices=["yes", "no"], help="Filter by chunking")

    # New run command
    subparsers.add_parser("new-run", help="Increment to next run number")

    # Current run command
    subparsers.add_parser("current", help="Show current run number")

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
        "new-run": cmd_new_run,
        "current": cmd_current,
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
