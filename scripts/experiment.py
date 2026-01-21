#!/usr/bin/env python3
"""
Unified experiment CLI - MLflow-like interface for WildFrostRAG.

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
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment_tracker import ExperimentRegistry
from src.utils.logger import logger


def cmd_retrieval(args):
    """Run a retrieval experiment."""
    registry = ExperimentRegistry()
    run_num = args.run if args.run != "current" else registry.get_current_run()

    # Build command for evaluate_retrievers.py
    cmd = [
        sys.executable, "-m", "scripts.evaluate_retrievers",
        "--run-num", str(run_num),
        "--retriever", args.retriever,
        "--chunking", args.chunking,
    ]

    if args.description:
        cmd.extend(["--description", args.description])

    if args.retriever == "text2cypher" and args.text2cypher_prompt:
        cmd.extend(["--text2cypher-prompt", args.text2cypher_prompt])

    if args.query_ids:
        cmd.extend(["--query-ids", args.query_ids])

    if args.exclude_query_ids:
        cmd.extend(["--exclude-query-ids", args.exclude_query_ids])

    if args.k != 10:  # Only add if non-default
        cmd.extend(["--k", str(args.k)])

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def cmd_generation(args):
    """Run a generation experiment."""
    registry = ExperimentRegistry()
    run_num = args.run if args.run != "current" else registry.get_current_run()

    # Resolve retrieval reference
    retrieval_ref = registry.resolve_retrieval_reference(run_num, args.retrieval)
    if not retrieval_ref:
        logger.error(f"Could not resolve retrieval reference: {args.retrieval}")
        logger.info(f"Available retrievals for run {run_num}:")
        for ret in registry.list_retrievals(run_num):
            logger.info(f"  - {ret['reference']}: {ret.get('description', '')}")
        sys.exit(1)

    # Build command for run_llm_generation.py
    cmd = [
        sys.executable, "-m", "scripts.run_llm_generation",
        "--run-num", str(run_num),
        "--retrieval-reference", retrieval_ref,
        "--system-prompt", args.prompt,
    ]

    if args.description:
        cmd.extend(["--description", args.description])

    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])

    if args.query_ids:
        cmd.extend(["--query-ids", args.query_ids])

    if args.exclude_query_ids:
        cmd.extend(["--exclude-query-ids", args.exclude_query_ids])

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def cmd_list(args):
    """List experiments."""
    registry = ExperimentRegistry()
    run_num = args.run if args.run != "current" else registry.get_current_run()

    print(f"\n{'='*80}")
    print(f"Experiments for Run {run_num}")
    print(f"{'='*80}\n")

    if args.type in [None, "retrieval"]:
        print("RETRIEVAL EXPERIMENTS")
        print("-" * 80)
        retrievals = registry.list_retrievals(run_num)
        if retrievals:
            for ret in retrievals:
                print(f"  {ret['reference']:<20} {ret.get('description', '')}")
                print(f"    Retriever: {ret.get('retriever_type')}, Chunking: {ret.get('chunking')}")
                print(f"    Queries: {ret.get('successful_queries')}/{ret.get('total_queries')}")
                print(f"    Time: {ret.get('timestamp')}")
                print()
        else:
            print("  No retrieval experiments found.\n")

    if args.type in [None, "generation"]:
        print("GENERATION EXPERIMENTS")
        print("-" * 80)
        generations = registry.list_generations(run_num)
        if generations:
            for gen in generations:
                print(f"  {gen['reference']:<20} {gen.get('description', '')}")
                print(f"    Retrieval: {gen.get('retrieval_reference')}")
                print(f"    Prompt: {gen.get('system_prompt_version')}")
                print(f"    Queries: {gen.get('successful_queries')}/{gen.get('total_queries')}")
                print(f"    Time: {gen.get('timestamp')}")
                print()
        else:
            print("  No generation experiments found.\n")


def cmd_search(args):
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
        **filters
    )

    print(f"\n{'='*80}")
    print(f"Search Results ({len(results)} matches)")
    print(f"{'='*80}\n")

    if results:
        for exp in results:
            print(f"  Run {exp['run_num']} - {exp['reference']}")
            print(f"    Type: {exp['type']}")
            print(f"    Description: {exp.get('description', '')}")
            if exp['type'] == 'retrieval':
                print(f"    Retriever: {exp.get('retriever_type')}, Chunking: {exp.get('chunking')}")
            elif exp['type'] == 'generation':
                print(f"    Retrieval: {exp.get('retrieval_reference')}")
                print(f"    Prompt: {exp.get('system_prompt_version')}")
            print()
    else:
        print("  No matching experiments found.\n")


def cmd_new_run(args):
    """Increment to new run number."""
    registry = ExperimentRegistry()
    new_run = registry.increment_run()
    print(f"\nIncremented to run {new_run}\n")


def cmd_current(args):
    """Show current run number."""
    registry = ExperimentRegistry()
    current = registry.get_current_run()
    print(f"\nCurrent run: {current}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment CLI for WildFrostRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Retrieval command
    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval experiment")
    retrieval_parser.add_argument("--run", default="current", help="Run number (default: current)")
    retrieval_parser.add_argument("--retriever", required=True,
                                  choices=["vector", "fulltext", "bm25", "bm25_vector", "fulltext_vector",
                                          "bm25_fulltext_vector", "text2cypher"],
                                  help="Retriever type")
    retrieval_parser.add_argument("--chunking", choices=["yes", "no"], default="no", help="Chunking enabled")
    retrieval_parser.add_argument("--description", default="", help="Experiment description")
    retrieval_parser.add_argument("--text2cypher-prompt", default="TEXT2CYPHER_PROMPT_V1",
                                  help="Text2Cypher prompt (for text2cypher retriever)")
    retrieval_parser.add_argument("--query-ids", help="Comma-separated query IDs to include (e.g., '1,5,10')")
    retrieval_parser.add_argument("--exclude-query-ids", help="Comma-separated query IDs to exclude")
    retrieval_parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve per query (default: 10)")

    # Generation command
    generation_parser = subparsers.add_parser("generation", help="Run generation experiment")
    generation_parser.add_argument("--run", default="current", help="Run number (default: current)")
    generation_parser.add_argument("--retrieval", required=True,
                                   help="Retrieval reference (e.g., 'bm25/001' or 'latest/bm25')")
    generation_parser.add_argument("--prompt", required=True, help="System prompt name")
    generation_parser.add_argument("--description", default="", help="Experiment description")
    generation_parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    generation_parser.add_argument("--query-ids", help="Comma-separated query IDs to include")
    generation_parser.add_argument("--exclude-query-ids", help="Comma-separated query IDs to exclude")

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
