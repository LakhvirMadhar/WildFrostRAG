#!/usr/bin/env python3
"""Retriever Pipeline for WildFrostRAG.

This script runs different retrieval strategies and saves raw results for manual evaluation:
1. Loads query data from CSV
2. Tests various retrieval methods (vector, fulltext, BM25, hybrid combinations, text2cypher)
3. Saves raw retrieval results to structured output directories with run numbers
4. Results can be manually evaluated later using a GUI

Usage:
    python -m wildfrost_rag.cli.evaluate_retrievers --run-num 1 --retriever vector --chunking yes
    python -m wildfrost_rag.cli.evaluate_retrievers --run-num 1 --retriever bm25 --chunking no
    python -m wildfrost_rag.cli.evaluate_retrievers --run-num 1 --retriever text2cypher --chunking no
"""

import asyncio
import argparse
from pathlib import Path

from wildfrost_rag.core.logger import logger
from wildfrost_rag.core.config import DEFAULT_QUERIES_FILE, get_settings
from wildfrost_rag.clients.neo4j_driver import neo4j_driver
from wildfrost_rag.domain.prompt_name import PromptName
from wildfrost_rag.domain.retriever_type import RetrieverType
from wildfrost_rag.services.embeddings.embedder_type import EmbedderType
from wildfrost_rag.services.retrieval.retrieval_service import RetrievalService


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run different retrievers and save raw results")
    parser.add_argument("--run-num", type=int, required=True, help="Experiment run number")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=[member.value for member in RetrieverType],
        required=True,
        help="Retriever to run",
    )
    parser.add_argument(
        "--chunking",
        type=str,
        choices=["yes", "no"],
        default="no",
        help="Whether chunking was used during ingestion",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Human-readable description of this experiment",
    )
    parser.add_argument(
        "--text2cypher-prompt",
        type=str,
        default=PromptName.TEXT2CYPHER_PROMPT,
        help="Text2cypher prompt reference: 'name' (latest) or 'name:version'",
    )
    parser.add_argument(
        "--query-ids",
        type=str,
        help="Comma-separated query IDs to include (e.g., '1,5,10')",
    )
    parser.add_argument(
        "--exclude-query-ids",
        type=str,
        help="Comma-separated query IDs to exclude (e.g., '2,3,4')",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_QUERIES_FILE,
        help="Path to input CSV file with queries",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="hf",
        help="Embedding provider (e.g., 'hf', 'openai')",
    )
    parser.add_argument(
        "--queries-json",
        type=str,
        default=None,
        help="Path to queries JSON with doc_references for auto-annotation",
    )
    parser.add_argument(
        "--sw-query",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Remove stop words from queries (BM25 + fulltext)",
    )
    parser.add_argument(
        "--sw-docs",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Remove stop words from documents (BM25 only)",
    )
    return parser.parse_args()


def _parse_id_list(raw: str | None) -> list[int] | None:
    """Parse a comma-separated CLI argument into a list of ints."""
    if not raw:
        return None
    return [int(qid.strip()) for qid in raw.split(",")]


async def run(args: argparse.Namespace) -> None:
    """Run a retrieval experiment from a parsed Namespace."""
    settings = get_settings()
    settings.create_directories()

    service = RetrievalService()
    retriever_type = RetrieverType(args.retriever)
    embedder = EmbedderType(args.embedder)

    df = service.load_and_filter_queries(
        args.file, _parse_id_list(args.query_ids), _parse_id_list(args.exclude_query_ids)
    )
    chunking = args.chunking == "yes"

    queries_json = Path(args.queries_json) if args.queries_json else None

    with neo4j_driver() as driver:
        experiment = await service.run_experiment(
            driver=driver,
            df=df,
            retriever_type=retriever_type,
            run_num=args.run_num,
            chunking=chunking,
            k=args.k,
            description=args.description,
            embedder=embedder,
            sw_query=args.sw_query == "yes",
            sw_docs=args.sw_docs == "yes",
            text2cypher_prompt_name=args.text2cypher_prompt,
            queries_json_path=queries_json,
            dataset_path=args.file,
        )

        if experiment.results:
            logger.info(
                "Retriever run completed successfully! Results saved for manual evaluation."
            )
        else:
            logger.error("Retriever run failed.")

    logger.info("Neo4j driver closed")


async def main() -> None:
    """CLI entry point — parse args and run."""
    args = parse_args()
    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
