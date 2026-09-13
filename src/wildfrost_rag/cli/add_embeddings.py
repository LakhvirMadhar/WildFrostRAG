#!/usr/bin/env python3
"""Add embedding properties to existing Document nodes in Neo4j.

This script allows adding embeddings from multiple providers to the same
Document nodes, enabling multi-embedder testing without data duplication.

Usage:
    poetry run python -m wildfrost_rag.cli.add_embeddings --embedder hf
    poetry run python -m wildfrost_rag.cli.add_embeddings --embedder openai
"""

import argparse
import asyncio

from wildfrost_rag.clients.neo4j_driver import neo4j_driver
from wildfrost_rag.core.logger import logger
from wildfrost_rag.services.embeddings.embedder_type import EmbedderType
from wildfrost_rag.services.embeddings.embedding_service import EmbeddingService


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add embedding properties to existing Document nodes"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        choices=[member.value for member in EmbedderType],
        required=True,
        help="Embedder provider to use",
    )
    return parser.parse_args()


async def main() -> None:
    """Add embeddings from a specified provider to Document nodes, and build its vector index."""
    args = parse_args()
    embedder = EmbedderType(args.embedder)
    service = EmbeddingService()

    with neo4j_driver() as driver:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully")

        updated = await service.add_embeddings(driver, embedder)
        logger.info(f"Added embeddings to {updated} documents")

        service.create_vector_index(driver, embedder)
        logger.info(f"Vector index ready for embedder '{embedder.value}'")


if __name__ == "__main__":
    asyncio.run(main())
