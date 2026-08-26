#!/usr/bin/env python3
"""Data ingestion pipeline for WildFrostRAG.

This script orchestrates the complete ETL pipeline:
1. Web scraping (sitemap + card pages)
2. HTML parsing & card data extraction
3. Data enrichment (tribe exclusivity)
4. Neo4j graph population
5. Document chunking
6. Document ingestion into Neo4j (text + metadata only, NO embeddings)

Note: Embeddings are added separately using scripts/add_embeddings.py

Usage:
    python -m scripts.ingest_data                    # Run full pipeline
    python -m scripts.ingest_data --skip-scrape      # Skip web scraping
    python -m scripts.ingest_data --skip-graph       # Skip graph creation
    python -m scripts.ingest_data --skip-vectors     # Skip document ingestion
    python -m scripts.ingest_data --no-chunking      # Ingest full documents (no splitting)
    python -m scripts.ingest_data --clear-db         # Clear database before running

All orchestration and business logic lives in GraphBuilderService
(wildfrost_rag.services.ingestion.graph_builder_service) - this script only parses
arguments, constructs the Neo4j driver, and hands off to the service.
"""

import argparse
import asyncio

from wildfrost_rag.neo4j_kg.driver import neo4j_driver
from wildfrost_rag.services.ingestion.graph_builder_service import GraphBuilderService


async def main() -> None:
    """Parse CLI arguments and run the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="WildFrostRAG data ingestion pipeline")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip web scraping stage (use existing HTML files)",
    )
    parser.add_argument(
        "--skip-graph", action="store_true", help="Skip Neo4j graph population stage"
    )
    parser.add_argument(
        "--skip-vectors", action="store_true", help="Skip vector store ingestion stage"
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear entire Neo4j database before starting (WARNING: destructive!)",
    )
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Skip HTML splitting (ingest full documents as single nodes)",
    )
    args = parser.parse_args()

    with neo4j_driver() as driver:
        service = GraphBuilderService(driver=driver)
        await service.run(
            skip_scrape=args.skip_scrape,
            skip_graph=args.skip_graph,
            skip_vectors=args.skip_vectors,
            clear_db=args.clear_db,
            no_chunking=args.no_chunking,
        )


if __name__ == "__main__":
    asyncio.run(main())
