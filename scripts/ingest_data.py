#!/usr/bin/env python3
"""
Data ingestion pipeline for WildFrostRAG.

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
"""
import re
import argparse
import asyncio
import os
import json
from typing import List
from tqdm import tqdm
from neo4j import GraphDatabase

# Import from src modules
from src.web_scraper.sitemap_scraper import scrape_multiple_links
from src.data_processing.cards import CardInfo, CardType
from src.data_processing.generate_schemas import generate_card_type_html_schema
from src.data_processing.enrichment import enrich_cards_with_tribes
from src.data_processing.html_splitter import process_html_files
from src.neo4j_kg.neo4j_utils import create_neo4j_data, clear_database
from src.neo4j_kg.vector_store import (
    ingest_documents_into_neo4j,
    link_documents_to_cards
)
from src.neo4j_kg.neo4j_indexes import (
    create_fulltext_index,
    wait_for_index_population
)
from src.utils.config import settings
from src.utils.logger import logger


def clean_name_for_url(name: str) -> str:
    """Clean card name for use in URLs by replacing spaces with underscores."""
    return re.sub(r'\s+', '_', name)


async def stage_1_scrape_cards() -> List[CardInfo]:
    """
    Stage 1: Web Scraping

    Scrapes card data from the Wildfrost wiki and saves HTML files.

    Returns:
        List of CardInfo objects with HTML content populated
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: WEB SCRAPING")
    logger.info("=" * 60)

    # Generate card type schema
    logger.info("Generating card type schema...")
    card_type_schema = generate_card_type_html_schema()

    # Save schema to file
    schema_path = settings.schemas_dir / 'card_type_schema.json'
    settings.schemas_dir.mkdir(parents=True, exist_ok=True)
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(card_type_schema, f, indent=4)
    logger.info(f"Schema saved to {schema_path}")

    # Create CardInfo objects for all cards
    card_infos = []
    for card_type, cards in card_type_schema.items():
        if card_type == 'leaders':
            # Skip leaders for now (they need special handling)
            continue

        for card_name in cards:
            cleaned_name = clean_name_for_url(card_name)
            card_info = CardInfo(
                card_name=card_name,
                card_type=CardType(card_type),
                card_url=f'{settings.wildfrost_wiki_base_url}/{cleaned_name}'
            )
            card_infos.append(card_info)

    logger.info(f"Created {len(card_infos)} CardInfo objects")

    # Scrape all card pages
    urls = [card.card_url for card in card_infos]
    logger.info(f"Scraping {len(urls)} card pages...")

    html_outputs = await scrape_multiple_links(
        urls,
        max_concurrent=settings.max_concurrent_requests
    )

    # Attach HTML to CardInfo objects and save
    successful_count = 0
    # logger.info("Parsing and saving HTML content...") # Using tqdm desc instead
    for card_info, html in tqdm(zip(card_infos, html_outputs), total=len(card_infos), desc="Parsing HTML", unit="card"):
        card_info.card_html = html
        if card_info.card_html is not None:
            card_info.save_html()
            card_info.parse_html()
            successful_count += 1

    logger.info(f"Successfully scraped and parsed {successful_count}/{len(card_infos)} cards")

    return card_infos


def stage_2_enrich_data(card_infos: List[CardInfo]) -> None:
    """
    Stage 2: Data Enrichment

    Enriches card data with tribe exclusivity information.

    Args:
        card_infos: List of CardInfo objects to enrich (modified in-place)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: DATA ENRICHMENT")
    logger.info("=" * 60)

    enrich_cards_with_tribes(
        card_infos=card_infos,
        companions_url=settings.companions_page_url,
        items_url=settings.items_page_url
    )


def stage_3_populate_graph(card_infos: List[CardInfo]) -> None:
    """
    Stage 3: Neo4j Graph Population

    Creates nodes and relationships in Neo4j knowledge graph.

    Args:
        card_infos: List of CardInfo objects to ingest
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: NEO4J GRAPH POPULATION")
    logger.info("=" * 60)

    # Convert CardInfo objects to dictionaries
    cards_dict_data = [card.to_dict() for card in card_infos]
    logger.info(f"Ingesting {len(cards_dict_data)} cards into Neo4j graph...")

    create_neo4j_data(cards_dict_data)
    logger.info("Graph population complete")


def stage_4_document_ingestion(card_infos: List[CardInfo], split_text: bool = True) -> None:
    """
    Stage 4: Document Ingestion

    Chunks HTML documents and ingests into Neo4j as Document nodes.
    Creates full-text search index. Does NOT generate embeddings.
    Use add_embeddings.py separately to add embeddings.

    Args:
        card_infos: List of CardInfo objects (used to find HTML files)
        split_text: If True, splits documents into chunks. If False, ingests full documents.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: DOCUMENT INGESTION")
    if not split_text:
        logger.info("(Full Document Mode: No Chunking)")
    logger.info("=" * 60)

    # Collect all HTML file paths
    logger.info("Collecting HTML files...")
    all_html_filepaths = []
    for root, dirs, files in os.walk(settings.structured_outputs_dir):
        for file in files:
            if file.endswith('.html'):
                filepath = os.path.join(root, file)
                all_html_filepaths.append(filepath)

    logger.info(f"Found {len(all_html_filepaths)} HTML files to process")

    # Chunk the HTML documents
    logger.info(f"Processing HTML documents (split_text={split_text})...")
    all_document_chunks = process_html_files(all_html_filepaths, split_text=split_text)
    logger.info(f"Created {len(all_document_chunks)} document objects")

    # Build filename -> URL lookup from card_infos
    url_lookup = {
        f"{card.sanitized_name()}.html": card.card_url
        for card in card_infos
    }
    logger.info(f"Built URL lookup with {len(url_lookup)} entries")

    # Ingest into Neo4j (no embeddings)
    logger.info("Ingesting documents into Neo4j...")
    ingest_documents_into_neo4j(
        document_chunks=all_document_chunks,
        url_lookup=url_lookup
    )

    # Create full-text search index
    logger.info("Creating full-text search index...")
    create_fulltext_index(
        index_name=settings.fulltext_index_name,
        node_label="Document",
        text_property="text"
    )

    # Wait for index to populate
    wait_for_index_population(seconds=5)

    # Link Document nodes to Card nodes in the knowledge graph
    logger.info("Linking documents to cards in knowledge graph...")
    link_count = link_documents_to_cards()
    logger.info(f"Linked {link_count} documents to cards")

    logger.info("Document ingestion complete")


async def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="WildFrostRAG data ingestion pipeline"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip web scraping stage (use existing HTML files)"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip Neo4j graph population stage"
    )
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Skip vector store ingestion stage"
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear entire Neo4j database before starting (WARNING: destructive!)"
    )
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Skip HTML splitting (ingest full documents as single nodes)"
    )
    args = parser.parse_args()

    logger.info("Starting WildFrostRAG data ingestion pipeline")
    logger.info(f"Configuration: {settings.model_dump()}")

    # Clear database if requested
    if args.clear_db:
        logger.warning("⚠️  CLEARING ENTIRE NEO4J DATABASE ⚠️")

        driver = GraphDatabase.driver(
            settings.neo4j_uri.get_secret_value(),
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
        )
        try:
            with driver.session() as session:
                session.execute_write(clear_database)
            logger.info("✅ Database cleared successfully")
        finally:
            driver.close()

    # Ensure directories exist
    settings.create_directories()

    # Stage 1: Scrape (or load existing data)
    if args.skip_scrape:
        logger.info("Skipping web scraping (--skip-scrape flag set)")
        # TODO: Load CardInfo objects from existing HTML files
        # For now, we'll need to scrape to get the CardInfo objects
        logger.warning("Loading from existing files not yet implemented, running scrape anyway")
        card_infos = await stage_1_scrape_cards()
    else:
        card_infos = await stage_1_scrape_cards()

    # Stage 2: Enrich
    stage_2_enrich_data(card_infos)

    # Stage 3: Graph
    if not args.skip_graph:
        stage_3_populate_graph(card_infos)
    else:
        logger.info("Skipping graph population (--skip-graph flag set)")

    # Stage 4: Documents
    if not args.skip_vectors:
        # If --no-chunking is passed, split_text should be False
        split_text = not args.no_chunking
        stage_4_document_ingestion(card_infos, split_text=split_text)
    else:
        logger.info("Skipping document ingestion (--skip-vectors flag set)")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
