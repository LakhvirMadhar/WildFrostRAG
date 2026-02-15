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
import argparse
import asyncio
import os
import json
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
from neo4j import GraphDatabase

# Import from src modules
from src.web_scraper.sitemap_scraper import scrape_multiple_links
from src.data_processing.cards import CardInfo, CardType
from src.data_processing.generate_schemas import generate_card_type_html_schema
from src.data_processing.enrichment import enrich_cards_with_tribes
from src.data_processing.html_splitter import process_html_files
from src.data_processing.stats import StatInfo
from src.data_processing.charms import CharmInfo
from src.data_processing.map import ZoneInfo, MapEventInfo, FightSlotInfo
from src.data_processing.shades import SummonInfo
from src.neo4j_kg.graph_builder import create_neo4j_data, clear_database
from src.neo4j_kg.stats import create_stats_from_parsed
from src.neo4j_kg.charms import create_charms_from_parsed, create_charm_tribe_relationships
from src.neo4j_kg.map import create_map_graph
from src.neo4j_kg.fights import create_fight_enemy_relationships
from src.neo4j_kg.shades import create_summon_relationships
from src.scraping.wiki_scraper import scrape_wiki_page, clean_name_for_url
from src.scraping.domain_scrapers import (
    scrape_leaders, scrape_stats, scrape_charms,
    scrape_shades, scrape_map, scrape_fight_pages
)
from src.neo4j_kg.vector_store import (
    ingest_documents_into_neo4j,
    link_documents_to_cards,
    link_documents_to_crowns,
    link_documents_to_stats,
    link_documents_to_charms,
    link_documents_to_shades,
    link_documents_to_map,
    link_documents_to_fights
)
from src.neo4j_kg.neo4j_indexes import (
    create_fulltext_index,
    wait_for_index_population
)
from src.utils.config import settings
from src.utils.logger import logger


@dataclass
class PipelineData:
    """All data collected during Stage 1 (scraping).

    Replaces the positional tuple that previously grew with every new data source.
    New scraping targets just add a field here — no tuple unpacking to update.
    """
    cards: List[CardInfo] = field(default_factory=list)
    stats: List[StatInfo] = field(default_factory=list)
    charms: List[CharmInfo] = field(default_factory=list)
    summons: List[SummonInfo] = field(default_factory=list)
    zones: List[ZoneInfo] = field(default_factory=list)
    map_events: List[MapEventInfo] = field(default_factory=list)
    fight_slots: List[FightSlotInfo] = field(default_factory=list)
    fight_page_mapping: dict[str, str] = field(default_factory=dict)
    fight_enemies: dict[str, list[str]] = field(default_factory=dict)


async def stage_1_scrape_cards() -> PipelineData:
    """
    Stage 1: Web Scraping

    Scrapes card data from the Wildfrost wiki and saves HTML files.

    Returns:
        PipelineData containing all scraped and parsed data
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

    # Create CardInfo objects for all cards (excluding leaders - handled separately)
    card_infos = []
    for card_type, cards in card_type_schema.items():
        if card_type == 'leaders':
            continue  # Leaders handled by scrape_leaders()

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

    # Parse HTML and create CardInfo objects (may return multiple for multi-phase cards)
    all_cards: List[CardInfo] = []
    successful_pages = 0

    for card_info, html in tqdm(zip(card_infos, html_outputs), total=len(card_infos), desc="Parsing HTML", unit="page"):
        if html is None:
            continue

        # Use parse_html_multi_phase which handles multi-phase cards
        parsed_cards = CardInfo.parse_html_multi_phase(
            html=html,
            card_type=card_info.card_type,
            card_url=card_info.card_url
        )

        if parsed_cards:
            successful_pages += 1
            # Save HTML for each parsed card
            for card in parsed_cards:
                card.save_html()
            all_cards.extend(parsed_cards)

    logger.info(f"Successfully scraped {successful_pages}/{len(card_infos)} pages")
    logger.info(f"Created {len(all_cards)} CardInfo objects (including multi-phase cards)")

    # Scrape leaders (separate page with different structure)
    leader_cards = await scrape_leaders()
    all_cards.extend(leader_cards)

    # Scrape crowns page (for Document node, Crown nodes are hardcoded)
    await scrape_wiki_page("Crowns", "crowns")

    # Scrape stats page
    stats = await scrape_stats()

    # Scrape charms page
    charms = await scrape_charms()

    # Scrape shades page (summoning relationships)
    summons = await scrape_shades()

    # Scrape map page
    zones, map_events, fight_slots, fight_page_mapping = await scrape_map()

    # Scrape individual fight pages and parse enemy names
    fight_enemies = {}
    if fight_page_mapping:
        fight_enemies = await scrape_fight_pages(fight_page_mapping)

    logger.info(f"Total cards: {len(all_cards)}")
    return PipelineData(
        cards=all_cards,
        stats=stats,
        charms=charms,
        summons=summons,
        zones=zones,
        map_events=map_events,
        fight_slots=fight_slots,
        fight_page_mapping=fight_page_mapping,
        fight_enemies=fight_enemies,
    )


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


def stage_3_populate_graph(data: PipelineData) -> None:
    """
    Stage 3: Neo4j Graph Population

    Creates nodes and relationships in Neo4j knowledge graph.

    Args:
        data: PipelineData containing all scraped data to populate the graph
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: NEO4J GRAPH POPULATION")
    logger.info("=" * 60)

    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
    )
    try:
        with driver.session() as session:
            # Create Stat nodes FIRST (cards need them for HAS_STAT relationships)
            if data.stats:
                count = session.execute_write(create_stats_from_parsed, data.stats)
                logger.info(f"Created {count} Stat nodes")

            # Create Charm nodes (tribe relationships after create_neo4j_data, which creates Tribes)
            if data.charms:
                charm_count = session.execute_write(create_charms_from_parsed, data.charms)
                logger.info(f"Created {charm_count} Charm nodes")

        # Convert CardInfo objects to dictionaries
        cards_dict_data = [card.to_dict() for card in data.cards]
        logger.info(f"Ingesting {len(cards_dict_data)} cards into Neo4j graph...")

        # Creates Tribes, Cards, Crowns, Stats relationships, etc.
        # NOTE: create_neo4j_data still manages its own driver internally
        create_neo4j_data(cards_dict_data)

        # Charm-Tribe relationships (Tribes must exist first)
        # Map graph (Map, Zone, MapEvent, Fight nodes + relationships)
        with driver.session() as session:
            if data.charms:
                tribe_count = session.execute_write(create_charm_tribe_relationships, data.charms)
                logger.info(f"Created {tribe_count} Charm-Tribe relationships")

            if data.zones or data.map_events or data.fight_slots:
                counts = session.execute_write(create_map_graph, data.zones, data.map_events, data.fight_slots, data.fight_page_mapping)
                logger.info(f"Map graph created: {counts}")

            if data.fight_enemies and data.fight_page_mapping:
                enemy_count = session.execute_write(create_fight_enemy_relationships, data.fight_enemies, data.fight_page_mapping)
                logger.info(f"Created {enemy_count} Fight-Enemy relationships")

            if data.summons:
                summon_count = session.execute_write(create_summon_relationships, data.summons)
                logger.info(f"Created {summon_count} SUMMONS relationships")
    finally:
        driver.close()

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

    # Single driver for all document operations
    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
    )
    try:
        with driver.session() as session:
            # Ingest into Neo4j (no embeddings)
            logger.info("Ingesting documents into Neo4j...")
            ingest_documents_into_neo4j(
                session=session,
                document_chunks=all_document_chunks,
                url_lookup=url_lookup
            )

            # Create full-text search index
            logger.info("Creating full-text search index...")
            create_fulltext_index(
                session=session,
                index_name=settings.fulltext_index_name,
                node_label="Document",
                text_property="text"
            )

            # Wait for index to populate
            wait_for_index_population(seconds=5)

            # Link Document nodes to Card nodes in the knowledge graph
            logger.info("Linking documents to cards in knowledge graph...")
            link_count = link_documents_to_cards(session)
            logger.info(f"Linked {link_count} documents to cards")

            # Link Document nodes to Crown nodes
            logger.info("Linking documents to crowns in knowledge graph...")
            crown_link_count = link_documents_to_crowns(session)
            logger.info(f"Linked {crown_link_count} documents to crowns")

            # Link Document nodes to Stat nodes
            logger.info("Linking documents to stats in knowledge graph...")
            stat_link_count = link_documents_to_stats(session)
            logger.info(f"Linked {stat_link_count} documents to stats")

            # Link Document nodes to Charm nodes
            logger.info("Linking documents to charms in knowledge graph...")
            charm_link_count = link_documents_to_charms(session)
            logger.info(f"Linked {charm_link_count} documents to charms")

            # Link shade Card nodes to Shades.html overview Document
            logger.info("Linking shade cards to Shades overview document...")
            shade_link_count = link_documents_to_shades(session)
            logger.info(f"Linked {shade_link_count} shade cards to Shades overview document")

            # Link Document nodes to Map nodes (Map, Zone, MapEvent)
            logger.info("Linking documents to map nodes in knowledge graph...")
            map_link_count = link_documents_to_map(session)
            logger.info(f"Linked {map_link_count} documents to map nodes")

            # Link Fight nodes to their individual fight page Documents
            logger.info("Linking fight nodes to fight page documents...")
            fight_link_count = link_documents_to_fights(session)
            logger.info(f"Linked {fight_link_count} fight nodes to their documents")

    finally:
        driver.close()

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
    pipeline_data = await stage_1_scrape_cards()

    # Stage 2: Enrich
    stage_2_enrich_data(pipeline_data.cards)

    # Stage 3: Graph
    if not args.skip_graph:
        stage_3_populate_graph(pipeline_data)
    else:
        logger.info("Skipping graph population (--skip-graph flag set)")

    # Stage 4: Documents
    if not args.skip_vectors:
        # If --no-chunking is passed, split_text should be False
        split_text = not args.no_chunking
        stage_4_document_ingestion(pipeline_data.cards, split_text=split_text)
    else:
        logger.info("Skipping document ingestion (--skip-vectors flag set)")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
