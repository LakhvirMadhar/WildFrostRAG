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
"""

import argparse
import asyncio
import os
import json
from dataclasses import dataclass, field
from typing import Any
from tqdm import tqdm
from neo4j import GraphDatabase, Session

from web_scraper.sitemap_scraper import scrape_multiple_links
from data_processing.cards import CardInfo, CardType
from data_processing.generate_schemas import generate_card_type_html_schema
from data_processing.enrichment import enrich_cards_with_tribes
from data_processing.html_splitter import process_html_files
from data_processing.stats import StatInfo
from data_processing.keywords import KeywordInfo
from data_processing.bling import EnemyBlingDrop, ShopListing
from data_processing.bells import BellInfo
from data_processing.charms import CharmInfo
from data_processing.map import ZoneInfo, MapEventInfo, FightSlotInfo
from data_processing.shades import SummonInfo
from neo4j_kg.graph_builder import create_neo4j_data, create_url_nodes, clear_database
from neo4j_kg.stats import create_stats_from_parsed, add_keyword_label_to_stats
from neo4j_kg.keywords import (
    create_keywords_from_parsed,
    create_card_keyword_relationships,
    create_charm_keyword_relationships,
)
from neo4j_kg.charms import create_charms_from_parsed, create_charm_tribe_relationships
from neo4j_kg.map import create_map_graph
from neo4j_kg.fights import create_fight_enemy_relationships
from neo4j_kg.shades import create_summon_relationships
from neo4j_kg.bling import (
    create_bling_and_shops,
    create_drops_bling_relationships,
    create_shop_sells_relationships,
)
from neo4j_kg.bells import create_bells_from_parsed, create_bell_relationships
from scraping.wiki_scraper import clean_name_for_url
from scraping.domain_scrapers import (
    scrape_leaders,
    scrape_stats,
    scrape_keywords,
    scrape_charms,
    scrape_individual_charm_pages,
    scrape_individual_stat_pages,
    scrape_individual_bell_pages,
    scrape_bling,
    scrape_shop,
    scrape_clunker_prices,
    scrape_bells,
    scrape_shades,
    scrape_map,
    scrape_fight_pages,
    scrape_crowns,
    scrape_getting_started,
)
from neo4j_kg.vector_store import (
    ingest_documents_into_neo4j,
    link_documents_to_cards,
    link_documents_to_crowns,
    link_documents_to_stats,
    link_documents_to_charms,
    link_documents_to_shades,
    link_documents_to_map,
    link_documents_to_fights,
    link_documents_to_shops,
    link_documents_to_bling,
    link_documents_to_bells,
)
from neo4j_kg.neo4j_indexes import create_fulltext_index, wait_for_index_population
from utils.config import settings
from utils.logger import logger


@dataclass
class PipelineData:
    """All data collected during Stage 1 (scraping).

    Replaces the positional tuple that previously grew with every new data source.
    New scraping targets just add a field here — no tuple unpacking to update.
    """

    cards: list[CardInfo] = field(default_factory=list)
    stats: list[StatInfo] = field(default_factory=list)
    keywords: list[KeywordInfo] = field(default_factory=list)
    bling_drops: list[EnemyBlingDrop] = field(default_factory=list)
    woolly_snail_listings: list[ShopListing] = field(default_factory=list)
    charm_merchant_listings: list[ShopListing] = field(default_factory=list)
    clunker_prices: list[ShopListing] = field(default_factory=list)
    bells: list[BellInfo] = field(default_factory=list)
    charms: list[CharmInfo] = field(default_factory=list)
    summons: list[SummonInfo] = field(default_factory=list)
    zones: list[ZoneInfo] = field(default_factory=list)
    map_events: list[MapEventInfo] = field(default_factory=list)
    fight_slots: list[FightSlotInfo] = field(default_factory=list)
    fight_page_mapping: dict[str, str] = field(default_factory=dict)
    fight_enemies: dict[str, list[str]] = field(default_factory=dict)
    page_urls: dict[str, str] = field(default_factory=dict)  # filename -> wiki URL


def _generate_schema() -> dict[str, list[str]]:
    """Generate and save card type schema from wiki."""
    logger.info("Generating card type schema...")
    card_type_schema = generate_card_type_html_schema()

    schema_path = settings.schemas_dir / "card_type_schema.json"
    settings.schemas_dir.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(card_type_schema, f, indent=4)
    logger.info(f"Schema saved to {schema_path}")
    return card_type_schema


def _create_card_infos(card_type_schema: dict[str, list[str]]) -> list[CardInfo]:
    """Create CardInfo objects from schema (excluding leaders)."""
    card_infos = []
    for card_type, cards in card_type_schema.items():
        if card_type == "leaders":
            continue

        for card_name in cards:
            cleaned_name = clean_name_for_url(card_name)
            card_infos.append(
                CardInfo(
                    card_name=card_name,
                    card_type=CardType.from_schema_key(card_type),
                    url=f"{settings.wildfrost_wiki_base_url}/{cleaned_name}",
                )
            )

    logger.info(f"Created {len(card_infos)} CardInfo objects")
    return card_infos


def _load_cached_cards(
    card_infos: list[CardInfo],
) -> tuple[list[CardInfo], list[CardInfo], int]:
    """Load card pages from cache, return parsed cards and cards needing scraping.

    Returns:
        Tuple of (parsed cards, cards to scrape, successful page count).
    """
    all_cards: list[CardInfo] = []
    successful_pages = 0
    cards_to_scrape = []

    for card_info in card_infos:
        html_path = card_info.save_path()
        if os.path.exists(html_path):
            with open(html_path, encoding="utf-8") as f:
                html = f.read()
            parsed_cards = CardInfo.parse_html_multi_phase(
                html=html, card_type=card_info.card_type, url=card_info.url
            )
            if parsed_cards:
                successful_pages += 1
                all_cards.extend(parsed_cards)
        else:
            cards_to_scrape.append(card_info)

    logger.info(f"Loaded {successful_pages} card pages from cache")
    return all_cards, cards_to_scrape, successful_pages


async def _scrape_missing_cards(
    cards_to_scrape: list[CardInfo],
) -> tuple[list[CardInfo], int]:
    """Scrape and parse cards not found in cache.

    Returns:
        Tuple of (newly parsed cards, successful page count).
    """
    logger.info(f"Scraping {len(cards_to_scrape)} missing card pages...")
    urls = [card.url for card in cards_to_scrape]
    html_outputs = await scrape_multiple_links(
        urls, max_concurrent=settings.max_concurrent_requests
    )

    new_cards: list[CardInfo] = []
    successful = 0
    for card_info, html in tqdm(
        zip(cards_to_scrape, html_outputs, strict=False),
        total=len(cards_to_scrape),
        desc="Parsing HTML",
        unit="page",
    ):
        if html is None:
            continue
        parsed_cards = CardInfo.parse_html_multi_phase(
            html=html, card_type=card_info.card_type, url=card_info.url
        )
        if parsed_cards:
            successful += 1
            for card in parsed_cards:
                card.save_html()
            new_cards.extend(parsed_cards)

    return new_cards, successful


async def _load_card_pages(
    skip_scrape: bool,
) -> tuple[list[CardInfo], dict[str, list[str]]]:
    """Generate card type schema and load/scrape individual card pages.

    Returns:
        Tuple of (all parsed CardInfo objects, card_type_schema dict).
    """
    card_type_schema = _generate_schema()
    card_infos = _create_card_infos(card_type_schema)
    all_cards, cards_to_scrape, successful_pages = _load_cached_cards(card_infos)

    if cards_to_scrape and not skip_scrape:
        new_cards, new_successes = await _scrape_missing_cards(cards_to_scrape)
        all_cards.extend(new_cards)
        successful_pages += new_successes
    elif cards_to_scrape:
        logger.warning(f"{len(cards_to_scrape)} card HTML files not found in cache (--skip-scrape)")

    logger.info(
        f"Total: {successful_pages}/{len(card_infos)} card pages, {len(all_cards)} CardInfo objects"
    )
    return all_cards, card_type_schema


async def _scrape_domain_pages(card_type_schema: dict[str, list[str]]) -> PipelineData:
    """Scrape all domain pages (leaders, stats, keywords, shops, etc.).

    Collects parsed data and page URLs from each scraper into PipelineData.

    Args:
        card_type_schema: Schema dict used to extract boss/miniboss names for bling scraping
    """
    page_urls = {}

    leader_cards, leader_urls = await scrape_leaders()
    page_urls.update(leader_urls)

    _, crowns_urls = await scrape_crowns()
    page_urls.update(crowns_urls)

    _, getting_started_urls = await scrape_getting_started()
    page_urls.update(getting_started_urls)

    stats, stats_urls = await scrape_stats()
    page_urls.update(stats_urls)

    # Scrape individual stat pages for per-stat Documents (detailed mechanics)
    individual_stat_urls = await scrape_individual_stat_pages(stats)
    page_urls.update(individual_stat_urls)

    keywords, keywords_urls = await scrape_keywords()
    page_urls.update(keywords_urls)

    boss_names = card_type_schema.get("bosses", [])
    miniboss_names = card_type_schema.get("minibosses", [])
    bling_drops, bling_urls = await scrape_bling(boss_names, miniboss_names)
    page_urls.update(bling_urls)

    woolly_snail_listings, woolly_urls = await scrape_shop("The_Woolly_Snail", "shops")
    page_urls.update(woolly_urls)
    charm_merchant_listings, charm_merchant_urls = await scrape_shop("Charm_Merchant", "shops")
    page_urls.update(charm_merchant_urls)

    clunker_prices, clunker_urls = await scrape_clunker_prices()
    page_urls.update(clunker_urls)

    bells, bell_urls = await scrape_bells()
    page_urls.update(bell_urls)

    # Scrape individual bell pages for per-bell Documents
    individual_bell_urls = await scrape_individual_bell_pages(bells)
    page_urls.update(individual_bell_urls)

    charms, charm_urls = await scrape_charms()
    page_urls.update(charm_urls)

    # Scrape individual charm pages for per-charm Documents (Strategy sections, etc.)
    individual_charm_urls = await scrape_individual_charm_pages(charms)
    page_urls.update(individual_charm_urls)

    summons, shades_urls = await scrape_shades()
    page_urls.update(shades_urls)

    zones, map_events, fight_slots, fight_page_mapping, map_urls = await scrape_map()
    page_urls.update(map_urls)

    fight_enemies: dict[str, list[str]] = {}
    if fight_page_mapping:
        fight_enemies, fight_urls = await scrape_fight_pages(fight_page_mapping)
        page_urls.update(fight_urls)

    return PipelineData(
        cards=leader_cards,
        stats=stats,
        keywords=keywords,
        bling_drops=bling_drops,
        woolly_snail_listings=woolly_snail_listings,
        charm_merchant_listings=charm_merchant_listings,
        clunker_prices=clunker_prices,
        bells=bells,
        charms=charms,
        summons=summons,
        zones=zones,
        map_events=map_events,
        fight_slots=fight_slots,
        fight_page_mapping=fight_page_mapping,
        fight_enemies=fight_enemies,
        page_urls=page_urls,
    )


async def stage_1_scrape_cards(skip_scrape: bool = False) -> PipelineData:
    """Stage 1: Data Collection.

    Loads card data from cache if available, otherwise scrapes from wiki.
    When skip_scrape=True, only loads from cache (no network requests).

    Args:
        skip_scrape: If True, only use cached HTML files (no web requests)

    Returns:
        PipelineData containing all scraped and parsed data
    """
    logger.info("=" * 60)
    if skip_scrape:
        logger.info("STAGE 1: LOADING FROM CACHE (--skip-scrape)")
    else:
        logger.info("STAGE 1: WEB SCRAPING")
    logger.info("=" * 60)

    all_cards, card_type_schema = await _load_card_pages(skip_scrape)

    # Domain pages (leaders, stats, keywords, shops, etc.)
    pipeline_data = await _scrape_domain_pages(card_type_schema)

    # Merge card pages into pipeline data
    # Leader cards come from _scrape_domain_pages, all other cards from _load_card_pages
    pipeline_data.cards = all_cards + pipeline_data.cards

    # Card page URLs — built from all_cards (includes variants from parse_html_multi_phase)
    card_page_urls = {f"{card.sanitized_name()}.html": card.url for card in all_cards}
    pipeline_data.page_urls.update(card_page_urls)

    logger.info(f"Total cards: {len(pipeline_data.cards)}")
    logger.info(f"Built page URL mapping with {len(pipeline_data.page_urls)} entries")
    return pipeline_data


def stage_2_enrich_data(card_infos: list[CardInfo]) -> None:
    """Stage 2: Data Enrichment.

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
        items_url=settings.items_page_url,
    )


def _populate_stats_and_keywords(
    session: Session, data: PipelineData, urls: dict[str, str]
) -> None:
    """Populate Stat and Keyword nodes."""
    if data.stats:
        count = session.execute_write(create_stats_from_parsed, data.stats)
        logger.info(f"Created {count} Stat nodes")
        session.execute_write(add_keyword_label_to_stats)

    if data.keywords:
        count = session.execute_write(
            create_keywords_from_parsed, data.keywords, urls.get("Keywords.html")
        )
        logger.info(f"Created {count} Keyword nodes")


def _populate_charms(session: Session, data: PipelineData) -> None:
    """Populate Charm nodes."""
    if data.charms:
        charm_count = session.execute_write(create_charms_from_parsed, data.charms)
        logger.info(f"Created {charm_count} Charm nodes")


def _populate_cards_and_core(
    session: Session, data: PipelineData, urls: dict[str, str]
) -> list[dict[str, Any]]:
    """Populate Cards, Tribes, Crowns, and core relationships.

    Returns:
        cards_dict_data for use by downstream relationship builders.
    """
    cards_dict_data = [card.to_dict() for card in data.cards]
    logger.info(f"Ingesting {len(cards_dict_data)} cards into Neo4j graph...")
    create_neo4j_data(session, cards_dict_data, crowns_url=urls.get("Crowns.html"))
    return cards_dict_data


def _populate_keyword_relationships(
    session: Session, data: PipelineData, cards_dict_data: list[dict[str, Any]]
) -> None:
    """Create Card-Keyword and Charm-Keyword relationships."""
    if data.keywords:
        kw_rel_count = session.execute_write(create_card_keyword_relationships, cards_dict_data)
        logger.info(f"Created {kw_rel_count} Card-Keyword relationships")
        charm_kw_count = session.execute_write(create_charm_keyword_relationships)
        logger.info(f"Created {charm_kw_count} Charm-Keyword relationships")

    if data.charms:
        tribe_count = session.execute_write(create_charm_tribe_relationships, data.charms)
        logger.info(f"Created {tribe_count} Charm-Tribe relationships")


def _populate_map_and_fights(session: Session, data: PipelineData, urls: dict[str, str]) -> None:
    """Populate Map, Zone, Fight nodes and relationships."""
    if data.zones or data.map_events or data.fight_slots:
        counts = session.execute_write(
            create_map_graph,
            data.zones,
            data.map_events,
            data.fight_slots,
            data.fight_page_mapping,
            url=urls.get("Map.html"),
            base_url=settings.wildfrost_wiki_base_url,
        )
        logger.info(f"Map graph created: {counts}")

    if data.fight_enemies and data.fight_page_mapping:
        enemy_count = session.execute_write(
            create_fight_enemy_relationships,
            data.fight_enemies,
            data.fight_page_mapping,
        )
        logger.info(f"Created {enemy_count} Fight-Enemy relationships")

    if data.summons:
        summon_count = session.execute_write(create_summon_relationships, data.summons)
        logger.info(f"Created {summon_count} SUMMONS relationships")


def _populate_bells(session: Session, data: PipelineData) -> None:
    """Populate Bell nodes and relationships."""
    if data.bells:
        bell_count = session.execute_write(create_bells_from_parsed, data.bells)
        logger.info(f"Created {bell_count} Bell nodes")
        bell_rel_count = session.execute_write(create_bell_relationships)
        logger.info(f"Created {bell_rel_count} bell linking relationships")


def _populate_bling_economy(session: Session, data: PipelineData, urls: dict[str, str]) -> None:
    """Populate Bling, Shop nodes, and economy relationships."""
    bling_shop_urls: dict[str, str] = {
        k: v
        for k, v in {
            "Bling": urls.get("Bling.html"),
            "The Woolly Snail": urls.get("The_Woolly_Snail.html"),
            "Charm Merchant": urls.get("Charm_Merchant.html"),
        }.items()
        if v is not None
    }
    session.execute_write(create_bling_and_shops, bling_shop_urls)

    if data.bling_drops:
        drop_count = session.execute_write(create_drops_bling_relationships, data.bling_drops)
        logger.info(f"Created {drop_count} DROPS_BLING relationships")
    if data.woolly_snail_listings:
        snail_count = session.execute_write(
            create_shop_sells_relationships,
            "The Woolly Snail",
            data.woolly_snail_listings,
            "Card",
        )
        logger.info(f"Created {snail_count} Woolly Snail SELLS relationships")
    if data.charm_merchant_listings:
        charm_shop_count = session.execute_write(
            create_shop_sells_relationships,
            "Charm Merchant",
            data.charm_merchant_listings,
            "Charm",
        )
        logger.info(f"Created {charm_shop_count} Charm Merchant SELLS Charm relationships")

    if data.woolly_snail_listings:
        cm_item_count = session.execute_write(
            create_shop_sells_relationships,
            "Charm Merchant",
            data.woolly_snail_listings,
            "Card",
        )
        logger.info(f"Created {cm_item_count} Charm Merchant SELLS Item relationships")
    if data.clunker_prices:
        cm_clunker_count = session.execute_write(
            create_shop_sells_relationships,
            "Charm Merchant",
            data.clunker_prices,
            "Card",
        )
        logger.info(f"Created {cm_clunker_count} Charm Merchant SELLS Clunker relationships")


def stage_3_populate_graph(data: PipelineData) -> None:
    """Stage 3: Neo4j Graph Population.

    Creates nodes and relationships in Neo4j knowledge graph.

    Args:
        data: PipelineData containing all scraped data to populate the graph
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: NEO4J GRAPH POPULATION")
    logger.info("=" * 60)

    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
    )
    try:
        with driver.session() as session:
            urls = data.page_urls

            _populate_stats_and_keywords(session, data, urls)
            _populate_charms(session, data)
            cards_dict_data = _populate_cards_and_core(session, data, urls)
            _populate_keyword_relationships(session, data, cards_dict_data)
            _populate_map_and_fights(session, data, urls)
            _populate_bells(session, data)
            _populate_bling_economy(session, data, urls)

            url_link_count = session.execute_write(create_url_nodes)
            logger.info(f"Created URL nodes with {url_link_count} HAS_LINK relationships")
    finally:
        driver.close()

    logger.info("Graph population complete")


def stage_4_document_ingestion(pipeline_data: PipelineData, split_text: bool = True) -> None:
    """Stage 4: Document Ingestion.

    Chunks HTML documents and ingests into Neo4j as Document nodes.
    Creates full-text search index. Does NOT generate embeddings.
    Use add_embeddings.py separately to add embeddings.

    Args:
        pipeline_data: Pipeline data containing cards and fight_page_mapping
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
    for root, _dirs, files in os.walk(settings.structured_outputs_dir):
        for file in files:
            if file.endswith(".html"):
                filepath = os.path.join(root, file)
                all_html_filepaths.append(filepath)

    logger.info(f"Found {len(all_html_filepaths)} HTML files to process")

    # Chunk the HTML documents
    logger.info(f"Processing HTML documents (split_text={split_text})...")
    all_document_chunks = process_html_files(all_html_filepaths, split_text=split_text)
    logger.info(f"Created {len(all_document_chunks)} document objects")

    url_lookup = pipeline_data.page_urls
    logger.info(f"URL lookup has {len(url_lookup)} entries")

    # Single driver for all document operations
    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
    )
    try:
        with driver.session() as session:
            # Ingest into Neo4j (no embeddings)
            logger.info("Ingesting documents into Neo4j...")
            ingest_documents_into_neo4j(
                session=session,
                document_chunks=all_document_chunks,
                url_lookup=url_lookup,
            )

            # Create full-text search index
            logger.info("Creating full-text search index...")
            create_fulltext_index(
                session=session,
                index_name=settings.fulltext_index_name,
                node_label="Document",
                text_property="text",
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

            # Link Shop nodes to their wiki page Documents
            logger.info("Linking shop nodes to documents...")
            shop_link_count = link_documents_to_shops(session)
            logger.info(f"Linked {shop_link_count} shop nodes to their documents")

            # Link Bling node to its wiki page Document
            logger.info("Linking bling node to document...")
            bling_link_count = link_documents_to_bling(session)
            logger.info(f"Linked {bling_link_count} bling node to its document")

            # Link Bell nodes to their wiki page Document
            logger.info("Linking bell nodes to document...")
            bell_link_count = link_documents_to_bells(session)
            logger.info(f"Linked {bell_link_count} bell nodes to their document")

    finally:
        driver.close()

    logger.info("Document ingestion complete")


async def main() -> None:
    """Main orchestration function."""
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

    logger.info("Starting WildFrostRAG data ingestion pipeline")
    logger.info(f"Configuration: {settings.model_dump()}")

    # Clear database if requested
    if args.clear_db:
        logger.warning("⚠️  CLEARING ENTIRE NEO4J DATABASE ⚠️")

        driver = GraphDatabase.driver(
            settings.neo4j_uri.get_secret_value(),
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        )
        try:
            with driver.session() as session:
                session.execute_write(clear_database)
            logger.info("✅ Database cleared successfully")
        finally:
            driver.close()

    # Ensure directories exist
    settings.create_directories()

    # Stage 1: Scrape (or load from cache)
    pipeline_data = await stage_1_scrape_cards(skip_scrape=args.skip_scrape)

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
        stage_4_document_ingestion(pipeline_data, split_text=split_text)
    else:
        logger.info("Skipping document ingestion (--skip-vectors flag set)")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
