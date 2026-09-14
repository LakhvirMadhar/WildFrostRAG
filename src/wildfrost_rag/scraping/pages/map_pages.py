"""Scraping for the Map page and the individual fight pages it links to."""

import aiohttp

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.fights import parse_fight_enemies
from wildfrost_rag.data_processing.map import (
    FightSlotInfo,
    MapEventInfo,
    ZoneInfo,
    get_fight_page_mapping,
    parse_map_page,
)
from wildfrost_rag.domain.scraping_types import FightEnemies, FightPageMapping, PageUrls
from wildfrost_rag.scraping._page_fetching import get_html


async def scrape_map(
    session: aiohttp.ClientSession,
) -> tuple[list[ZoneInfo], list[MapEventInfo], list[FightSlotInfo], FightPageMapping, PageUrls]:
    """Parse the Map page (from cache or web)."""
    html, urls = await get_html(session, "Map", "maps")
    if not html:
        return [], [], [], {}, urls

    zones, map_events, fight_slots = parse_map_page(html)
    fight_page_mapping = get_fight_page_mapping(html)
    logger.info(
        f"Parsed {len(zones)} zones, {len(map_events)} map events, {len(fight_slots)} fight slots"
    )
    logger.info(f"Extracted {len(fight_page_mapping)} fight page mappings")
    return zones, map_events, fight_slots, fight_page_mapping, urls


async def scrape_fight_pages(
    session: aiohttp.ClientSession, fight_page_mapping: FightPageMapping
) -> tuple[FightEnemies, PageUrls]:
    """Parse individual fight pages and extract enemy names (from cache or web).

    Returns:
        Tuple of (fight_enemies dict, page_urls dict mapping filename -> URL)
    """
    page_slugs = list(set(fight_page_mapping.values()))
    logger.info(f"Processing {len(page_slugs)} fight pages...")

    fight_enemies = {}
    page_urls: PageUrls = {}
    for page_slug in page_slugs:
        html, slug_urls = await get_html(session, page_slug, "fights")
        if html:
            enemies = parse_fight_enemies(html)
            fight_enemies[page_slug] = enemies
            page_urls.update(slug_urls)
            logger.info(f"  {page_slug}: {len(enemies)} enemies")

    total = sum(len(e) for e in fight_enemies.values())
    logger.info(f"Finished {len(page_slugs)} fight pages ({total} total enemy entries)")
    return fight_enemies, page_urls
