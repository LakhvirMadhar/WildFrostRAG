"""Scraping for the Map page and the individual fight pages it links to."""

import aiohttp

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.map import FightSlotInfo, MapEventInfo, ZoneInfo
from wildfrost_rag.data_processing.pages.fights.parser import parse_fight_enemies
from wildfrost_rag.data_processing.pages.map.parser import get_fight_page_mapping, parse_map_page
from wildfrost_rag.domain.scraping_types import FightEnemies, FightPageMapping, PageUrls
from wildfrost_rag.scraping.page_fetching import get_html, get_html_many

_MAP_PAGE_NAME = "Map"
_MAP_CACHE_SUBDIR = "maps"
_FIGHT_PAGES_CACHE_SUBDIR = "fights"


async def scrape_map(
    session: aiohttp.ClientSession,
) -> tuple[list[ZoneInfo], list[MapEventInfo], list[FightSlotInfo], FightPageMapping, PageUrls]:
    """Parse the Map page (from cache or web)."""
    html, urls = await get_html(session, _MAP_PAGE_NAME, _MAP_CACHE_SUBDIR)
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
    """Parse individual fight pages and extract enemy names (from cache or web, in parallel).

    Returns:
        Tuple of (fight_enemies dict, page_urls dict mapping filename -> URL)
    """
    page_slugs = list(set(fight_page_mapping.values()))
    logger.info(f"Processing {len(page_slugs)} fight pages...")

    results = await get_html_many(session, page_slugs, _FIGHT_PAGES_CACHE_SUBDIR)

    fight_enemies: FightEnemies = {}
    page_urls: PageUrls = {}
    for page_slug, (html, slug_urls) in zip(page_slugs, results, strict=True):
        if html:
            enemies = parse_fight_enemies(html)
            fight_enemies[page_slug] = enemies
            page_urls.update(slug_urls)
            logger.info(f"  {page_slug}: {len(enemies)} enemies")

    total = sum(len(e) for e in fight_enemies.values())
    logger.info(f"Finished {len(page_slugs)} fight pages ({total} total enemy entries)")
    return fight_enemies, page_urls
