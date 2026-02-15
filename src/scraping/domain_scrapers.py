from typing import List

from src.data_processing.leaders import parse_leaders_page
from src.data_processing.stats import parse_stats_page, StatInfo
from src.data_processing.charms import parse_charms_page, CharmInfo
from src.data_processing.map import parse_map_page, get_fight_page_mapping, ZoneInfo, MapEventInfo, FightSlotInfo
from src.data_processing.fights import parse_fight_enemies
from src.data_processing.shades import parse_shades_page, SummonInfo
from src.data_processing.cards import CardInfo
from src.scraping.wiki_scraper import scrape_wiki_page, load_cached_html
from src.utils.logger import logger


async def _get_html(page_name: str, output_subdir: str) -> str | None:
    """Load HTML from cache if available, otherwise scrape it."""
    html = load_cached_html(page_name, output_subdir)
    if html:
        return html
    return await scrape_wiki_page(page_name, output_subdir)


async def scrape_leaders() -> List[CardInfo]:
    """Parse the Leaders page (from cache or web)."""
    html = await _get_html("Leaders", "leaders")
    if not html:
        return []

    leader_cards = parse_leaders_page(html)
    logger.info(f"Parsed {len(leader_cards)} leader cards")
    return leader_cards


async def scrape_stats() -> List[StatInfo]:
    """Parse the Stats page (from cache or web)."""
    html = await _get_html("Stats", "stats")
    if not html:
        return []

    stats = parse_stats_page(html)
    logger.info(f"Parsed {len(stats)} stats")
    return stats


async def scrape_charms() -> List[CharmInfo]:
    """Parse the Charms page (from cache or web)."""
    html = await _get_html("Charms", "charms")
    if not html:
        return []

    charms = parse_charms_page(html)
    logger.info(f"Parsed {len(charms)} charms")
    return charms


async def scrape_shades() -> List[SummonInfo]:
    """Parse the Shades page for summoning relationships (from cache or web)."""
    html = await _get_html("Shades", "shades")
    if not html:
        return []

    summons = parse_shades_page(html)
    logger.info(f"Parsed {len(summons)} summoning relationships")
    return summons


async def scrape_map() -> tuple[List[ZoneInfo], List[MapEventInfo], List[FightSlotInfo], dict[str, str]]:
    """Parse the Map page (from cache or web)."""
    html = await _get_html("Map", "maps")
    if not html:
        return [], [], [], {}

    zones, map_events, fight_slots = parse_map_page(html)
    fight_page_mapping = get_fight_page_mapping(html)
    logger.info(f"Parsed {len(zones)} zones, {len(map_events)} map events, {len(fight_slots)} fight slots")
    logger.info(f"Extracted {len(fight_page_mapping)} fight page mappings")
    return zones, map_events, fight_slots, fight_page_mapping


async def scrape_fight_pages(fight_page_mapping: dict[str, str]) -> dict[str, List[str]]:
    """Parse individual fight pages and extract enemy names (from cache or web)."""
    page_slugs = list(set(fight_page_mapping.values()))
    logger.info(f"Processing {len(page_slugs)} fight pages...")

    fight_enemies = {}
    for page_slug in page_slugs:
        html = await _get_html(page_slug, "fights")
        if html:
            enemies = parse_fight_enemies(html)
            fight_enemies[page_slug] = enemies
            logger.info(f"  {page_slug}: {len(enemies)} enemies")

    total = sum(len(e) for e in fight_enemies.values())
    logger.info(f"Finished {len(page_slugs)} fight pages ({total} total enemy entries)")
    return fight_enemies
