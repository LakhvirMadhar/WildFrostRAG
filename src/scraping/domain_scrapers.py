from typing import List

from src.data_processing.leaders import parse_leaders_page
from src.data_processing.stats import parse_stats_page, StatInfo
from src.data_processing.charms import parse_charms_page, CharmInfo
from src.data_processing.map import parse_map_page, get_fight_page_mapping, ZoneInfo, MapEventInfo, FightSlotInfo
from src.data_processing.fights import parse_fight_enemies
from src.data_processing.shades import parse_shades_page, SummonInfo
from src.data_processing.cards import CardInfo
from src.scraping.wiki_scraper import scrape_wiki_page
from src.utils.logger import logger


async def scrape_leaders() -> List[CardInfo]:
    """
    Scrape and parse the Leaders page.

    Returns:
        List of CardInfo objects for all leaders
    """
    html = await scrape_wiki_page("Leaders", "leaders")
    if not html:
        return []

    leader_cards = parse_leaders_page(html)
    logger.info(f"Parsed {len(leader_cards)} leader cards")
    return leader_cards


async def scrape_stats() -> List[StatInfo]:
    """
    Scrape and parse the Stats page.

    Returns:
        List of StatInfo objects for all stats
    """
    html = await scrape_wiki_page("Stats", "stats")
    if not html:
        return []

    stats = parse_stats_page(html)
    logger.info(f"Parsed {len(stats)} stats")
    return stats


async def scrape_charms() -> List[CharmInfo]:
    """
    Scrape and parse the Charms page.

    Returns:
        List of CharmInfo objects for all charms
    """
    html = await scrape_wiki_page("Charms", "charms")
    if not html:
        return []

    charms = parse_charms_page(html)
    logger.info(f"Parsed {len(charms)} charms")
    return charms


async def scrape_shades() -> List[SummonInfo]:
    """
    Scrape and parse the Shades page for summoning relationships.

    Returns:
        List of SummonInfo objects linking summoner cards to shades
    """
    html = await scrape_wiki_page("Shades", "shades")
    if not html:
        return []

    summons = parse_shades_page(html)
    logger.info(f"Parsed {len(summons)} summoning relationships")
    return summons


async def scrape_map() -> tuple[List[ZoneInfo], List[MapEventInfo], List[FightSlotInfo], dict[str, str]]:
    """
    Scrape and parse the Map page.

    Returns:
        Tuple of (zones, map_events, fight_slots, fight_page_mapping)
    """
    html = await scrape_wiki_page("Map", "maps")
    if not html:
        return [], [], [], {}

    zones, map_events, fight_slots = parse_map_page(html)
    fight_page_mapping = get_fight_page_mapping(html)
    logger.info(f"Parsed {len(zones)} zones, {len(map_events)} map events, {len(fight_slots)} fight slots")
    logger.info(f"Extracted {len(fight_page_mapping)} fight page mappings")
    return zones, map_events, fight_slots, fight_page_mapping


async def scrape_fight_pages(fight_page_mapping: dict[str, str]) -> dict[str, List[str]]:
    """
    Scrape individual fight pages and parse enemy names from each.

    Each fight page is saved to data/structured_outputs/fights/{page_slug}.html.

    Args:
        fight_page_mapping: Display name -> wiki page slug mapping

    Returns:
        Dict mapping page_slug -> list of enemy card names
    """
    page_slugs = list(set(fight_page_mapping.values()))
    logger.info(f"Scraping {len(page_slugs)} fight pages...")

    fight_enemies = {}
    for page_slug in page_slugs:
        html = await scrape_wiki_page(page_slug, "fights")
        if html:
            enemies = parse_fight_enemies(html)
            fight_enemies[page_slug] = enemies
            logger.info(f"  {page_slug}: {len(enemies)} enemies")

    total = sum(len(e) for e in fight_enemies.values())
    logger.info(f"Finished scraping {len(page_slugs)} fight pages ({total} total enemy entries)")
    return fight_enemies
