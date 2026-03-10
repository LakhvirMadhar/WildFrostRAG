import os
from typing import List

from src.data_processing.leaders import parse_leaders_page
from src.data_processing.stats import parse_stats_page, StatInfo
from src.data_processing.keywords import parse_keywords_page, KeywordInfo
from src.data_processing.charms import parse_charms_page, CharmInfo
from src.data_processing.bling import parse_bling_page, parse_shop_page, parse_clunker_prices, EnemyBlingDrop, ShopListing
from src.data_processing.bells import parse_bells_page, BellInfo
from src.data_processing.map import parse_map_page, get_fight_page_mapping, ZoneInfo, MapEventInfo, FightSlotInfo
from src.data_processing.fights import parse_fight_enemies
from src.data_processing.shades import parse_shades_page, SummonInfo
from src.data_processing.cards import CardInfo
from src.scraping.wiki_scraper import scrape_wiki_page, load_cached_html
from src.web_scraper.sitemap_scraper import scrape_multiple_links
from src.utils.config import settings
from src.utils.logger import logger

# Type alias: maps HTML filename -> wiki URL
PageUrls = dict[str, str]


async def _get_html(page_name: str, output_subdir: str) -> tuple[str | None, PageUrls]:
    """
    Load HTML from cache if available, otherwise scrape it.

    Returns:
        Tuple of (html_content, page_urls). HTML is None if scraping failed.
        page_urls maps the output filename to the wiki URL (e.g. {"Leaders.html": "https://..."}).
    """
    url = f"{settings.wildfrost_wiki_base_url}/{page_name}"
    urls = {f"{page_name}.html": url}
    html = load_cached_html(page_name, output_subdir)
    if html:
        return html, urls
    return await scrape_wiki_page(page_name, output_subdir), urls


async def scrape_leaders() -> tuple[List[CardInfo], PageUrls]:
    """Parse the Leaders page (from cache or web)."""
    html, urls = await _get_html("Leaders", "leaders")
    if not html:
        return [], urls

    leader_cards = parse_leaders_page(html, url=urls.get("Leaders.html", ""))
    logger.info(f"Parsed {len(leader_cards)} leader cards")
    return leader_cards, urls


async def scrape_stats() -> tuple[List[StatInfo], PageUrls]:
    """Parse the Stats page (from cache or web)."""
    html, urls = await _get_html("Stats", "stats")
    if not html:
        return [], urls

    stats = parse_stats_page(html, base_url=settings.wildfrost_wiki_base_url)
    logger.info(f"Parsed {len(stats)} stats")
    return stats, urls


async def scrape_individual_stat_pages(stats: List[StatInfo]) -> PageUrls:
    """
    Scrape individual stat wiki pages for per-stat Document content.

    Checks cache first (via stat.save_path()), scrapes any missing pages.
    Each stat's HTML is saved to data/structured_outputs/stats/{name}.html.

    Args:
        stats: List of StatInfo objects (already parsed from summary page with url set)

    Returns:
        PageUrls dict mapping filename -> wiki URL for each stat
    """
    page_urls: PageUrls = {}
    stats_to_scrape: list[StatInfo] = []

    for stat in stats:
        if not stat.url:
            continue
        filename = f"{stat.sanitized_name()}.html"
        page_urls[filename] = stat.url

        if os.path.exists(stat.save_path()):
            continue
        stats_to_scrape.append(stat)

    logger.info(f"Individual stat pages: {len(stats) - len(stats_to_scrape)} cached, {len(stats_to_scrape)} to scrape")

    if stats_to_scrape:
        urls = [s.url for s in stats_to_scrape]
        htmls = await scrape_multiple_links(urls, max_concurrent=settings.max_concurrent_requests)

        for stat, html in zip(stats_to_scrape, htmls):
            if html is None:
                logger.warning(f"Failed to scrape individual page for {stat.name}")
                continue
            stat.stat_html = html
            stat.save_html()

    return page_urls


async def scrape_keywords() -> tuple[List[KeywordInfo], PageUrls]:
    """Parse the Keywords page (from cache or web)."""
    html, urls = await _get_html("Keywords", "keywords")
    if not html:
        return [], urls

    keywords = parse_keywords_page(html)
    logger.info(f"Parsed {len(keywords)} keywords")
    return keywords, urls


async def scrape_charms() -> tuple[List[CharmInfo], PageUrls]:
    """Parse the Charms page (from cache or web)."""
    html, urls = await _get_html("Charms", "charms")
    if not html:
        return [], urls

    charms = parse_charms_page(html, base_url=settings.wildfrost_wiki_base_url)
    logger.info(f"Parsed {len(charms)} charms")
    return charms, urls


async def scrape_individual_charm_pages(charms: List[CharmInfo]) -> PageUrls:
    """
    Scrape individual charm wiki pages for per-charm Document content.

    Checks cache first (via charm.save_path()), scrapes any missing pages.
    Each charm's HTML is saved to data/structured_outputs/charms/{name}.html.

    Args:
        charms: List of CharmInfo objects (already parsed from summary page with url set)

    Returns:
        PageUrls dict mapping filename -> wiki URL for each charm
    """
    page_urls: PageUrls = {}
    charms_to_scrape: list[CharmInfo] = []

    for charm in charms:
        if not charm.url:
            continue
        filename = f"{charm.sanitized_name()}.html"
        page_urls[filename] = charm.url

        if os.path.exists(charm.save_path()):
            continue
        charms_to_scrape.append(charm)

    logger.info(f"Individual charm pages: {len(charms) - len(charms_to_scrape)} cached, {len(charms_to_scrape)} to scrape")

    if charms_to_scrape:
        urls = [c.url for c in charms_to_scrape]
        htmls = await scrape_multiple_links(urls, max_concurrent=settings.max_concurrent_requests)

        for charm, html in zip(charms_to_scrape, htmls):
            if html is None:
                logger.warning(f"Failed to scrape individual page for {charm.name}")
                continue
            charm.charm_html = html
            charm.save_html()

    return page_urls


async def scrape_shades() -> tuple[List[SummonInfo], PageUrls]:
    """Parse the Shades page for summoning relationships (from cache or web)."""
    html, urls = await _get_html("Shades", "shades")
    if not html:
        return [], urls

    summons = parse_shades_page(html)
    logger.info(f"Parsed {len(summons)} summoning relationships")
    return summons, urls


async def scrape_map() -> tuple[List[ZoneInfo], List[MapEventInfo], List[FightSlotInfo], dict[str, str], PageUrls]:
    """Parse the Map page (from cache or web)."""
    html, urls = await _get_html("Map", "maps")
    if not html:
        return [], [], [], {}, urls

    zones, map_events, fight_slots = parse_map_page(html)
    fight_page_mapping = get_fight_page_mapping(html)
    logger.info(f"Parsed {len(zones)} zones, {len(map_events)} map events, {len(fight_slots)} fight slots")
    logger.info(f"Extracted {len(fight_page_mapping)} fight page mappings")
    return zones, map_events, fight_slots, fight_page_mapping, urls


async def scrape_fight_pages(fight_page_mapping: dict[str, str]) -> tuple[dict[str, List[str]], PageUrls]:
    """Parse individual fight pages and extract enemy names (from cache or web).

    Returns:
        Tuple of (fight_enemies dict, page_urls dict mapping filename -> URL)
    """
    page_slugs = list(set(fight_page_mapping.values()))
    logger.info(f"Processing {len(page_slugs)} fight pages...")

    fight_enemies = {}
    page_urls: PageUrls = {}
    for page_slug in page_slugs:
        html, slug_urls = await _get_html(page_slug, "fights")
        if html:
            enemies = parse_fight_enemies(html)
            fight_enemies[page_slug] = enemies
            page_urls.update(slug_urls)
            logger.info(f"  {page_slug}: {len(enemies)} enemies")

    total = sum(len(e) for e in fight_enemies.values())
    logger.info(f"Finished {len(page_slugs)} fight pages ({total} total enemy entries)")
    return fight_enemies, page_urls


async def scrape_bling(boss_names: List[str], miniboss_names: List[str]) -> tuple[List[EnemyBlingDrop], PageUrls]:
    """Parse the Bling page for enemy drop values (from cache or web)."""
    html, urls = await _get_html("Bling", "bling")
    if not html:
        return [], urls

    drops = parse_bling_page(html, boss_names, miniboss_names)
    logger.info(f"Parsed {len(drops)} enemy bling drops")
    return drops, urls


async def scrape_shop(page_name: str, subdir: str) -> tuple[List[ShopListing], PageUrls]:
    """Parse a shop page for item/charm listings (from cache or web)."""
    html, urls = await _get_html(page_name, subdir)
    if not html:
        return [], urls

    listings = parse_shop_page(html)
    logger.info(f"Parsed {len(listings)} listings from {page_name}")
    return listings, urls


async def scrape_clunker_prices() -> tuple[List[ShopListing], PageUrls]:
    """Parse the Clunkers page for clunker shop prices (from cache or web)."""
    html, urls = await _get_html("Clunkers", "clunkers_page")
    if not html:
        return [], urls

    listings = parse_clunker_prices(html)
    logger.info(f"Parsed {len(listings)} clunker prices")
    return listings, urls


async def scrape_bells() -> tuple[List[BellInfo], PageUrls]:
    """Parse the Bells page (from cache or web)."""
    html, urls = await _get_html("Bells", "bells")
    if not html:
        return [], urls

    bells = parse_bells_page(html, base_url=settings.wildfrost_wiki_base_url)
    logger.info(f"Parsed {len(bells)} bells")
    return bells, urls


async def scrape_individual_bell_pages(bells: List[BellInfo]) -> PageUrls:
    """
    Scrape individual bell wiki pages for per-bell Document content.

    Only bells with real wiki pages (not red links) get scraped.
    Each bell's HTML is saved to data/structured_outputs/bells/{name}.html.

    Args:
        bells: List of BellInfo objects (already parsed from summary page with url set)

    Returns:
        PageUrls dict mapping filename -> wiki URL for each bell with a page
    """
    page_urls: PageUrls = {}
    bells_to_scrape: list[BellInfo] = []

    for bell in bells:
        if not bell.url:
            continue
        filename = f"{bell.sanitized_name()}.html"
        page_urls[filename] = bell.url

        if os.path.exists(bell.save_path()):
            continue
        bells_to_scrape.append(bell)

    logger.info(f"Individual bell pages: {len(page_urls)} have pages, {len(bells_to_scrape)} to scrape")

    if bells_to_scrape:
        urls = [b.url for b in bells_to_scrape]
        htmls = await scrape_multiple_links(urls, max_concurrent=settings.max_concurrent_requests)

        for bell, html in zip(bells_to_scrape, htmls):
            if html is None:
                logger.warning(f"Failed to scrape individual page for {bell.name}")
                continue
            bell.bell_html = html
            bell.save_html()

    return page_urls


async def scrape_crowns() -> tuple[None, PageUrls]:
    """Fetch the Crowns page (from cache or web). No structured parsing."""
    _, urls = await _get_html("Crowns", "crowns")
    return None, urls


async def scrape_getting_started() -> tuple[None, PageUrls]:
    """Fetch the Getting Started page (from cache or web). No structured parsing."""
    _, urls = await _get_html("Getting_Started", "getting_started")
    return None, urls
