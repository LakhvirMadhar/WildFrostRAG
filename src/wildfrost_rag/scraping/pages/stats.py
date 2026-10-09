"""Scraping for the Stats wiki page and its individual stat pages."""

from functools import partial

import aiohttp

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.data_processing.stats import StatInfo, parse_stats_page
from wildfrost_rag.domain.scraping_types import PageUrls
from wildfrost_rag.scraping.page_fetching import scrape_individual_pages, scrape_page

_PAGE_NAME = "Stats"
_CACHE_SUBDIR = "stats"


async def scrape_stats(session: aiohttp.ClientSession) -> tuple[list[StatInfo], PageUrls]:
    """Parse the Stats page (from cache or web)."""
    base_url = get_settings().scraping.wildfrost_wiki_base_url
    return await scrape_page(
        session, _PAGE_NAME, _CACHE_SUBDIR, partial(parse_stats_page, base_url=base_url)
    )


async def scrape_individual_stat_pages(
    session: aiohttp.ClientSession, stats: list[StatInfo]
) -> PageUrls:
    """Scrape individual stat wiki pages for per-stat Document content."""
    return await scrape_individual_pages(session, stats, "stat", _CACHE_SUBDIR)
