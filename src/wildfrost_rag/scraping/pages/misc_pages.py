"""Scraping for standalone wiki pages that don't fit elsewhere.

Each of these is a single fetch-and-parse (or fetch-and-cache) operation
with no individual-page variant and no natural pairing with another
domain - grouped here deliberately, rather than given one file each.
"""

import aiohttp

from wildfrost_rag.data_processing.keywords import KeywordInfo, parse_keywords_page
from wildfrost_rag.data_processing.shades import SummonInfo, parse_shades_page
from wildfrost_rag.domain.scraping_types import PageUrls
from wildfrost_rag.scraping._page_fetching import prefetch_page, scrape_page


async def scrape_keywords(session: aiohttp.ClientSession) -> tuple[list[KeywordInfo], PageUrls]:
    """Parse the Keywords page (from cache or web)."""
    return await scrape_page(session, "Keywords", "keywords", parse_keywords_page)


async def scrape_shades(session: aiohttp.ClientSession) -> tuple[list[SummonInfo], PageUrls]:
    """Parse the Shades page for summoning relationships (from cache or web)."""
    return await scrape_page(session, "Shades", "shades", parse_shades_page)


async def scrape_crowns(session: aiohttp.ClientSession) -> tuple[None, PageUrls]:
    """Fetch the Crowns page (from cache or web). No structured parsing."""
    return await prefetch_page(session, "Crowns", "crowns")


async def scrape_getting_started(session: aiohttp.ClientSession) -> tuple[None, PageUrls]:
    """Fetch the Getting Started page (from cache or web). No structured parsing."""
    return await prefetch_page(session, "Getting_Started", "getting_started")
