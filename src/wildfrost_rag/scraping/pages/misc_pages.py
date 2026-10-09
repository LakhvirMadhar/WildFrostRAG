"""Scraping for standalone wiki pages that don't fit elsewhere.

Each of these is a single fetch-and-parse (or fetch-and-cache) operation
with no individual-page variant and no natural pairing with another
domain - grouped here deliberately, rather than given one file each.
"""

import aiohttp

from wildfrost_rag.data_processing.keywords import KeywordInfo
from wildfrost_rag.data_processing.pages.keywords.parser import parse_keywords_page
from wildfrost_rag.data_processing.pages.shades.parser import parse_shades_page
from wildfrost_rag.data_processing.shades import SummonInfo
from wildfrost_rag.domain.scraping_types import PageUrls
from wildfrost_rag.scraping.page_fetching import prefetch_page, scrape_page

_KEYWORDS_PAGE_NAME = "Keywords"
_KEYWORDS_CACHE_SUBDIR = "keywords"
_SHADES_PAGE_NAME = "Shades"
_SHADES_CACHE_SUBDIR = "shades"
_CROWNS_PAGE_NAME = "Crowns"
_CROWNS_CACHE_SUBDIR = "crowns"
_GETTING_STARTED_PAGE_NAME = "Getting_Started"
_GETTING_STARTED_CACHE_SUBDIR = "getting_started"


async def scrape_keywords(session: aiohttp.ClientSession) -> tuple[list[KeywordInfo], PageUrls]:
    """Parse the Keywords page (from cache or web)."""
    return await scrape_page(
        session, _KEYWORDS_PAGE_NAME, _KEYWORDS_CACHE_SUBDIR, parse_keywords_page
    )


async def scrape_shades(session: aiohttp.ClientSession) -> tuple[list[SummonInfo], PageUrls]:
    """Parse the Shades page for summoning relationships (from cache or web)."""
    return await scrape_page(session, _SHADES_PAGE_NAME, _SHADES_CACHE_SUBDIR, parse_shades_page)


async def scrape_crowns(session: aiohttp.ClientSession) -> tuple[None, PageUrls]:
    """Fetch the Crowns page (from cache or web). No structured parsing."""
    return await prefetch_page(session, _CROWNS_PAGE_NAME, _CROWNS_CACHE_SUBDIR)


async def scrape_getting_started(session: aiohttp.ClientSession) -> tuple[None, PageUrls]:
    """Fetch the Getting Started page (from cache or web). No structured parsing."""
    return await prefetch_page(session, _GETTING_STARTED_PAGE_NAME, _GETTING_STARTED_CACHE_SUBDIR)
