"""Scraping for the Bells wiki page and its individual bell pages."""

from functools import partial

import aiohttp

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.data_processing.bells import BellInfo
from wildfrost_rag.data_processing.pages.bells.parser import parse_bells_page
from wildfrost_rag.domain.scraping_types import PageUrls
from wildfrost_rag.scraping.page_fetching import scrape_individual_pages, scrape_page

_PAGE_NAME = "Bells"
_CACHE_SUBDIR = "bells"


async def scrape_bells(session: aiohttp.ClientSession) -> tuple[list[BellInfo], PageUrls]:
    """Parse the Bells page (from cache or web)."""
    base_url = get_settings().scraping.wildfrost_wiki_base_url
    return await scrape_page(
        session, _PAGE_NAME, _CACHE_SUBDIR, partial(parse_bells_page, base_url=base_url)
    )


async def scrape_individual_bell_pages(
    session: aiohttp.ClientSession, bells: list[BellInfo]
) -> PageUrls:
    """Scrape individual bell wiki pages for per-bell Document content.

    Only bells with real wiki pages (not red links) get scraped.
    """
    return await scrape_individual_pages(session, bells, "bell", _CACHE_SUBDIR)
