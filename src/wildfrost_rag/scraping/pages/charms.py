"""Scraping for the Charms wiki page and its individual charm pages."""

from functools import partial

import aiohttp

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.data_processing.charms import CharmInfo, parse_charms_page
from wildfrost_rag.domain.scraping_types import PageUrls
from wildfrost_rag.scraping.page_fetching import scrape_individual_pages, scrape_page

_PAGE_NAME = "Charms"
_CACHE_SUBDIR = "charms"


async def scrape_charms(session: aiohttp.ClientSession) -> tuple[list[CharmInfo], PageUrls]:
    """Parse the Charms page (from cache or web)."""
    base_url = get_settings().scraping.wildfrost_wiki_base_url
    return await scrape_page(
        session, _PAGE_NAME, _CACHE_SUBDIR, partial(parse_charms_page, base_url=base_url)
    )


async def scrape_individual_charm_pages(
    session: aiohttp.ClientSession, charms: list[CharmInfo]
) -> PageUrls:
    """Scrape individual charm wiki pages for per-charm Document content."""
    return await scrape_individual_pages(session, charms, "charm")
