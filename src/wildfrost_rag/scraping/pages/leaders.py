"""Scraping for the Leaders wiki page."""

import aiohttp

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.cards import CardInfo
from wildfrost_rag.data_processing.leaders import parse_leaders_page
from wildfrost_rag.domain.scraping_types import PageUrls
from wildfrost_rag.scraping._page_fetching import get_html

_PAGE_NAME = "Leaders"
_CACHE_SUBDIR = "leaders"


async def scrape_leaders(session: aiohttp.ClientSession) -> tuple[list[CardInfo], PageUrls]:
    """Parse the Leaders page (from cache or web).

    Not built on scrape_page: parse_leaders_page needs the page's own
    scraped URL, which isn't known until get_html returns - not a fixed
    base_url like the other summary-page parsers take.
    """
    html, urls = await get_html(session, _PAGE_NAME, _CACHE_SUBDIR)
    if not html:
        return [], urls

    leader_cards = parse_leaders_page(html, url=urls.get("Leaders.html", ""))
    logger.info(f"Parsed {len(leader_cards)} leader cards")
    return leader_cards, urls
