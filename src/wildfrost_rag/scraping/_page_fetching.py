"""Shared HTML-fetching helpers used by every scraper under scraping/pages/.

These carry no game-specific knowledge - they only know how to get HTML
(from cache or the web) and hand it to a caller-supplied parser. Domain
knowledge (what a Stat or Charm is) lives entirely in scraping/pages/.
"""

import asyncio
import os
from collections.abc import Callable

import aiohttp

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger
from wildfrost_rag.domain.scraping_types import HasWikiPage, PageUrls
from wildfrost_rag.scraping.sitemap_scraper import scrape_multiple_links
from wildfrost_rag.scraping.wiki_scraper import load_cached_html, scrape_wiki_page


async def get_html(
    session: aiohttp.ClientSession, page_name: str, output_subdir: str
) -> tuple[str | None, PageUrls]:
    """Load HTML from cache if available, otherwise scrape it.

    Returns:
        Tuple of (html_content, page_urls). HTML is None if scraping failed.
        page_urls maps the output filename to the wiki URL (e.g. {"Leaders.html": "https://..."}).
    """
    url = f"{get_settings().scraping.wildfrost_wiki_base_url}/{page_name}"
    urls = {f"{page_name}.html": url}
    html = load_cached_html(page_name, output_subdir)
    if html:
        return html, urls
    return await scrape_wiki_page(session, page_name, output_subdir), urls


async def scrape_page[T](
    session: aiohttp.ClientSession,
    page_name: str,
    output_subdir: str,
    parser: Callable[[str], list[T]],
) -> tuple[list[T], PageUrls]:
    """Fetch (cache-or-web), parse, log - the shared skeleton every summary-page scrape follows."""
    html, urls = await get_html(session, page_name, output_subdir)
    if not html:
        return [], urls

    items = parser(html)
    logger.info(f"Parsed {len(items)} items from {page_name}")
    return items, urls


async def prefetch_page(
    session: aiohttp.ClientSession, page_name: str, output_subdir: str
) -> tuple[None, PageUrls]:
    """Warm the HTML cache for a page with no structured parser."""
    _, urls = await get_html(session, page_name, output_subdir)
    return None, urls


async def get_html_many(
    session: aiohttp.ClientSession, page_names: list[str], output_subdir: str
) -> list[tuple[str | None, PageUrls]]:
    """Fetch multiple pages' HTML (cache-or-web) concurrently, preserving order.

    Bounded by the same max_concurrent_requests setting scrape_multiple_links
    uses, so a page list can't fire off more simultaneous requests than the
    rest of the scraper does.
    """
    semaphore = asyncio.Semaphore(get_settings().scraping.max_concurrent_requests)

    async def _fetch(page_name: str) -> tuple[str | None, PageUrls]:
        async with semaphore:
            return await get_html(session, page_name, output_subdir)

    return await asyncio.gather(*(_fetch(page_name) for page_name in page_names))


def _partition_by_cache[T: HasWikiPage](entities: list[T]) -> tuple[PageUrls, list[T]]:
    """Split entities into their known page_urls and the ones missing a cached page."""
    page_urls: PageUrls = {}
    to_scrape: list[T] = []

    for entity in entities:
        if not entity.url:
            continue
        page_urls[f"{entity.sanitized_name()}.html"] = entity.url
        if not os.path.exists(entity.save_path()):
            to_scrape.append(entity)

    return page_urls, to_scrape


async def _scrape_and_save[T: HasWikiPage](
    session: aiohttp.ClientSession, entities: list[T], entity_label: str
) -> None:
    """Fetch HTML for entities missing a cached page, then save each one."""
    urls = [e.url for e in entities if e.url is not None]
    htmls = await scrape_multiple_links(
        session, urls, max_concurrent=get_settings().scraping.max_concurrent_requests
    )
    for entity, html in zip(entities, htmls, strict=False):
        if html is None:
            logger.warning(f"Failed to scrape individual page for {entity_label} '{entity.name}'")
            continue
        entity.set_html(html)
        entity.save_html()


async def scrape_individual_pages[T: HasWikiPage](
    session: aiohttp.ClientSession, entities: list[T], entity_label: str
) -> PageUrls:
    """Scrape+cache per-entity wiki pages for any entity with its own wiki page.

    Args:
        session: Shared HTTP session (constructed and owned by the caller).
        entities: Entities already parsed from a summary page, with url set.
        entity_label: Singular noun for log messages (e.g. "stat", "charm", "bell").

    Returns:
        PageUrls dict mapping filename -> wiki URL for each entity with a page.
    """
    page_urls, to_scrape = _partition_by_cache(entities)
    logger.info(
        f"Individual {entity_label} pages: {len(entities) - len(to_scrape)} cached, "
        f"{len(to_scrape)} to scrape"
    )

    if to_scrape:
        await _scrape_and_save(session, to_scrape, entity_label)

    return page_urls
