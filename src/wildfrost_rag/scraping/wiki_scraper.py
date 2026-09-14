"""Fetches individual wiki pages and manages their on-disk HTML cache.

Layer above sitemap_scraper.py (raw batched HTTP) - this module knows the
convention every wiki page follows: a page has a name, and its HTML is
cached under structured_outputs_dir/{output_subdir}/{page_name}.html. It
still knows nothing about Wildfrost's game content (that's data_processing/
and scraping/pages/).
"""

import re
from pathlib import Path

import aiohttp

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger
from wildfrost_rag.scraping.sitemap_scraper import scrape_multiple_links


def clean_name_for_url(name: str) -> str:
    """Clean card name for use in URLs by replacing spaces with underscores."""
    return re.sub(r"\s+", "_", name)


def _cache_path(page_name: str, output_subdir: str) -> Path:
    """Return the on-disk path a wiki page's cached HTML lives at."""
    return get_settings().paths.structured_outputs_dir / output_subdir / f"{page_name}.html"


async def _fetch_html(session: aiohttp.ClientSession, page_name: str) -> str | None:
    """Fetch a wiki page's HTML over HTTP. No disk access."""
    url = f"{get_settings().scraping.wildfrost_wiki_base_url}/{page_name}"
    html_list = await scrape_multiple_links(session, [url], max_concurrent=1)
    return html_list[0] if html_list else None


def _save_to_cache(html: str, page_name: str, output_subdir: str) -> Path:
    """Write a wiki page's HTML to its cache path, creating the directory if needed."""
    output_path = _cache_path(page_name, output_subdir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


async def scrape_wiki_page(
    session: aiohttp.ClientSession, page_name: str, output_subdir: str
) -> str | None:
    """Fetch a wiki page over HTTP and cache it to disk.

    Args:
        session: Shared HTTP session (constructed and owned by the caller).
        page_name: Name of the wiki page (e.g., "Crowns", "Leaders", "Stats")
        output_subdir: Subdirectory under structured_outputs_dir to save the HTML

    Returns:
        HTML content if successful, None otherwise
    """
    logger.info(f"Scraping {page_name} page...")
    html = await _fetch_html(session, page_name)

    if not html:
        logger.warning(f"Failed to scrape {page_name} page")
        return None

    output_path = _save_to_cache(html, page_name, output_subdir)
    logger.info(f"Saved {page_name} HTML to {output_path}")

    return html


def load_cached_html(page_name: str, output_subdir: str) -> str | None:
    """Load a previously scraped wiki page from disk.

    Args:
        page_name: Name of the wiki page (e.g., "Crowns", "Leaders", "Stats")
        output_subdir: Subdirectory under structured_outputs_dir where HTML was saved

    Returns:
        HTML content if file exists, None otherwise
    """
    path = _cache_path(page_name, output_subdir)
    if not path.exists():
        logger.warning(f"Cached HTML not found: {path}")
        return None

    with open(path, encoding="utf-8") as f:
        html = f.read()
    logger.info(f"Loaded cached {page_name} from {path}")
    return html
