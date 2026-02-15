import re

from src.web_scraper.sitemap_scraper import scrape_multiple_links
from src.utils.config import settings
from src.utils.logger import logger


def clean_name_for_url(name: str) -> str:
    """Clean card name for use in URLs by replacing spaces with underscores."""
    return re.sub(r'\s+', '_', name)


async def scrape_wiki_page(page_name: str, output_subdir: str) -> str | None:
    """
    Scrape a wiki page and save HTML.

    Args:
        page_name: Name of the wiki page (e.g., "Crowns", "Leaders", "Stats")
        output_subdir: Subdirectory under structured_outputs_dir to save the HTML

    Returns:
        HTML content if successful, None otherwise
    """
    logger.info(f"Scraping {page_name} page...")
    url = f"{settings.wildfrost_wiki_base_url}/{page_name}"
    html_list = await scrape_multiple_links([url], max_concurrent=1)
    html = html_list[0] if html_list else None

    if not html:
        logger.warning(f"Failed to scrape {page_name} page")
        return None

    output_path = settings.structured_outputs_dir / output_subdir / f'{page_name}.html'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Saved {page_name} HTML to {output_path}")

    return html


def load_cached_html(page_name: str, output_subdir: str) -> str | None:
    """
    Load a previously scraped wiki page from disk.

    Args:
        page_name: Name of the wiki page (e.g., "Crowns", "Leaders", "Stats")
        output_subdir: Subdirectory under structured_outputs_dir where HTML was saved

    Returns:
        HTML content if file exists, None otherwise
    """
    path = settings.structured_outputs_dir / output_subdir / f'{page_name}.html'
    if not path.exists():
        logger.warning(f"Cached HTML not found: {path}")
        return None

    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    logger.info(f"Loaded cached {page_name} from {path}")
    return html
