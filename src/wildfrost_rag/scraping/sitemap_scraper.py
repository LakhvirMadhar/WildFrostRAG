import asyncio
from asyncio import Semaphore
from typing import Any

import aiohttp
import requests
from bs4 import BeautifulSoup

from wildfrost_rag.core.logger import logger


def scrape_sitemap(sitemap_url: str) -> list[dict[str, Any]]:
    """Scrapes a sitemap XML file and extracts URLs with their last modification dates.

    Args:
        sitemap_url (str): The URL pointing to the sitemap.xml file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing:
            - 'url': The page URL as a string
            - 'last_updated': The last modification date as a string
            Returns an empty list if scraping fails.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    logger.info(f"Starting sitemap scrape for: {sitemap_url}")

    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        logger.info(f"Successfully fetched sitemap: {sitemap_url}")

        soup = BeautifulSoup(response.content, features="lxml-xml")
        url_tags = soup.find_all("url")
        logger.info(f"Found {len(url_tags)} URLs in sitemap")

        sitemap_links_to_scrape = []
        for url_tag in url_tags:
            loc_element = url_tag.find("loc")
            lastmod_element = url_tag.find("lastmod")

            if loc_element is None:
                logger.warning("Found URL tag without 'loc' element, skipping")
                continue

            loc_tag = loc_element.text.strip()
            lastmod_tag = lastmod_element.text.strip() if lastmod_element else ""

            if not loc_tag:
                logger.warning("Found URL tag without 'loc' element, skipping")
                continue

            dict_item = {"url": loc_tag, "last_updated": lastmod_tag}

            sitemap_links_to_scrape.append(dict_item)

        logger.info(f"Successfully parsed {len(sitemap_links_to_scrape)} URLs from sitemap")
        return sitemap_links_to_scrape

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to scrape sitemap {sitemap_url}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error while parsing sitemap {sitemap_url}: {e}")
        return []


def process_sitemap_urls(sitemap_urls: list[dict[str, Any]]) -> list[str]:
    """Extracts URL strings from sitemap data dictionaries.

    Args:
        sitemap_urls (List[Dict[str, Any]]): List of dictionaries containing
            sitemap data with 'url' keys.

    Returns:
        List[str]: List of URL strings extracted from the input dictionaries.
    """
    logger.info(f"Processing {len(sitemap_urls)} sitemap URLs")

    urls = [url["url"] for url in sitemap_urls]

    logger.info(f"Extracted {len(urls)} URL strings")
    return urls


async def scrape_single_link(
    session: aiohttp.ClientSession, semaphore: Semaphore, url: str
) -> str | None:
    """Scrapes a single URL asynchronously with semaphore-based concurrency control.

    Args:
        session (aiohttp.ClientSession): HTTP session for making requests.
        semaphore (Semaphore): Asyncio semaphore to limit concurrent requests.
        url (str): The URL to scrape.

    Returns:
        Optional[str]: HTML content of the scraped page, or None if scraping failed.
    """
    async with semaphore:
        try:
            logger.debug(f"Starting scrape for: {url}")
            async with session.get(url) as response:
                response.raise_for_status()

                html_output = await response.text()

                logger.info(f"Successfully scraped {url}")
                return html_output

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error has occurred for url: {url}\nError: {e}")
            return None


async def scrape_multiple_links(
    session: aiohttp.ClientSession, urls: list[str], max_concurrent: int = 5
) -> list[str]:
    """Scrape multiple URLs asynchronously with concurrent request limiting.

    Args:
        session: Shared HTTP session (constructed and owned by the caller) used
            for every request in this batch.
        urls: List of URL strings to scrape
        max_concurrent: Maximum number of simultaneous requests (default: 5)

    Returns:
        List[str]: List of HTML content from scraped pages. Failed requests
                   return None in their respective positions.
    """
    logger.info(f"Starting batch scrape of {len(urls)} URLs with max_concurrent={max_concurrent}")

    if not urls:
        logger.warning("No URLs provided for scraping")
        return []

    semaphore = Semaphore(max_concurrent)

    tasks = [scrape_single_link(session, semaphore, url) for url in urls]
    results = await asyncio.gather(*tasks)

    successful_scrapes = sum(1 for result in results if result is not None)
    logger.info(f"Batch scrape completed: {successful_scrapes}/{len(urls)} URLs successful")

    return [r for r in results if r is not None]
