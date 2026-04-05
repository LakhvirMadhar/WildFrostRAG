import os
import re
import asyncio
from asyncio import Semaphore
from bs4 import BeautifulSoup, Comment
import aiohttp
import requests
from typing import List, Dict, Any, Optional
from utils.logger import logger

def scrape_sitemap(sitemap_url: str) -> List[Dict[str, Any]]:
    """
    Scrapes a sitemap XML file and extracts URLs with their last modification dates.

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
        response = requests.get(sitemap_url)
        response.raise_for_status()
        logger.info(f"Successfully fetched sitemap: {sitemap_url}")

        soup = BeautifulSoup(response.content, features='lxml-xml')
        url_tags = soup.find_all('url')
        logger.info(f"Found {len(url_tags)} URLs in sitemap")

        sitemap_links_to_scrape = []
        for url_tag in url_tags:

            loc_tag = url_tag.find('loc').text.strip()
            lastmod_tag = url_tag.find('lastmod').text.strip()

            if loc_tag is None:
                logger.warning("Found URL tag without 'loc' element, skipping")
                continue

            dict_item = {
                'url': loc_tag,
                'last_updated': lastmod_tag
            }

            sitemap_links_to_scrape.append(dict_item)

        logger.info(f"Successfully parsed {len(sitemap_links_to_scrape)} URLs from sitemap")
        return sitemap_links_to_scrape

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to scrape sitemap {sitemap_url}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error while parsing sitemap {sitemap_url}: {e}")
        return []


def process_sitemap_urls(sitemap_urls: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts URL strings from sitemap data dictionaries.

    Args:
        sitemap_urls (List[Dict[str, Any]]): List of dictionaries containing
            sitemap data with 'url' keys.

    Returns:
        List[str]: List of URL strings extracted from the input dictionaries.
    """
    logger.info(f"Processing {len(sitemap_urls)} sitemap URLs")

    urls = [url['url'] for url in sitemap_urls]

    logger.info(f"Extracted {len(urls)} URL strings")
    return urls


async def scrape_single_link(session: aiohttp.ClientSession, semaphore: Semaphore, url: str) -> Optional[str]:
    """
    Scrapes a single URL asynchronously with semaphore-based concurrency control.

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


async def scrape_multiple_links(urls:List[str], max_concurrent: int = 5) -> List[str]:
    """
    Scrape multiple URLs asynchronously with concurrent request limiting

    Args:
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

    async with aiohttp.ClientSession() as session:
        tasks = [scrape_single_link(session, semaphore, url) for url in urls]
        results = await asyncio.gather(*tasks)

        successful_scrapes = sum(1 for result in results if result is not None)
        logger.info(f"Batch scrape completed: {successful_scrapes}/{len(urls)} URLs successful")

        return results


def process_raw_html_output(html_output: str, sub_directory: str='raw_htmls') -> Optional[str]:
    """
    Processes HTML content by cleaning it and saving it to a file.

    Args:
        html_output (str): Raw HTML content to process.
        sub_directory (str):
    Returns:
        Optional[str]: The filename where the HTML was saved, or None if processing failed.
    """
    soup = BeautifulSoup(html_output, 'html.parser')

    # Remove the comments from the HTML
    comments = soup.find_all(string=lambda text:isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    # Extract title & remove characters that are invalid in filenames
    title = soup.find('title').text if soup.find('title') else "untitled"
    sanitized_title = re.sub(r'[\\/:*?"<>|]', '', title)

    if not sanitized_title:
        sanitized_title = "output"

    # Sanitize sub_directory to prevent path traversal
    safe_sub_directory = os.path.normpath(sub_directory).replace('..', '').replace('/', '_').replace('\\', '_')
    filename = f"data/{safe_sub_directory}/{sanitized_title}.html"

    # Ensure the filename is within the expected directory
    full_path = os.path.abspath(filename)
    expected_prefix = os.path.abspath('data')
    if not full_path.startswith(expected_prefix):
        logger.error(f"Invalid path detected: {filename}")
        return None

    # save the html
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())


def save_raw_html_outputs(html_outputs: List[Optional[str]], sub_directory: str) -> None:
    """
    Processes and saves multiple HTML outputs to individual files.

    Args:
        html_outputs: List of HTML content strings from scraping results
        sub_directory:
    Returns:
        None: This function performs file I/O operations but doesn't return a value.
    """
    logger.info(f"Starting to process {len(html_outputs)} HTML outputs")
    for html_output in html_outputs:
        process_raw_html_output(html_output, sub_directory)


