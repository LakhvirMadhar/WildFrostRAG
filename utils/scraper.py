import asyncio
from asyncio import Semaphore
from bs4 import BeautifulSoup
import aiohttp
import requests
from typing import List, Dict, Any


def scrape_sitemap(sitemap_url: str) -> List[Dict[str, Any]]:
    """
    Scrapes the sitemap for a given link.

    Args:
        sitemap_url (str): The URL to the sitemap.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
        contains the URL and its last modification date.
    """
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, features='lxml-xml')

        url_tags = soup.find_all('url')

        sitemap_links_to_scrape = []
        for url_tag in url_tags:

            loc_tag = url_tag.find('loc').text.strip()
            lastmod_tag = url_tag.find('lastmod').text.strip()

            dict_item = {
                'url': loc_tag,
                'last_updated': lastmod_tag
            }

            sitemap_links_to_scrape.append(dict_item)

        return sitemap_links_to_scrape

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []


# async def scrape_multiple_links
