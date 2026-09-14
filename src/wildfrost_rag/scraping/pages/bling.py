"""Scraping for enemy bling drops and shop/clunker prices."""

from functools import partial

import aiohttp

from wildfrost_rag.data_processing.bling import (
    EnemyBlingDrop,
    ShopListing,
    parse_bling_page,
    parse_clunker_prices,
    parse_shop_page,
)
from wildfrost_rag.domain.scraping_types import PageUrls
from wildfrost_rag.scraping._page_fetching import scrape_page


async def scrape_bling(
    session: aiohttp.ClientSession, boss_names: list[str], miniboss_names: list[str]
) -> tuple[list[EnemyBlingDrop], PageUrls]:
    """Parse the Bling page for enemy drop values (from cache or web)."""
    parser = partial(parse_bling_page, boss_names=boss_names, miniboss_names=miniboss_names)
    return await scrape_page(session, "Bling", "bling", parser)


async def scrape_shop(
    session: aiohttp.ClientSession, page_name: str, subdir: str
) -> tuple[list[ShopListing], PageUrls]:
    """Parse a shop page for item/charm listings (from cache or web)."""
    return await scrape_page(session, page_name, subdir, parse_shop_page)


async def scrape_clunker_prices(
    session: aiohttp.ClientSession,
) -> tuple[list[ShopListing], PageUrls]:
    """Parse the Clunkers page for clunker shop prices (from cache or web)."""
    return await scrape_page(session, "Clunkers", "clunkers_page", parse_clunker_prices)
