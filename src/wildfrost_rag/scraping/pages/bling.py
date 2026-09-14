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
from wildfrost_rag.scraping.page_fetching import scrape_page

_BLING_PAGE_NAME = "Bling"
_BLING_CACHE_SUBDIR = "bling"
_CLUNKERS_PAGE_NAME = "Clunkers"
_CLUNKERS_CACHE_SUBDIR = "clunkers_page"


async def scrape_bling(
    session: aiohttp.ClientSession, boss_names: list[str], miniboss_names: list[str]
) -> tuple[list[EnemyBlingDrop], PageUrls]:
    """Parse the Bling page for enemy drop values (from cache or web)."""
    parser = partial(parse_bling_page, boss_names=boss_names, miniboss_names=miniboss_names)
    return await scrape_page(session, _BLING_PAGE_NAME, _BLING_CACHE_SUBDIR, parser)


async def scrape_shop(
    session: aiohttp.ClientSession, page_name: str, subdir: str
) -> tuple[list[ShopListing], PageUrls]:
    """Parse a shop page for item/charm listings (from cache or web).

    page_name/subdir are caller-supplied (not a module constant): each shop
    is a distinct wiki page (e.g. "The_Woolly_Snail", "Charm_Merchant") with
    no fixed identity of its own here.
    """
    return await scrape_page(session, page_name, subdir, parse_shop_page)


async def scrape_clunker_prices(
    session: aiohttp.ClientSession,
) -> tuple[list[ShopListing], PageUrls]:
    """Parse the Clunkers page for clunker shop prices (from cache or web)."""
    return await scrape_page(
        session, _CLUNKERS_PAGE_NAME, _CLUNKERS_CACHE_SUBDIR, parse_clunker_prices
    )
