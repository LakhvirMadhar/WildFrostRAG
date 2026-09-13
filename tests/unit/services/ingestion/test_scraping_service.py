"""Unit tests for ScrapingService._scrape_domain_pages.

Every domain scraper is patched with a fake returning canned (value, urls) -
or, for scrape_map/scrape_fight_pages, their own distinct shapes - so these
tests prove the _run/_merge_urls extraction preserves the original
sequential-call behavior: every scraper invoked exactly once, every url
dict merged into one page_urls, and PipelineData's fields populated from
the right scraper's result.

pytest-asyncio isn't a project dependency, so each async call is driven
with asyncio.run() from an ordinary (synchronous) test function.
"""

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from wildfrost_rag.services.ingestion.scraping_service import ScrapingService


def _make_patches(fight_page_mapping: dict[str, str]) -> dict[str, Any]:
    async def _pair(value: object, url_key: str) -> tuple[object, dict[str, str]]:
        return value, {url_key: f"https://example.test/{url_key}"}

    async def _urls_only(url_key: str) -> dict[str, str]:
        return {url_key: f"https://example.test/{url_key}"}

    async def _map_result() -> tuple[
        list[str], list[str], list[str], dict[str, str], dict[str, str]
    ]:
        return (
            ["a_zone"],
            ["a_map_event"],
            ["a_fight_slot"],
            fight_page_mapping,
            {"map": "https://example.test/map"},
        )

    return {
        "scrape_leaders": lambda: _pair(["leader_card"], "leaders"),
        "scrape_crowns": lambda: _pair("discarded", "crowns"),
        "scrape_getting_started": lambda: _pair("discarded", "getting_started"),
        "scrape_stats": lambda: _pair(["a_stat"], "stats"),
        "scrape_individual_stat_pages": lambda stats: _urls_only("individual_stats"),
        "scrape_keywords": lambda: _pair(["a_keyword"], "keywords"),
        "scrape_bling": lambda bosses, minibosses: _pair(["a_bling_drop"], "bling"),
        "scrape_shop": lambda name, category: _pair([f"{name}_listing"], f"shop_{name}"),
        "scrape_clunker_prices": lambda: _pair(["a_clunker_price"], "clunker"),
        "scrape_bells": lambda: _pair(["a_bell"], "bells"),
        "scrape_individual_bell_pages": lambda bells: _urls_only("individual_bells"),
        "scrape_charms": lambda: _pair(["a_charm"], "charms"),
        "scrape_individual_charm_pages": lambda charms: _urls_only("individual_charms"),
        "scrape_shades": lambda: _pair(["a_summon"], "shades"),
        "scrape_map": lambda: _map_result(),
        "scrape_fight_pages": lambda mapping: _pair({"boss": ["enemy"]}, "fight_pages"),
    }


@pytest.mark.parametrize("has_fights", [True, False])
def test_scrape_domain_pages_calls_every_scraper_and_merges_urls(has_fights: bool) -> None:
    """Every domain scraper runs once; every url dict ends up in page_urls."""
    fight_page_mapping = {"Some_Fight.html": "fight1"} if has_fights else {}
    patches = _make_patches(fight_page_mapping)

    with patch.multiple("wildfrost_rag.services.ingestion.scraping_service", **patches):
        service = ScrapingService()
        # Fakes return plain strings, not real StatInfo/CardInfo/etc. instances -
        # Any reflects that this test verifies wiring/sequencing, not domain-object
        # fidelity, which the fakes deliberately don't provide.
        result: Any = asyncio.run(
            service._scrape_domain_pages({"bosses": ["Boss"], "minibosses": []})
        )

    assert result.cards == ["leader_card"]
    assert result.stats == ["a_stat"]
    assert result.keywords == ["a_keyword"]
    assert result.bling_drops == ["a_bling_drop"]
    assert result.woolly_snail_listings == ["The_Woolly_Snail_listing"]
    assert result.charm_merchant_listings == ["Charm_Merchant_listing"]
    assert result.clunker_prices == ["a_clunker_price"]
    assert result.bells == ["a_bell"]
    assert result.charms == ["a_charm"]
    assert result.summons == ["a_summon"]
    assert result.zones == ["a_zone"]
    assert result.map_events == ["a_map_event"]
    assert result.fight_slots == ["a_fight_slot"]
    assert result.fight_page_mapping == fight_page_mapping

    if has_fights:
        assert result.fight_enemies == {"boss": ["enemy"]}
        assert "fight_pages" in result.page_urls
    else:
        assert result.fight_enemies == {}
        assert "fight_pages" not in result.page_urls

    # Every non-conditional scraper's url key made it into the merged dict.
    expected_keys = {
        "leaders",
        "crowns",
        "getting_started",
        "stats",
        "individual_stats",
        "keywords",
        "bling",
        "shop_The_Woolly_Snail",
        "shop_Charm_Merchant",
        "clunker",
        "bells",
        "individual_bells",
        "charms",
        "individual_charms",
        "shades",
        "map",
    }
    assert expected_keys.issubset(result.page_urls.keys())
