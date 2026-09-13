"""Stage 1 of the WildFrostRAG ingestion pipeline: scraping and parsing.

Extracted from GraphBuilderService so scraping is unit-testable and reusable
(by both the CLI-facing GraphBuilderService and the Dagster asset chain)
without depending on any Neo4j resources - this stage never touches a driver.
"""

import json
import os
from collections.abc import Awaitable

from tqdm import tqdm

from wildfrost_rag.data_processing.cards import CardInfo, CardType
from wildfrost_rag.data_processing.generate_schemas import generate_card_type_html_schema
from wildfrost_rag.clients.domain_scrapers import (
    scrape_bells,
    scrape_bling,
    scrape_charms,
    scrape_clunker_prices,
    scrape_crowns,
    scrape_fight_pages,
    scrape_getting_started,
    scrape_individual_bell_pages,
    scrape_individual_charm_pages,
    scrape_individual_stat_pages,
    scrape_keywords,
    scrape_leaders,
    scrape_map,
    scrape_shades,
    scrape_shop,
    scrape_stats,
)
from wildfrost_rag.clients.wiki_scraper import clean_name_for_url
from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger
from wildfrost_rag.clients.sitemap_scraper import scrape_multiple_links
from wildfrost_rag.domain.scraping_types import FightEnemies, PageUrls
from wildfrost_rag.services.ingestion.pipeline_data import PipelineData


class ScrapingService:
    """Stage 1: scrape and parse card and domain page data from the Wildfrost Wiki."""

    def _generate_schema(self) -> dict[str, list[str]]:
        """Generate and save card type schema from wiki."""
        logger.info("Generating card type schema...")
        card_type_schema = generate_card_type_html_schema()

        settings = get_settings()
        schema_path = settings.paths.schemas_dir / "card_type_schema.json"
        settings.paths.schemas_dir.mkdir(parents=True, exist_ok=True)
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(card_type_schema, f, indent=4)
        logger.info(f"Schema saved to {schema_path}")
        return card_type_schema

    def _create_card_infos(self, card_type_schema: dict[str, list[str]]) -> list[CardInfo]:
        """Create CardInfo objects from schema (excluding leaders)."""
        card_infos = []
        for card_type, cards in card_type_schema.items():
            if card_type == "leaders":
                continue

            for card_name in cards:
                cleaned_name = clean_name_for_url(card_name)
                card_infos.append(
                    CardInfo(
                        card_name=card_name,
                        card_type=CardType.from_schema_key(card_type),
                        url=f"{get_settings().scraping.wildfrost_wiki_base_url}/{cleaned_name}",
                    )
                )

        logger.info(f"Created {len(card_infos)} CardInfo objects")
        return card_infos

    def _load_cached_cards(
        self, card_infos: list[CardInfo]
    ) -> tuple[list[CardInfo], list[CardInfo], int]:
        """Load card pages from cache, return parsed cards and cards needing scraping.

        Returns:
            Tuple of (parsed cards, cards to scrape, successful page count).
        """
        all_cards: list[CardInfo] = []
        successful_pages = 0
        cards_to_scrape = []

        for card_info in card_infos:
            html_path = card_info.save_path()
            if os.path.exists(html_path):
                with open(html_path, encoding="utf-8") as f:
                    html = f.read()
                parsed_cards = CardInfo.parse_html_multi_phase(
                    html=html, card_type=card_info.card_type, url=card_info.url
                )
                if parsed_cards:
                    successful_pages += 1
                    all_cards.extend(parsed_cards)
            else:
                cards_to_scrape.append(card_info)

        logger.info(f"Loaded {successful_pages} card pages from cache")
        return all_cards, cards_to_scrape, successful_pages

    async def _scrape_missing_cards(
        self, cards_to_scrape: list[CardInfo]
    ) -> tuple[list[CardInfo], int]:
        """Scrape and parse cards not found in cache.

        Returns:
            Tuple of (newly parsed cards, successful page count).
        """
        logger.info(f"Scraping {len(cards_to_scrape)} missing card pages...")
        urls = [card.url for card in cards_to_scrape]
        html_outputs = await scrape_multiple_links(
            urls, max_concurrent=get_settings().scraping.max_concurrent_requests
        )

        new_cards: list[CardInfo] = []
        successful = 0
        for card_info, html in tqdm(
            zip(cards_to_scrape, html_outputs, strict=False),
            total=len(cards_to_scrape),
            desc="Parsing HTML",
            unit="page",
        ):
            if html is None:
                continue
            parsed_cards = CardInfo.parse_html_multi_phase(
                html=html, card_type=card_info.card_type, url=card_info.url
            )
            if parsed_cards:
                successful += 1
                for card in parsed_cards:
                    card.save_html()
                new_cards.extend(parsed_cards)

        return new_cards, successful

    async def _load_card_pages(
        self, skip_scrape: bool
    ) -> tuple[list[CardInfo], dict[str, list[str]]]:
        """Generate card type schema and load/scrape individual card pages.

        Returns:
            Tuple of (all parsed CardInfo objects, card_type_schema dict).
        """
        card_type_schema = self._generate_schema()
        card_infos = self._create_card_infos(card_type_schema)
        all_cards, cards_to_scrape, successful_pages = self._load_cached_cards(card_infos)

        if cards_to_scrape and not skip_scrape:
            new_cards, new_successes = await self._scrape_missing_cards(cards_to_scrape)
            all_cards.extend(new_cards)
            successful_pages += new_successes
        elif cards_to_scrape:
            logger.warning(
                f"{len(cards_to_scrape)} card HTML files not found in cache (--skip-scrape)"
            )

        logger.info(
            f"Total: {successful_pages}/{len(card_infos)} card pages, "
            f"{len(all_cards)} CardInfo objects"
        )
        return all_cards, card_type_schema

    async def _scrape_and_collect_urls[T](
        self, coro: Awaitable[tuple[T, PageUrls]], page_urls: PageUrls
    ) -> T:
        """Await a scraper returning (value, urls); merge urls, return the value."""
        value, urls = await coro
        page_urls.update(urls)
        return value

    async def _merge_urls(self, coro: Awaitable[PageUrls], page_urls: PageUrls) -> None:
        """Await a scraper returning urls only; merge them."""
        page_urls.update(await coro)

    async def _scrape_domain_pages(self, card_type_schema: dict[str, list[str]]) -> PipelineData:
        """Scrape all domain pages (leaders, stats, keywords, shops, etc.).

        Collects parsed data and page URLs from each scraper into PipelineData.

        Args:
            card_type_schema: Schema dict used to extract boss/miniboss names for
                bling scraping
        """
        page_urls: PageUrls = {}

        leader_cards = await self._scrape_and_collect_urls(scrape_leaders(), page_urls)
        await self._scrape_and_collect_urls(scrape_crowns(), page_urls)
        await self._scrape_and_collect_urls(scrape_getting_started(), page_urls)

        stats = await self._scrape_and_collect_urls(scrape_stats(), page_urls)
        # Individual stat pages, for per-stat Documents (detailed mechanics)
        await self._merge_urls(scrape_individual_stat_pages(stats), page_urls)

        keywords = await self._scrape_and_collect_urls(scrape_keywords(), page_urls)

        boss_names = card_type_schema.get("bosses", [])
        miniboss_names = card_type_schema.get("minibosses", [])
        bling_drops = await self._scrape_and_collect_urls(
            scrape_bling(boss_names, miniboss_names), page_urls
        )

        woolly_snail_listings = await self._scrape_and_collect_urls(
            scrape_shop("The_Woolly_Snail", "shops"), page_urls
        )
        charm_merchant_listings = await self._scrape_and_collect_urls(
            scrape_shop("Charm_Merchant", "shops"), page_urls
        )
        clunker_prices = await self._scrape_and_collect_urls(scrape_clunker_prices(), page_urls)

        bells = await self._scrape_and_collect_urls(scrape_bells(), page_urls)
        # Individual bell pages, for per-bell Documents
        await self._merge_urls(scrape_individual_bell_pages(bells), page_urls)

        charms = await self._scrape_and_collect_urls(scrape_charms(), page_urls)
        # Individual charm pages, for per-charm Documents (Strategy sections, etc.)
        await self._merge_urls(scrape_individual_charm_pages(charms), page_urls)

        summons = await self._scrape_and_collect_urls(scrape_shades(), page_urls)

        zones, map_events, fight_slots, fight_page_mapping, map_urls = await scrape_map()
        page_urls.update(map_urls)

        fight_enemies: FightEnemies = {}
        if fight_page_mapping:
            fight_enemies = await self._scrape_and_collect_urls(
                scrape_fight_pages(fight_page_mapping), page_urls
            )

        return PipelineData(
            cards=leader_cards,
            stats=stats,
            keywords=keywords,
            bling_drops=bling_drops,
            woolly_snail_listings=woolly_snail_listings,
            charm_merchant_listings=charm_merchant_listings,
            clunker_prices=clunker_prices,
            bells=bells,
            charms=charms,
            summons=summons,
            zones=zones,
            map_events=map_events,
            fight_slots=fight_slots,
            fight_page_mapping=fight_page_mapping,
            fight_enemies=fight_enemies,
            page_urls=page_urls,
        )

    async def scrape(self, skip_scrape: bool = False) -> PipelineData:
        """Stage 1: Data Collection.

        Loads card data from cache if available, otherwise scrapes from wiki.
        When skip_scrape=True, only loads from cache (no network requests).

        Args:
            skip_scrape: If True, only use cached HTML files (no web requests)

        Returns:
            PipelineData containing all scraped and parsed data
        """
        logger.info("=" * 60)
        if skip_scrape:
            logger.info("STAGE 1: LOADING FROM CACHE (--skip-scrape)")
        else:
            logger.info("STAGE 1: WEB SCRAPING")
        logger.info("=" * 60)

        all_cards, card_type_schema = await self._load_card_pages(skip_scrape)

        # Domain pages (leaders, stats, keywords, shops, etc.)
        pipeline_data = await self._scrape_domain_pages(card_type_schema)

        # Merge card pages into pipeline data
        # Leader cards come from _scrape_domain_pages, all other cards from _load_card_pages
        pipeline_data.cards = all_cards + pipeline_data.cards

        # Card page URLs — built from all_cards (includes variants from parse_html_multi_phase)
        card_page_urls = {f"{card.sanitized_name()}.html": card.url for card in all_cards}
        pipeline_data.page_urls.update(card_page_urls)

        logger.info(f"Total cards: {len(pipeline_data.cards)}")
        logger.info(f"Built page URL mapping with {len(pipeline_data.page_urls)} entries")
        return pipeline_data
