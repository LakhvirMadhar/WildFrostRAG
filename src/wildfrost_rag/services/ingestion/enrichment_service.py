"""Stage 2 of the WildFrostRAG ingestion pipeline: data enrichment.

Extracted from GraphBuilderService so enrichment is unit-testable and reusable
(by both the CLI-facing GraphBuilderService and the Dagster asset chain).
"""

import aiohttp

from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.cards import CardInfo
from wildfrost_rag.data_processing.enrichment import enrich_cards_with_tribes
from wildfrost_rag.scraping.page_fetching import get_html

_COMPANIONS_PAGE_NAME = "Companions"
_COMPANIONS_CACHE_SUBDIR = "companions"
_ITEMS_PAGE_NAME = "Items"
_ITEMS_CACHE_SUBDIR = "items"


class EnrichmentService:
    """Stage 2: enrich card data with tribe exclusivity information."""

    async def enrich(self, session: aiohttp.ClientSession, card_infos: list[CardInfo]) -> None:
        """Stage 2: Data Enrichment.

        Enriches card data with tribe exclusivity information.

        Args:
            session: Shared HTTP session (constructed and owned by the caller).
            card_infos: List of CardInfo objects to enrich (modified in-place)
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: DATA ENRICHMENT")
        logger.info("=" * 60)

        companions_html, _ = await get_html(
            session, _COMPANIONS_PAGE_NAME, _COMPANIONS_CACHE_SUBDIR
        )
        items_html, _ = await get_html(session, _ITEMS_PAGE_NAME, _ITEMS_CACHE_SUBDIR)

        if companions_html is None:
            raise RuntimeError(f"Failed to fetch {_COMPANIONS_PAGE_NAME} page for tribe enrichment")
        if items_html is None:
            raise RuntimeError(f"Failed to fetch {_ITEMS_PAGE_NAME} page for tribe enrichment")

        enrich_cards_with_tribes(
            card_infos=card_infos,
            companions_html=companions_html,
            items_html=items_html,
        )
