"""Stage 2 of the WildFrostRAG ingestion pipeline: data enrichment.

Extracted from GraphBuilderService so enrichment is unit-testable and reusable
(by both the CLI-facing GraphBuilderService and the Dagster asset chain).
"""

from wildfrost_rag.data_processing.cards import CardInfo
from wildfrost_rag.data_processing.enrichment import enrich_cards_with_tribes
from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger


class EnrichmentService:
    """Stage 2: enrich card data with tribe exclusivity information."""

    def enrich(self, card_infos: list[CardInfo]) -> None:
        """Stage 2: Data Enrichment.

        Enriches card data with tribe exclusivity information.

        Args:
            card_infos: List of CardInfo objects to enrich (modified in-place)
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: DATA ENRICHMENT")
        logger.info("=" * 60)

        enrich_cards_with_tribes(
            card_infos=card_infos,
            companions_url=get_settings().scraping.companions_page_url,
            items_url=get_settings().scraping.items_page_url,
        )
