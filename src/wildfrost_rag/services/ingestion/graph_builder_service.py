"""Orchestration and business logic for the WildFrostRAG data ingestion pipeline.

Extracted from the CLI ingestion entry point so it's a thin wrapper around a
unit-testable service: argument parsing and Neo4j driver construction stay
in the CLI script; everything else - scraping, enrichment, graph population,
and document ingestion - lives here.

Follows the same dependency-injection pattern as CardRepository, DocumentRepository,
and BaseNeo4jRetriever: the Neo4j Driver is constructed once by the caller and passed
in, never created internally, so the service is easy to unit-test with a fake driver.

The four stages' actual logic lives in ScrapingService, EnrichmentService,
GraphPopulationService, and DocumentIngestionService (one file each, in this same
package) so each stage is independently unit-testable and so the Dagster asset chain
in defs/ingestion/assets.py can call the same stages directly. GraphBuilderService
composes them and keeps its stage_N method names/signatures stable for existing
callers (the CLI, and this file's own test suite).
"""

from neo4j import Driver

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.cards import CardInfo
from wildfrost_rag.repositories.graph_builder import clear_database
from wildfrost_rag.services.ingestion.document_ingestion_service import (
    DocumentIngestionService,
)
from wildfrost_rag.services.ingestion.enrichment_service import EnrichmentService
from wildfrost_rag.services.ingestion.graph_population_service import (
    GraphPopulationService,
)
from wildfrost_rag.services.ingestion.pipeline_data import PipelineData
from wildfrost_rag.services.ingestion.scraping_service import ScrapingService


class GraphBuilderService:
    """Orchestrates the WildFrostRAG ETL pipeline: scrape, enrich, populate, ingest.

    Follows dependency injection - the Neo4j Driver is constructed externally
    (by the calling script) and passed in, never created here. This keeps the
    service unit-testable with a fake driver/session and no live Neo4j instance.

    Delegates each stage to its own service class (ScrapingService, EnrichmentService,
    GraphPopulationService, DocumentIngestionService); this class's job is orchestration
    only, so it stays a stable entry point for the CLI while each stage's logic can be
    tested, and reused by the Dagster asset chain, independently.
    """

    def __init__(self, driver: Driver) -> None:
        """Initialize the service.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
        """
        self.driver = driver
        self._scraping_service = ScrapingService()
        self._enrichment_service = EnrichmentService()
        self._graph_population_service = GraphPopulationService(driver)
        self._document_ingestion_service = DocumentIngestionService(driver)

    async def run(
        self,
        *,
        skip_scrape: bool,
        skip_graph: bool,
        skip_vectors: bool,
        clear_db: bool,
        no_chunking: bool,
    ) -> None:
        """Run the full ingestion pipeline, mirroring the original script's main().

        Args:
            skip_scrape: If True, only use cached HTML files (no web requests) in Stage 1
            skip_graph: If True, skip Stage 3 (Neo4j graph population)
            skip_vectors: If True, skip Stage 4 (document ingestion)
            clear_db: If True, clear the entire Neo4j database before starting
            no_chunking: If True, ingest full documents in Stage 4 instead of chunking
        """
        settings = get_settings()
        logger.info("Starting WildFrostRAG data ingestion pipeline")
        logger.info(
            f"Configuration: neo4j_uri=<redacted>, openai_model={settings.openai.model_name}, "
            f"embedding_model={settings.embedding.model_name}, "
            f"wiki_base_url={settings.scraping.wildfrost_wiki_base_url}"
        )

        if clear_db:
            logger.warning("⚠️  CLEARING ENTIRE NEO4J DATABASE ⚠️")
            with self.driver.session() as session:
                session.execute_write(clear_database)
            logger.info("✅ Database cleared successfully")

        settings.create_directories()

        pipeline_data = await self.stage_1_scrape_cards(skip_scrape=skip_scrape)

        self.stage_2_enrich_data(pipeline_data.cards)

        if not skip_graph:
            self.stage_3_populate_graph(pipeline_data)
        else:
            logger.info("Skipping graph population (--skip-graph flag set)")

        if not skip_vectors:
            # If --no-chunking is passed, split_text should be False
            split_text = not no_chunking
            self.stage_4_document_ingestion(pipeline_data, split_text=split_text)
        else:
            logger.info("Skipping document ingestion (--skip-vectors flag set)")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

    async def stage_1_scrape_cards(self, skip_scrape: bool = False) -> PipelineData:
        """Stage 1: Data Collection. Delegates to ScrapingService.

        Args:
            skip_scrape: If True, only use cached HTML files (no web requests)

        Returns:
            PipelineData containing all scraped and parsed data
        """
        return await self._scraping_service.scrape(skip_scrape)

    def stage_2_enrich_data(self, card_infos: list[CardInfo]) -> None:
        """Stage 2: Data Enrichment. Delegates to EnrichmentService.

        Args:
            card_infos: List of CardInfo objects to enrich (modified in-place)
        """
        self._enrichment_service.enrich(card_infos)

    def stage_3_populate_graph(self, data: PipelineData) -> None:
        """Stage 3: Neo4j Graph Population. Delegates to GraphPopulationService.

        Args:
            data: PipelineData containing all scraped data to populate the graph
        """
        self._graph_population_service.populate(data)

    def stage_4_document_ingestion(
        self, pipeline_data: PipelineData, split_text: bool = True
    ) -> None:
        """Stage 4: Document Ingestion. Delegates to DocumentIngestionService.

        Args:
            pipeline_data: Pipeline data containing cards and fight_page_mapping
            split_text: If True, splits documents into chunks. If False, ingests full
                documents.
        """
        self._document_ingestion_service.ingest(pipeline_data, split_text=split_text)
