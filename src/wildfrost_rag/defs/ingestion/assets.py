"""Dagster asset chain for the WildFrostRAG ingestion pipeline.

Mirrors GraphBuilderService's four stages as a chain of @asset functions so
Dagster can infer the pipeline DAG (scraped_cards -> enriched_cards ->
neo4j_graph -> neo4j_documents) instead of GraphBuilderService.run() hand-
orchestrating it. Each asset delegates to the same service classes
GraphBuilderService itself delegates to, so there is exactly one copy of the
business logic and two consumers (the CLI's GraphBuilderService, and this
asset chain).

Known simplification: the original stage_1_scrape_cards took a skip_scrape
CLI flag; scraped_cards below always scrapes fresh (skip_scrape=False).
Making that configurable via Dagster's asset Config mechanism is out of scope
for this pass.
"""

from dagster import asset

from wildfrost_rag.defs.resources import Neo4jResource
from wildfrost_rag.services.ingestion.document_ingestion_service import DocumentIngestionService
from wildfrost_rag.services.ingestion.enrichment_service import EnrichmentService
from wildfrost_rag.services.ingestion.pipeline_data import PipelineData
from wildfrost_rag.services.ingestion.graph_population_service import GraphPopulationService
from wildfrost_rag.services.ingestion.scraping_service import ScrapingService


@asset
async def scraped_cards() -> PipelineData:
    """Stage 1: scrape/parse card and domain page data from the Wildfrost Wiki."""
    return await ScrapingService().scrape()


@asset
def enriched_cards(scraped_cards: PipelineData) -> PipelineData:
    """Stage 2: enrich cards with tribe-exclusivity info."""
    EnrichmentService().enrich(scraped_cards.cards)
    return scraped_cards


@asset
def neo4j_graph(neo4j: Neo4jResource, enriched_cards: PipelineData) -> None:
    """Stage 3: populate the Neo4j knowledge graph from enriched card data."""
    with neo4j.get_driver() as driver:
        GraphPopulationService(driver).populate(enriched_cards)


@asset(deps=[neo4j_graph])
def neo4j_documents(neo4j: Neo4jResource, enriched_cards: PipelineData) -> None:
    """Stage 4: chunk HTML into Document nodes, link to nodes neo4j_graph created."""
    with neo4j.get_driver() as driver:
        DocumentIngestionService(driver).ingest(enriched_cards)
