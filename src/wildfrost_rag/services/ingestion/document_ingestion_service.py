"""Stage 4 of the WildFrostRAG ingestion pipeline: document ingestion.

Extracted from GraphBuilderService so document ingestion is unit-testable and
reusable (by both the CLI-facing GraphBuilderService and the Dagster asset
chain). Follows the same dependency-injection pattern as CardRepository,
DocumentRepository, and BaseNeo4jRetriever: the Neo4j Driver is constructed
externally and passed in, never created here.
"""

import os
from collections.abc import Callable
from typing import ClassVar

from neo4j import Driver, Session

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger
from wildfrost_rag.data_processing.html_splitter import process_html_files
from wildfrost_rag.repositories.neo4j_indexes import (
    create_fulltext_index,
    wait_for_index_population,
)
from wildfrost_rag.repositories.vector_store import (
    ingest_documents_into_neo4j,
    link_documents_to_bells,
    link_documents_to_bling,
    link_documents_to_cards,
    link_documents_to_charms,
    link_documents_to_crowns,
    link_documents_to_fights,
    link_documents_to_map,
    link_documents_to_shades,
    link_documents_to_shops,
    link_documents_to_stats,
)
from wildfrost_rag.services.ingestion.pipeline_data import PipelineData


class DocumentIngestionService:
    """Stage 4: chunk HTML into Document nodes and link them to graph nodes."""

    # (label, linker) pairs for the Document-to-node linking step. Each linker
    # takes the active session and returns the number of relationships it created.
    LINKERS: ClassVar[list[tuple[str, Callable[[Session], int]]]] = [
        ("cards", link_documents_to_cards),
        ("crowns", link_documents_to_crowns),
        ("stats", link_documents_to_stats),
        ("charms", link_documents_to_charms),
        ("shades", link_documents_to_shades),
        ("map nodes", link_documents_to_map),
        ("fight nodes", link_documents_to_fights),
        ("shop nodes", link_documents_to_shops),
        ("bling", link_documents_to_bling),
        ("bells", link_documents_to_bells),
    ]

    def __init__(self, driver: Driver) -> None:
        """Initialize the service.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
        """
        self.driver = driver

    def ingest(self, pipeline_data: PipelineData, split_text: bool = True) -> None:
        """Stage 4: Document Ingestion.

        Chunks HTML documents and ingests into Neo4j as Document nodes.
        Creates full-text search index. Does NOT generate embeddings.
        Use add_embeddings.py separately to add embeddings.

        Args:
            pipeline_data: Pipeline data containing cards and fight_page_mapping
            split_text: If True, splits documents into chunks. If False, ingests full
                documents.
        """
        logger.info("=" * 60)
        logger.info("STAGE 4: DOCUMENT INGESTION")
        if not split_text:
            logger.info("(Full Document Mode: No Chunking)")
        logger.info("=" * 60)

        # Collect all HTML file paths
        logger.info("Collecting HTML files...")
        settings = get_settings()
        all_html_filepaths = []
        for root, _dirs, files in os.walk(settings.paths.structured_outputs_dir):
            for file in files:
                if file.endswith(".html"):
                    filepath = os.path.join(root, file)
                    all_html_filepaths.append(filepath)

        logger.info(f"Found {len(all_html_filepaths)} HTML files to process")

        # Chunk the HTML documents
        logger.info(f"Processing HTML documents (split_text={split_text})...")
        all_document_chunks = process_html_files(all_html_filepaths, split_text=split_text)
        logger.info(f"Created {len(all_document_chunks)} document objects")

        url_lookup = pipeline_data.page_urls
        logger.info(f"URL lookup has {len(url_lookup)} entries")

        # Single session for all document operations
        with self.driver.session() as session:
            # Ingest into Neo4j (no embeddings)
            logger.info("Ingesting documents into Neo4j...")
            ingest_documents_into_neo4j(
                session=session,
                document_chunks=all_document_chunks,
                url_lookup=url_lookup,
            )

            # Create full-text search index
            logger.info("Creating full-text search index...")
            create_fulltext_index(
                session=session,
                index_name=settings.embedding.fulltext_index_name,
                node_label="Document",
                text_property="text",
            )

            # Wait for index to populate
            wait_for_index_population(seconds=5)

            # Link Document nodes to their corresponding domain nodes in the knowledge graph.
            for label, linker in self.LINKERS:
                logger.info(f"Linking documents to {label} in knowledge graph...")
                link_count = linker(session)
                logger.info(f"Linked {link_count} documents to {label}")

        logger.info("Document ingestion complete")
