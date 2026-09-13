"""Materializes the ingestion asset chain (scraped_cards -> ... -> neo4j_documents).

Proves the Dagster DAG wiring and dependency injection work end-to-end with fakes
at the I/O boundary: ScrapingService.scrape is patched to avoid real network/HTML
scraping, and Neo4jResource.get_driver is patched to yield a MagicMock(spec=Driver)
instead of a live Neo4j connection. It does NOT prove real Neo4j node/relationship
counts match a live run - that needs a real Neo4j instance and real network
scraping, neither of which exist in this sandbox.

Run in isolation (not as part of the full suite) to avoid depending on some other,
unrelated test file happening to import wildfrost_rag.services.ingestion.graph_builder_service
first and incidentally resolving its circular import with services/ingestion's four
new service modules - see the known-issue note in graph_builder_service.py.
"""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from unittest.mock import MagicMock, patch

import pytest
from dagster import materialize
from neo4j import Driver

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.data_processing.cards import CardInfo, CardType
from wildfrost_rag.defs.ingestion.assets import (
    enriched_cards,
    neo4j_documents,
    neo4j_graph,
    scraped_cards,
)
from wildfrost_rag.defs.resources import Neo4jResource
from wildfrost_rag.services.ingestion.pipeline_data import PipelineData


@pytest.fixture(autouse=True)
def _fake_neo4j_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub NEO4J_* env vars so get_settings() validates without a real Neo4j instance.

    EnrichmentService.enrich() reads get_settings().scraping.* fields; get_settings()
    validates the whole Settings model, including required Neo4j fields, even though
    this stage never touches Neo4j. Same pattern as tests/unit/defs/test_resources.py.
    """
    monkeypatch.setenv("NEO4J_URI", "bolt://fake-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "fake")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_pipeline_data() -> PipelineData:
    """A small, fake PipelineData standing in for a real scrape (2-3 fake cards)."""
    cards = [
        CardInfo(
            card_name="Fake Card One", card_type=CardType.COMPANIONS, url="https://example.test/1"
        ),
        CardInfo(
            card_name="Fake Card Two", card_type=CardType.COMPANIONS, url="https://example.test/2"
        ),
        CardInfo(
            card_name="Fake Card Three", card_type=CardType.ITEMS, url="https://example.test/3"
        ),
    ]
    return PipelineData(cards=cards, page_urls={"fake.html": "https://example.test/fake"})


class _FakeNeo4jResource(Neo4jResource):
    """A Neo4jResource whose driver is a MagicMock(spec=Driver), never a live connection."""

    def get_driver(self) -> AbstractContextManager[Driver]:
        @contextmanager
        def _fake_driver_context() -> Iterator[Driver]:
            # session.run(...).single() needs to look like a real (empty) Neo4j record:
            # not None (query_utils.single_value raises RuntimeError on None), but
            # indexable with any key and returning an int (repositories do arithmetic and
            # comparisons on the value) - a defaultdict(int) satisfies both call patterns
            # used across the linking repositories without hand-mocking each query's shape.
            fake_session = MagicMock()
            fake_session.run.return_value.single.return_value = defaultdict(int)

            driver = MagicMock(spec=Driver)
            session_context = MagicMock()
            session_context.__enter__ = MagicMock(return_value=fake_session)
            session_context.__exit__ = MagicMock(return_value=False)
            driver.session.return_value = session_context
            yield driver

        return _fake_driver_context()


def test_ingestion_asset_chain_materializes_in_dependency_order() -> None:
    """All 4 assets materialize successfully, scraped_cards -> ... -> neo4j_documents."""
    with patch(
        "wildfrost_rag.defs.ingestion.assets.ScrapingService.scrape",
        return_value=_fake_pipeline_data(),
    ):
        result = materialize(
            [scraped_cards, enriched_cards, neo4j_graph, neo4j_documents],
            resources={"neo4j": _FakeNeo4jResource()},
        )

    assert result.success

    materialized_order = []
    for event in result.get_asset_materialization_events():
        assert event.asset_key is not None
        materialized_order.append(event.asset_key.to_user_string())

    assert materialized_order == [
        "scraped_cards",
        "enriched_cards",
        "neo4j_graph",
        "neo4j_documents",
    ]
