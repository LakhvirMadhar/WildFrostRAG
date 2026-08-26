"""Unit tests for GraphBuilderService's run() orchestration (T6.2).

GraphBuilderService.run() reproduces what scripts/ingest_data.py's main() used to do
directly: clear the database when asked, always run Stage 1 and Stage 2, and run
Stages 3/4 unless their skip flags are set. These tests exercise that orchestration
in isolation - every stage method is monkeypatched away on the service instance, and
a fake Driver/Session stands in for Neo4j - so only run()'s control flow is under
test, with zero Neo4j or network access required.

pytest-asyncio isn't a project dependency, so each async run() call is driven with
asyncio.run() from an ordinary (synchronous) test function.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j import Driver, Session

from wildfrost_rag.services.ingestion.graph_builder_service import (
    GraphBuilderService,
    PipelineData,
)


def _make_fake_driver(session: Session) -> Driver:
    """Build a fake Driver whose session() context manager yields `session`."""
    driver = MagicMock(spec=Driver)
    session_context = MagicMock()
    session_context.__enter__ = MagicMock(return_value=session)
    session_context.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = session_context
    return driver


def _make_service(monkeypatch: pytest.MonkeyPatch, session: Session) -> GraphBuilderService:
    """Build a GraphBuilderService with every stage method stubbed out."""
    driver = _make_fake_driver(session)
    service = GraphBuilderService(driver=driver)

    monkeypatch.setattr(
        service, "stage_1_scrape_cards", AsyncMock(return_value=PipelineData(page_urls={}))
    )
    monkeypatch.setattr(service, "stage_2_enrich_data", MagicMock())
    monkeypatch.setattr(service, "stage_3_populate_graph", MagicMock())
    monkeypatch.setattr(service, "stage_4_document_ingestion", MagicMock())

    fake_settings = MagicMock()
    monkeypatch.setattr(
        "wildfrost_rag.services.ingestion.graph_builder_service.get_settings",
        lambda: fake_settings,
    )

    return service


def _run(service: GraphBuilderService, **kwargs: bool) -> None:
    """Drive service.run() to completion via asyncio.run()."""
    asyncio.run(service.run(**kwargs))


def test_constructor_stores_the_injected_driver_without_constructing_one() -> None:
    """The service never builds its own driver - it only holds the one it's given."""
    fake_driver = MagicMock(spec=Driver)

    service = GraphBuilderService(driver=fake_driver)

    assert service.driver is fake_driver


def test_run_always_calls_stage_1_and_stage_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 1 (scrape) and Stage 2 (enrich) run regardless of the skip flags."""
    fake_session = MagicMock(spec=Session)
    service = _make_service(monkeypatch, fake_session)

    _run(
        service,
        skip_scrape=True,
        skip_graph=True,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    service.stage_1_scrape_cards.assert_awaited_once_with(skip_scrape=True)
    service.stage_2_enrich_data.assert_called_once()


def test_run_calls_stage_3_when_skip_graph_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 3 (graph population) runs when --skip-graph is not set."""
    fake_session = MagicMock(spec=Session)
    service = _make_service(monkeypatch, fake_session)

    _run(
        service,
        skip_scrape=False,
        skip_graph=False,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    service.stage_3_populate_graph.assert_called_once()


def test_run_skips_stage_3_when_skip_graph_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 3 (graph population) is skipped when --skip-graph is set."""
    fake_session = MagicMock(spec=Session)
    service = _make_service(monkeypatch, fake_session)

    _run(
        service,
        skip_scrape=False,
        skip_graph=True,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    service.stage_3_populate_graph.assert_not_called()


@pytest.mark.parametrize(
    ("no_chunking", "expected_split_text"),
    [(False, True), (True, False)],
)
def test_run_calls_stage_4_with_split_text_derived_from_no_chunking(
    monkeypatch: pytest.MonkeyPatch, no_chunking: bool, expected_split_text: bool
) -> None:
    """--no-chunking flips split_text to False; its absence keeps chunking on."""
    fake_session = MagicMock(spec=Session)
    service = _make_service(monkeypatch, fake_session)

    _run(
        service,
        skip_scrape=False,
        skip_graph=True,
        skip_vectors=False,
        clear_db=False,
        no_chunking=no_chunking,
    )

    args, kwargs = service.stage_4_document_ingestion.call_args
    assert kwargs.get("split_text", args[-1] if args else None) == expected_split_text


def test_run_skips_stage_4_when_skip_vectors_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 4 (document ingestion) is skipped when --skip-vectors is set."""
    fake_session = MagicMock(spec=Session)
    service = _make_service(monkeypatch, fake_session)

    _run(
        service,
        skip_scrape=False,
        skip_graph=True,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    service.stage_4_document_ingestion.assert_not_called()


def test_run_clears_the_database_when_clear_db_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """--clear-db runs clear_database against the injected driver's session."""
    fake_session = MagicMock(spec=Session)
    service = _make_service(monkeypatch, fake_session)

    _run(
        service,
        skip_scrape=True,
        skip_graph=True,
        skip_vectors=True,
        clear_db=True,
        no_chunking=False,
    )

    fake_session.execute_write.assert_called_once()
    called_fn: Any = fake_session.execute_write.call_args.args[0]
    assert called_fn.__name__ == "clear_database"


def test_run_does_not_touch_the_database_when_clear_db_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without --clear-db, run() never opens a session to clear anything."""
    fake_session = MagicMock(spec=Session)
    service = _make_service(monkeypatch, fake_session)

    _run(
        service,
        skip_scrape=True,
        skip_graph=True,
        skip_vectors=True,
        clear_db=False,
        no_chunking=False,
    )

    fake_session.execute_write.assert_not_called()
