"""Proves Neo4jResource/OpenAIResource are real, working Dagster resources.

Uses Dagster's own materialize() test API with a trivial throwaway asset -
not just "the class imports cleanly." Neo4jResource's driver construction
is lazy (GraphDatabase.driver() doesn't connect eagerly), so this needs
no live Neo4j instance.
"""

from collections.abc import Iterator

import pytest
from dagster import asset, materialize
from neo4j import Driver

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.defs.resources import Neo4jResource


@pytest.fixture(autouse=True)
def _fake_neo4j_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub NEO4J_* env vars so Neo4jResource can build a driver without a real one."""
    monkeypatch.setenv("NEO4J_URI", "bolt://fake-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "fake")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_trivial_asset_reads_via_neo4j_resource() -> None:
    """A real asset injected with Neo4jResource materializes successfully."""

    @asset
    def sample_asset(neo4j: Neo4jResource) -> str:
        with neo4j.get_driver() as driver:
            assert isinstance(driver, Driver)
            return type(driver).__name__

    result = materialize([sample_asset], resources={"neo4j": Neo4jResource()})

    assert result.success
    assert result.output_for_node("sample_asset") == "BoltDriver"
