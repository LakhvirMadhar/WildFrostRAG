"""Materializes card_embeddings -> vector_index with fakes at the I/O boundary.

EmbeddingService is patched entirely (its own unit tests cover its real behavior),
and Neo4jResource.get_driver is patched to yield a MagicMock(spec=Driver) instead of
a live connection - this proves the Dagster wiring (config passed through, dependency
order, retry_policy attached), not real embedding generation or Neo4j writes.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from unittest.mock import MagicMock, patch

import pytest
from dagster import materialize
from neo4j import Driver

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.embedder_type import EmbedderType
from wildfrost_rag.defs.embeddings.assets import EmbeddingConfig, card_embeddings, vector_index
from wildfrost_rag.defs.resources import Neo4jResource


@pytest.fixture(autouse=True)
def _fake_neo4j_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub NEO4J_* env vars so get_settings() validates without a real Neo4j instance."""
    monkeypatch.setenv("NEO4J_URI", "bolt://fake-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "fake")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeNeo4jResource(Neo4jResource):
    """A Neo4jResource whose driver is a MagicMock(spec=Driver), never a live connection."""

    def get_driver(self) -> AbstractContextManager[Driver]:
        @contextmanager
        def _fake_driver_context() -> Iterator[Driver]:
            yield MagicMock(spec=Driver)

        return _fake_driver_context()


def test_asset_chain_materializes_in_dependency_order() -> None:
    """card_embeddings runs before vector_index; both succeed with fakes."""
    with (
        patch(
            "wildfrost_rag.defs.embeddings.assets.EmbeddingService.add_embeddings",
            return_value=3,
        ),
        patch(
            "wildfrost_rag.defs.embeddings.assets.EmbeddingService.create_vector_index",
            return_value=None,
        ),
    ):
        result = materialize(
            [card_embeddings, vector_index],
            resources={"neo4j": _FakeNeo4jResource()},
        )

    assert result.success

    materialized_order = []
    for event in result.get_asset_materialization_events():
        assert event.asset_key is not None
        materialized_order.append(event.asset_key.to_user_string())

    assert materialized_order == ["card_embeddings", "vector_index"]


def test_card_embeddings_passes_configured_embedder_to_the_service() -> None:
    """The Config's embedder field reaches EmbeddingService.add_embeddings unchanged."""
    with patch(
        "wildfrost_rag.defs.embeddings.assets.EmbeddingService.add_embeddings",
        return_value=0,
    ) as fake_add_embeddings:
        result = materialize(
            [card_embeddings],
            resources={"neo4j": _FakeNeo4jResource()},
            run_config={
                "ops": {"card_embeddings": {"config": {"embedder": "OPENAI"}}},
            },
        )

    assert result.success
    called_embedder = fake_add_embeddings.call_args[0][1]
    assert called_embedder is EmbedderType.OPENAI


def test_embedding_config_defaults_to_hf() -> None:
    """No config supplied -> defaults to EmbedderType.HF, matching the CLI's own default."""
    assert EmbeddingConfig().embedder is EmbedderType.HF
