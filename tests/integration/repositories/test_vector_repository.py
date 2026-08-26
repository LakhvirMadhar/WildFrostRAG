"""Proves VectorRepository (T4.1) uses only its injected driver.

Starts a disposable Neo4j container via testcontainers-python, same pattern
as test_neo4j_container.py. Skipped automatically when Docker isn't running.
"""

from collections.abc import Iterator
from unittest.mock import patch

import docker.errors
import numpy as np
import pytest
from neo4j import Driver
from testcontainers.community.neo4j import Neo4jContainer

from wildfrost_rag.neo4j_kg.vector_store import VectorRepository


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
    except docker.errors.DockerException:
        return False
    return True


@pytest.fixture(scope="module")
def neo4j_driver() -> Iterator[Driver]:
    """Start a real, disposable Neo4j 5 container and yield a connected driver."""
    with Neo4jContainer(image="neo4j:5") as container:
        yield container.get_driver()


@pytest.mark.skipif(not _docker_available(), reason="Docker is not running")
def test_create_embedding_index_uses_only_the_injected_driver(neo4j_driver: Driver) -> None:
    """create_embedding_index never self-constructs a driver - only the injected one is used."""
    repository = VectorRepository(neo4j_driver)

    with patch("neo4j.GraphDatabase.driver") as mock_driver_constructor:
        repository.create_embedding_index(
            property_name="test_embedding", index_name="test-index", dimension=4
        )
        mock_driver_constructor.assert_not_called()

    with neo4j_driver.session() as session:
        record = session.run(
            "SHOW VECTOR INDEXES YIELD name WHERE name = $name RETURN name",
            name="test-index",
        ).single()
        assert record is not None
        assert record["name"] == "test-index"


@pytest.mark.skipif(not _docker_available(), reason="Docker is not running")
def test_get_retrieved_chunks_uses_only_the_injected_driver(neo4j_driver: Driver) -> None:
    """get_retrieved_chunks never self-constructs a driver - only the injected one is used."""
    repository = VectorRepository(neo4j_driver)
    # property must be named "embedding" - get_retrieved_chunks only excludes
    # that exact key from the returned dict, not arbitrary embedding properties
    repository.create_embedding_index(
        property_name="embedding", index_name="chunk-index", dimension=3
    )

    with neo4j_driver.session() as session:
        session.run(
            "CREATE (:Document {text: $text, embedding: $embedding})",
            text="a test document chunk",
            embedding=[1.0, 0.0, 0.0],
        )

    class _FakeEmbeddingModel:
        def encode(self, _query: str) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0])

    with patch("neo4j.GraphDatabase.driver") as mock_driver_constructor:
        chunks = repository.get_retrieved_chunks(
            query="anything",
            embedding_model=_FakeEmbeddingModel(),  # type: ignore[arg-type]
            index_name="chunk-index",
            k=1,
        )
        mock_driver_constructor.assert_not_called()

    assert len(chunks) == 1
    assert chunks[0]["text"] == "a test document chunk"
    assert "embedding" not in chunks[0]
