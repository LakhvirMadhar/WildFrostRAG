"""Proves a real Neo4j instance is reachable from this test suite (T4.0).

Starts a disposable Neo4j container via testcontainers-python - works
identically on a developer's own machine (if Docker is running) and in CI,
using the same code path. Skipped automatically when Docker isn't available.
"""

from collections.abc import Iterator

import docker.errors
import pytest
from neo4j import Driver
from testcontainers.community.neo4j import Neo4jContainer


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
def test_neo4j_container_is_reachable(neo4j_driver: Driver) -> None:
    """A trivial query against the live Neo4j container succeeds."""
    with neo4j_driver.session() as session:
        record = session.run("MATCH (n) RETURN count(n) AS node_count").single()
        assert record is not None
        assert record["node_count"] >= 0
