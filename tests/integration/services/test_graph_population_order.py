"""Integration test: graph population creates cross-entity links on the FIRST run.

GraphPopulationService.populate uses MERGE throughout, so a step that links
to a node created by a *later* step silently matches nothing on a fresh
database and only "works" on a re-run. This runs populate once against an
empty, disposable Neo4j to catch that.
"""

from collections.abc import Iterator

import docker.errors
import pytest
from neo4j import Driver
from testcontainers.community.neo4j import Neo4jContainer

from wildfrost_rag.data_processing.bells import BellCategory, BellInfo
from wildfrost_rag.services.ingestion.graph_population_service import GraphPopulationService
from wildfrost_rag.services.ingestion.pipeline_data import PipelineData


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
def test_bling_bell_is_linked_to_bling_on_a_fresh_database(neo4j_driver: Driver) -> None:
    """Blingsack Bell gets its AFFECTS_BLING link even though the graph started empty."""
    data = PipelineData(
        bells=[
            BellInfo(
                name="Blingsack Bell",
                category=BellCategory.SUN,
                description="Gain bling after each battle.",
            )
        ],
        page_urls={"Bling.html": "https://wildfrostwiki.com/Bling"},
    )

    GraphPopulationService(neo4j_driver).populate(data)

    with neo4j_driver.session() as session:
        record = session.run(
            "MATCH (:Bell {name: 'Blingsack Bell'})-[r:AFFECTS_BLING]->(:Bling) "
            "RETURN count(r) AS links"
        ).single()
    assert record is not None
    assert record["links"] == 1
