"""Integration tests for DocumentRepository against a real Neo4j container (T4.3)."""

from collections.abc import Iterator

import docker.errors
import pytest
from neo4j import Driver
from testcontainers.community.neo4j import Neo4jContainer

from wildfrost_rag.neo4j_kg.document_repository import DocumentRepository


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


@pytest.fixture
def seeded_documents(neo4j_driver: Driver) -> Iterator[None]:
    """Seed two Document nodes and a fulltext index, clean up afterwards."""
    with neo4j_driver.session() as session:
        session.run(
            "CREATE (:Document {text: $text, source_file: $source_file})",
            text="Bombom is a Companion card that deals damage.",
            source_file="Bombom.html",
        )
        session.run(
            "CREATE (:Document {text: $text, source_file: $source_file})",
            text="Foxee is a Leader card with high attack.",
            source_file="Foxee.html",
        )
        session.run(
            "CREATE FULLTEXT INDEX test_document_fulltext "
            "IF NOT EXISTS FOR (n:Document) ON EACH [n.text]"
        )
        session.run("CALL db.awaitIndex('test_document_fulltext', 30)")
    yield
    with neo4j_driver.session() as session:
        session.run("DROP INDEX test_document_fulltext IF EXISTS")
        session.run("MATCH (n:Document) DETACH DELETE n")


@pytest.mark.skipif(not _docker_available(), reason="Docker is not running")
def test_load_all_documents_returns_seeded_nodes(
    neo4j_driver: Driver, seeded_documents: None
) -> None:
    """load_all_documents reads back the real seeded nodes as (text, properties) pairs."""
    repository = DocumentRepository(neo4j_driver)

    results = repository.load_all_documents("Document")

    assert len(results) == 2
    texts = {text for text, _ in results}
    assert "Bombom is a Companion card that deals damage." in texts
    assert "Foxee is a Leader card with high attack." in texts
    for _, properties in results:
        assert "embedding" not in properties
        assert "source_file" in properties


@pytest.mark.skipif(not _docker_available(), reason="Docker is not running")
def test_fulltext_search_finds_matching_document(
    neo4j_driver: Driver, seeded_documents: None
) -> None:
    """fulltext_search runs a real Lucene query against the seeded index."""
    repository = DocumentRepository(neo4j_driver)

    results = repository.fulltext_search("test_document_fulltext", "Bombom", k=5)

    assert len(results) == 1
    assert results[0]["text"] == "Bombom is a Companion card that deals damage."
    assert repository.last_cypher_query is not None
