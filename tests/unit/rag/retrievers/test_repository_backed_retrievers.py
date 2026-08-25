"""Retriever tests using fake repositories instead of a live driver (T4.3).

The point of pushing Cypher execution into DocumentRepository/CardRepository:
a retriever's search() logic (building RetrievedChunk objects, applying
metadata) can now be tested with a canned repository response, no Neo4j
connection needed at all.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from wildfrost_rag.neo4j_kg.document_repository import DocumentRepository
from wildfrost_rag.rag.retrievers.neo4j_fulltext_search import Neo4jFullTextSearch
from wildfrost_rag.rag.retrievers.neo4j_vector_search import Neo4jVectorSearch
from wildfrost_rag.utils.config import get_settings


class FakeDocumentRepository(DocumentRepository):
    """Test double for DocumentRepository - returns canned data, executes nothing.

    Subclasses the real repository (rather than a Protocol) so retriever
    constructors can keep their simple `document_repository: DocumentRepository`
    type hint - the fake overrides every query method, the inherited driver
    machinery is never exercised.
    """

    def __init__(self, canned_results: list[dict[str, Any]]) -> None:
        """Store the canned results this fake will return from any query method."""
        super().__init__(driver=MagicMock())
        self._canned_results = canned_results
        self.vector_search_calls: list[tuple[str, list[float], int]] = []
        self.fulltext_search_calls: list[tuple[str, str, int]] = []

    def vector_search(
        self, index_name: str, query_embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Record the call and return the canned results, no real query executed."""
        self.vector_search_calls.append((index_name, query_embedding, k))
        self.last_cypher_query = "FAKE VECTOR QUERY"
        return self._canned_results

    def fulltext_search(self, index_name: str, query_text: str, k: int) -> list[dict[str, Any]]:
        """Record the call and return the canned results, no real query executed."""
        self.fulltext_search_calls.append((index_name, query_text, k))
        self.last_cypher_query = "FAKE FULLTEXT QUERY"
        return self._canned_results


@pytest.fixture(autouse=True)
def _fake_neo4j_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub NEO4J_* env vars for these tests.

    BaseNeo4jRetriever.__init__ reads settings.neo4j.uri for log-only port
    extraction, so constructing any retriever needs *some* value there even
    though these tests never make a real connection.
    """
    monkeypatch.setenv("NEO4J_URI", "bolt://fake-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "fake")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_vector_search_builds_retrieved_chunks_from_fake_repository() -> None:
    """Neo4jVectorSearch.search() correctly builds RetrievedChunk from canned repo output."""
    canned = [{"text": "Bombom deals damage.", "source_url": "https://wiki/Bombom", "score": 0.95}]
    fake_repository = FakeDocumentRepository(canned)
    driver = MagicMock()
    retriever = Neo4jVectorSearch(
        driver,
        embed_fn=lambda _query: [0.1, 0.2, 0.3],
        document_repository=fake_repository,
        index_name="test_index",
    )

    results = retriever.search("what does bombom do", k=3)

    assert len(results) == 1
    assert results[0].score == 0.95
    assert results[0].search_type == "vector"
    assert fake_repository.vector_search_calls == [("test_index", [0.1, 0.2, 0.3], 3)]
    # No real driver call was ever made - the repository is the only thing that ran.
    driver.session.assert_not_called()


def test_fulltext_search_builds_retrieved_chunks_from_fake_repository() -> None:
    """Neo4jFullTextSearch.search() correctly builds RetrievedChunk from canned repo output."""
    canned = [{"text": "Foxee has high attack.", "source_url": "https://wiki/Foxee", "score": 2.1}]
    fake_repository = FakeDocumentRepository(canned)
    driver = MagicMock()
    retriever = Neo4jFullTextSearch(driver, document_repository=fake_repository, index_name="idx")

    results = retriever.search("foxee attack", k=5)

    assert len(results) == 1
    assert results[0].score == 2.1
    assert results[0].search_type == "fulltext"
    assert fake_repository.fulltext_search_calls == [("idx", "foxee attack", 5)]
    driver.session.assert_not_called()
