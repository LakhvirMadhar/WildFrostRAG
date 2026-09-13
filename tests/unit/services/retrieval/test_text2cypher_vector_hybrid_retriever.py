"""Unit tests for Text2CypherVectorHybridRetriever.search().

search() is decomposed into _search_text2cypher (async, catches
WildFrostRAGError) and _build_vector_fallback (sync). These tests inject fake
component retrievers directly onto the instance so no live Neo4j/LLM call is
ever made, and exercise both the fallback path and the RRF-fusion path.
"""

import asyncio
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from mlflow.entities.model_registry.prompt_version import PromptVersion

from wildfrost_rag.core.exceptions import CypherExecutionError
from wildfrost_rag.domain.retrieval import RetrievedChunk
from wildfrost_rag.services.retrieval.hybrid_retrievers import Text2CypherVectorHybridRetriever
from wildfrost_rag.services.retrieval.neo4j_vector_search import Neo4jVectorSearch
from wildfrost_rag.services.retrieval.text2cypher_retriever import Text2CypherRetriever
from wildfrost_rag.core.config import get_settings


class FakeText2CypherRetriever(Text2CypherRetriever):
    """Test double subclassing Text2CypherRetriever - overrides search(), no real LLM/driver call.

    Subclassing (rather than duck-typing) lets retriever.text2cypher hold this
    fake without a mypy assignment error, since the attribute is typed as the
    real Text2CypherRetriever.
    """

    def __init__(
        self, results: list[RetrievedChunk] | None = None, error: Exception | None = None
    ) -> None:
        """Store either the canned results to return, or the error to raise."""
        super().__init__(
            driver=MagicMock(),
            text2cypher_prompt=MagicMock(spec=PromptVersion, uri="prompts:/fake/1"),
        )
        self._results = results or []
        self._error = error

    async def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Raise the configured error, or return the canned results."""
        if self._error is not None:
            raise self._error
        return self._results


class FakeVectorRetriever(Neo4jVectorSearch):
    """Test double subclassing Neo4jVectorSearch - overrides search(), no real driver call."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        """Store the canned results this fake will return."""
        super().__init__(
            driver=MagicMock(), embed_fn=lambda _query: [0.0], document_repository=MagicMock()
        )
        self._results = results

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the canned results, no real query executed."""
        return self._results


@pytest.fixture(autouse=True)
def _fake_neo4j_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub NEO4J_* env vars, needed by BaseNeo4jRetriever.__init__ during construction."""
    monkeypatch.setenv("NEO4J_URI", "bolt://fake-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "fake")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _build_retriever() -> Text2CypherVectorHybridRetriever:
    prompt = MagicMock(spec=PromptVersion, uri="prompts:/test_prompt/1")
    return Text2CypherVectorHybridRetriever(
        driver=MagicMock(),
        embed_fn=lambda _query: [0.1, 0.2, 0.3],
        text2cypher_prompt=prompt,
    )


def test_falls_back_to_vector_only_when_text2cypher_raises_domain_error() -> None:
    """A WildFrostRAGError from Text2Cypher yields vector-only fallback results."""
    retriever = _build_retriever()
    vector_chunk = RetrievedChunk(
        score=0.8, search_type="vector", retrieved_text="Bombom deals damage.", source_url=None
    )
    retriever.text2cypher = FakeText2CypherRetriever(
        error=CypherExecutionError(cypher_query="MATCH (n) RETURN n", reason="syntax error")
    )
    retriever.vector = FakeVectorRetriever([vector_chunk])

    results = asyncio.run(retriever.search("what does bombom do", k=5))

    assert len(results) == 1
    assert results[0].search_type == "text2cypher_vector_fallback"
    assert results[0].retrieved_text == "Bombom deals damage."
    assert retriever.text2cypher_success is False
    assert retriever.last_individual_results == {"text2cypher": [], "vector": [vector_chunk]}


def test_fuses_results_with_rrf_when_text2cypher_succeeds() -> None:
    """When both retrievers succeed, results are fused via RRF, not vector-only fallback."""
    retriever = _build_retriever()
    text2cypher_chunk = RetrievedChunk(
        score=1.0, search_type="text2cypher", retrieved_text="Bombom", source_url=None
    )
    vector_chunk = RetrievedChunk(
        score=0.8, search_type="vector", retrieved_text="Bombom deals damage.", source_url=None
    )
    retriever.text2cypher = FakeText2CypherRetriever(results=[text2cypher_chunk])
    retriever.vector = FakeVectorRetriever([vector_chunk])

    results = asyncio.run(retriever.search("what does bombom do", k=5))

    assert len(results) > 0
    assert all(chunk.search_type == "hybrid_rrf" for chunk in results)
    assert retriever.text2cypher_success is True
    assert retriever.last_individual_results == {
        "text2cypher": [text2cypher_chunk],
        "vector": [vector_chunk],
    }
