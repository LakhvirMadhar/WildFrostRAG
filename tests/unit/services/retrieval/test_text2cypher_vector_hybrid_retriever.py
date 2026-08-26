"""Unit tests for Text2CypherVectorHybridRetriever.search() (T5.3).

search() is decomposed into _search_text2cypher (async, catches
WildFrostRAGError) and _build_vector_fallback (sync). These tests inject fake
component retrievers directly onto the instance so no live Neo4j/LLM call is
ever made, and exercise both the fallback path and the RRF-fusion path.
"""

import asyncio
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from wildfrost_rag.core.exceptions import CypherExecutionError
from wildfrost_rag.domain.retrieval import RetrievedChunk
from wildfrost_rag.prompts.prompt_utils import VersionedPrompt
from wildfrost_rag.services.retrieval.hybrid_retrievers import Text2CypherVectorHybridRetriever
from wildfrost_rag.core.config import get_settings


class FakeText2CypherRetriever:
    """Test double standing in for Text2CypherRetriever.search()."""

    def __init__(
        self, results: list[RetrievedChunk] | None = None, error: Exception | None = None
    ) -> None:
        """Store either the canned results to return, or the error to raise."""
        self._results = results or []
        self._error = error

    async def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Raise the configured error, or return the canned results."""
        if self._error is not None:
            raise self._error
        return self._results


class FakeVectorRetriever:
    """Test double standing in for Neo4jVectorSearch.search()."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        """Store the canned results this fake will return."""
        self._results = results

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
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
    prompt = VersionedPrompt(prompt_version_name="TEST_PROMPT", prompt_tuple=("test",))
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
