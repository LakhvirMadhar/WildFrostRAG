"""Unit tests for the retriever types not covered by test_repository_backed_retrievers.py.

Covers (T5.4): BM25Retriever, Text2CypherRetriever, GraphRagRetriever,
VectorThenCypherRetriever, FulltextThenCypherRetriever, and construction/wiring
for the three non-Text2Cypher hybrid retrievers. RRF fusion math and the
Text2Cypher-vs-vector fallback logic in hybrid_retrievers.py are covered by
T5.2/T5.3's own test files, not duplicated here.
"""

import asyncio
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from wildfrost_rag.core.exceptions import CypherExecutionError
from wildfrost_rag.repositories.card_repository import CardRepository
from wildfrost_rag.repositories.document_repository import DocumentRepository
from wildfrost_rag.prompts.prompt_utils import VersionedPrompt
from wildfrost_rag.services.retrieval.bm25_retriever import BM25Retriever
from wildfrost_rag.services.retrieval.fulltext_then_cypher_retriever import (
    FulltextThenCypherRetriever,
)
from wildfrost_rag.services.retrieval.graph_rag_retriever import GraphRagRetriever
from wildfrost_rag.services.retrieval.hybrid_retrievers import (
    BM25FulltextVectorHybridRetriever,
    BM25VectorHybridRetriever,
    FulltextVectorHybridRetriever,
)
from wildfrost_rag.services.retrieval.text2cypher_retriever import Text2CypherRetriever
from wildfrost_rag.services.retrieval.vector_then_cypher_retriever import (
    VectorThenCypherRetriever,
)
from wildfrost_rag.core.config import get_settings


@pytest.fixture(autouse=True)
def _fake_neo4j_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub NEO4J_* env vars - BaseNeo4jRetriever.__init__ reads settings.neo4j.uri."""
    monkeypatch.setenv("NEO4J_URI", "bolt://fake-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "fake")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeDocumentRepository(DocumentRepository):
    """Test double for DocumentRepository used by BM25Retriever's bulk-load path."""

    def __init__(self, canned_documents: list[tuple[str, dict[str, Any]]]) -> None:
        """Store the (text, properties) pairs load_all_documents() should return."""
        super().__init__(driver=MagicMock())
        self._canned_documents = canned_documents
        self.load_all_documents_calls: list[str] = []

    def load_all_documents(self, index_name: str) -> list[tuple[str, dict[str, Any]]]:
        """Record the call and return the canned documents, no real query executed."""
        self.load_all_documents_calls.append(index_name)
        return self._canned_documents


class FakeCardRepository(CardRepository):
    """Test double for CardRepository - returns canned enriched rows, executes nothing."""

    def __init__(self, canned_results: list[dict[str, Any]]) -> None:
        """Store the canned results this fake will return from any query method."""
        super().__init__(driver=MagicMock())
        self._canned_results = canned_results
        self.vector_calls: list[tuple[str, list[float], int]] = []
        self.fulltext_calls: list[tuple[str, str, int]] = []

    def vector_search_with_enrichment(
        self, index_name: str, query_embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Record the call and return the canned results, no real query executed."""
        self.vector_calls.append((index_name, query_embedding, k))
        self.last_cypher_query = "FAKE ENRICHED VECTOR QUERY"
        return self._canned_results

    def fulltext_search_with_enrichment(
        self, index_name: str, query_text: str, k: int
    ) -> list[dict[str, Any]]:
        """Record the call and return the canned results, no real query executed."""
        self.fulltext_calls.append((index_name, query_text, k))
        self.last_cypher_query = "FAKE ENRICHED FULLTEXT QUERY"
        return self._canned_results


def test_bm25_search_builds_retrieved_chunks_from_fake_repository() -> None:
    """BM25Retriever.search() ranks canned documents and builds RetrievedChunk objects."""
    canned_documents = [
        ("Bombom deals heavy damage.", {"source_url": "https://wiki/Bombom"}),
        ("Foxee is a fast leader.", {"source_url": "https://wiki/Foxee"}),
    ]
    fake_repository = FakeDocumentRepository(canned_documents)
    driver = MagicMock()

    retriever = BM25Retriever(driver, fake_repository)
    results = retriever.search("damage", k=1)

    assert len(results) == 1
    assert results[0].search_type == "bm25"
    assert "Bombom" in results[0].retrieved_text or "damage" in results[0].retrieved_text.lower()
    assert fake_repository.load_all_documents_calls == [retriever.index_name]
    driver.session.assert_not_called()


def test_bm25_search_uses_cache_on_second_instance() -> None:
    """BM25Retriever's class-level cache avoids a second document load for the same index."""
    canned_documents = [("Bombom deals damage.", {"source_url": "https://wiki/Bombom"})]
    driver = MagicMock()

    first_repository = FakeDocumentRepository(canned_documents)
    first_retriever = BM25Retriever(driver, first_repository)
    first_retriever.search("damage", k=1)

    second_repository = FakeDocumentRepository(canned_documents)
    second_retriever = BM25Retriever(driver, second_repository)
    second_retriever.search("damage", k=1)

    assert second_repository.load_all_documents_calls == []


def test_graph_rag_retriever_returns_empty_placeholder() -> None:
    """GraphRagRetriever is an unimplemented placeholder - it returns no chunks."""
    driver = MagicMock()
    retriever = GraphRagRetriever(driver)

    results = retriever.search("what does bombom do", k=5)

    assert results == []
    driver.session.assert_not_called()


def test_vector_then_cypher_search_builds_retrieved_chunks_from_fake_repository() -> None:
    """VectorThenCypherRetriever.search() builds RetrievedChunk from canned enriched rows."""
    canned = [
        {
            "text": "Bombom deals damage.",
            "source_url": "https://wiki/Bombom",
            "score": 0.9,
            "card_name": "Bombom",
        }
    ]
    fake_repository = FakeCardRepository(canned)
    driver = MagicMock()
    retriever = VectorThenCypherRetriever(
        driver,
        embed_fn=lambda _query: [0.1, 0.2, 0.3],
        card_repository=fake_repository,
        index_name="test_index",
    )

    results = retriever.search("what does bombom do", k=3)

    assert len(results) == 1
    assert results[0].score == 0.9
    assert results[0].search_type == "vector_then_cypher"
    assert fake_repository.vector_calls == [("test_index", [0.1, 0.2, 0.3], 3)]
    assert retriever.last_cypher_query == "FAKE ENRICHED VECTOR QUERY"
    driver.session.assert_not_called()


def test_fulltext_then_cypher_search_builds_retrieved_chunks_from_fake_repository() -> None:
    """FulltextThenCypherRetriever.search() builds RetrievedChunk from canned enriched rows."""
    canned = [
        {
            "text": "Foxee has high attack.",
            "source_url": "https://wiki/Foxee",
            "score": 2.5,
            "card_name": "Foxee",
        }
    ]
    fake_repository = FakeCardRepository(canned)
    driver = MagicMock()
    retriever = FulltextThenCypherRetriever(
        driver, card_repository=fake_repository, index_name="idx"
    )

    results = retriever.search("foxee attack", k=5)

    assert len(results) == 1
    assert results[0].score == 2.5
    assert results[0].search_type == "fulltext_then_cypher"
    assert fake_repository.fulltext_calls == [("idx", "foxee attack", 5)]
    driver.session.assert_not_called()


def test_fulltext_then_cypher_preprocesses_query_when_stopwords_enabled() -> None:
    """remove_stopwords=True strips stop words from the query before it reaches the repository."""
    fake_repository = FakeCardRepository([])
    driver = MagicMock()
    retriever = FulltextThenCypherRetriever(
        driver, card_repository=fake_repository, index_name="idx", remove_stopwords=True
    )

    retriever.search("what is the attack of Foxee", k=5)

    assert len(fake_repository.fulltext_calls) == 1
    _, query_text, _ = fake_repository.fulltext_calls[0]
    assert "the" not in query_text.split()
    assert "attack" in query_text
    assert "foxee" in query_text


def _make_versioned_prompt() -> VersionedPrompt:
    return VersionedPrompt(
        prompt_version_name="TEST_TEXT2CYPHER_PROMPT",
        prompt_tuple=("Schema: {schema}\nQuery: {query}", "schema", "query"),
    )


@patch("wildfrost_rag.services.retrieval.text2cypher_retriever.call_openai_api")
def test_text2cypher_search_builds_retrieved_chunks_from_mocked_llm_and_driver(
    mock_call_openai_api: MagicMock,
) -> None:
    """Text2CypherRetriever.search() builds RetrievedChunk from a mocked LLM + driver session.

    Text2Cypher has no repository seam (its Cypher is LLM-generated per call), so the
    fake-repository pattern doesn't apply here - the driver/session and the LLM call
    are mocked directly instead. Run via asyncio.run() since no pytest-asyncio plugin
    is configured for this project.
    """

    async def _fake_call_openai_api(*_args: object, **_kwargs: object) -> str:
        return "MATCH (c:Card {name: 'Bombom'}) RETURN c.name AS name"

    mock_call_openai_api.side_effect = _fake_call_openai_api

    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    schema_record = MagicMock()
    schema_record.__getitem__ = lambda _self, key: {
        "nodeType": ":`Card`",
        "properties": [{"propertyName": "name", "propertyTypes": ["String"], "mandatory": True}],
    }[key]
    viz_record = MagicMock()
    viz_record.get.return_value = []

    result_record = MagicMock()
    result_record.keys.return_value = ["name"]
    result_record.__getitem__ = lambda _self, key: "Bombom" if key == "name" else None
    result_record.__iter__ = lambda self: iter(["name"])

    def _execute_read(read_tx: Callable[[MagicMock], object]) -> object:
        tx = MagicMock()

        def _run(query: str, *_args: object, **_kwargs: object) -> object:
            if "nodeTypeProperties" in query:
                return [schema_record]
            if "visualization" in query:
                result = MagicMock()
                result.single.return_value = viz_record
                return result
            return [result_record]

        tx.run.side_effect = _run
        return read_tx(tx)

    session.execute_read.side_effect = _execute_read

    retriever = Text2CypherRetriever(driver, _make_versioned_prompt())
    results = asyncio.run(retriever.search("what does bombom do", k=5))

    assert len(results) == 1
    assert results[0].search_type == "text2cypher_llm"
    assert retriever.last_cypher_query is not None
    assert "LIMIT" in retriever.last_cypher_query


def test_text2cypher_add_limit_clause_appends_when_missing() -> None:
    """_add_limit_clause appends LIMIT only when the generated query lacks one."""
    driver = MagicMock()
    retriever = Text2CypherRetriever(driver, _make_versioned_prompt())

    with_limit = retriever._add_limit_clause("MATCH (c) RETURN c LIMIT 10", k=5)
    without_limit = retriever._add_limit_clause("MATCH (c) RETURN c", k=5)

    assert with_limit == "MATCH (c) RETURN c LIMIT 10"
    assert without_limit == "MATCH (c) RETURN c LIMIT 5"


def test_text2cypher_search_wraps_execution_failure_in_domain_exception() -> None:
    """A Cypher execution failure surfaces as CypherExecutionError, not a raw driver error."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    def _execute_read(read_tx: Callable[[MagicMock], object]) -> object:
        tx = MagicMock()
        tx.run.side_effect = RuntimeError("syntax error")
        return read_tx(tx)

    session.execute_read.side_effect = _execute_read

    retriever = Text2CypherRetriever(driver, _make_versioned_prompt())

    with pytest.raises(CypherExecutionError):
        retriever._execute_cypher_query(session, "MATCH (c) RETURN c", k=5)


def test_bm25_vector_hybrid_wires_up_component_retrievers() -> None:
    """BM25VectorHybridRetriever constructs and names its two component retrievers."""
    driver = MagicMock()
    retriever = BM25VectorHybridRetriever(driver, embed_fn=lambda _q: [0.1])

    assert retriever.retriever_names == ["bm25", "vector"]
    assert len(retriever.retrievers) == 2
    assert isinstance(retriever.retrievers[0], BM25Retriever)


def test_fulltext_vector_hybrid_wires_up_component_retrievers() -> None:
    """FulltextVectorHybridRetriever constructs and names its two component retrievers."""
    driver = MagicMock()
    retriever = FulltextVectorHybridRetriever(driver, embed_fn=lambda _q: [0.1])

    assert retriever.retriever_names == ["fulltext", "vector"]
    assert len(retriever.retrievers) == 2


def test_bm25_fulltext_vector_hybrid_wires_up_component_retrievers() -> None:
    """BM25FulltextVectorHybridRetriever constructs and names all three component retrievers."""
    driver = MagicMock()
    retriever = BM25FulltextVectorHybridRetriever(driver, embed_fn=lambda _q: [0.1])

    assert retriever.retriever_names == ["bm25", "fulltext", "vector"]
    assert len(retriever.retrievers) == 3
