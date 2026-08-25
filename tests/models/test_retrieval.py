"""Tests for typed retrieval schemas."""

import json
from typing import Any

import pytest

from wildfrost_rag.models.retrieval import RetrievedChunk, QueryResult, CypherExecution


# ---------------------------------------------------------------------------
# Fixtures: representative raw dicts from each retriever type
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_bm25_chunk() -> dict[str, Any]:
    """Raw dict as returned by BM25Retriever."""
    return {
        "text": "Bombom is a Companion card...",
        "title": "Bombom",
        "source_file": "data/structured_outputs/companions/Bombom.html",
        "source_url": "https://wildfrostwiki.com/Bombom",
        "score": 12.345,
        "search_type": "bm25",
        "rag_context": "Title: Bombom\nText: Bombom is a Companion card...",
    }


@pytest.fixture
def raw_text2cypher_chunk() -> dict[str, Any]:
    """Raw dict as returned by Text2CypherRetriever (arbitrary RETURN aliases)."""
    return {
        "card_name": "Foxee",
        "attack": 2,
        "health": 4,
        "counter": 3,
        "score": 1.0,
        "search_type": "text2cypher_llm",
        "rag_context": "Card Name: Foxee\nAttack: 2\nHealth: 4\nCounter: 3",
        "generated_cypher": "MATCH (c:Card) WHERE c.card_name = 'Foxee' RETURN c.card_name AS card_name",
        "result_index": 0,
    }


@pytest.fixture
def raw_hybrid_chunk() -> dict[str, Any]:
    """Raw dict as returned by a HybridRetriever (RRF scores)."""
    return {
        "text": "Mimik copies abilities...",
        "source_file": "data/structured_outputs/companions/Mimik.html",
        "doc_source_url": "https://wildfrostwiki.com/Mimik",
        "score": 0.032,
        "search_type": "bm25_vector_hybrid",
        "rag_context": "Text: Mimik copies abilities...",
        "rrf_score": 0.032,
        "retriever_scores": {"bm25": 0.016, "vector": 0.016},
    }


# ---------------------------------------------------------------------------
# RetrievedChunk tests
# ---------------------------------------------------------------------------


class TestRetrievedChunk:
    """Tests for the RetrievedChunk dataclass."""

    def test_from_raw_bm25(self, raw_bm25_chunk: dict[str, Any]) -> None:
        """Verify BM25 raw dict is correctly parsed into a RetrievedChunk."""
        chunk = RetrievedChunk.from_raw_retriever_dict(raw_bm25_chunk)

        assert chunk.score == 12.345
        assert chunk.search_type == "bm25"
        assert chunk.retrieved_text == "Title: Bombom\nText: Bombom is a Companion card..."
        assert chunk.source_url == "https://wildfrostwiki.com/Bombom"
        # cypher_result should contain the raw Cypher row data
        assert chunk.cypher_result["text"] == "Bombom is a Companion card..."
        assert chunk.cypher_result["title"] == "Bombom"
        assert (
            chunk.cypher_result["source_file"] == "data/structured_outputs/companions/Bombom.html"
        )
        # Consumed keys must NOT be in cypher_result
        assert "score" not in chunk.cypher_result
        assert "search_type" not in chunk.cypher_result
        assert "rag_context" not in chunk.cypher_result
        assert "source_url" not in chunk.cypher_result

    def test_from_raw_text2cypher(self, raw_text2cypher_chunk: dict[str, Any]) -> None:
        """Verify text2cypher raw dict is correctly parsed."""
        chunk = RetrievedChunk.from_raw_retriever_dict(raw_text2cypher_chunk)

        assert chunk.score == 1.0
        assert chunk.search_type == "text2cypher_llm"
        assert "Foxee" in chunk.retrieved_text
        assert chunk.source_url is None  # text2cypher has no source_url
        # Arbitrary LLM-chosen field names end up in cypher_result
        assert chunk.cypher_result["card_name"] == "Foxee"
        assert chunk.cypher_result["attack"] == 2
        # Metadata keys are excluded
        assert "generated_cypher" not in chunk.cypher_result
        assert "result_index" not in chunk.cypher_result

    def test_from_raw_hybrid_doc_source_url_fallback(
        self, raw_hybrid_chunk: dict[str, Any]
    ) -> None:
        """source_url should consolidate doc_source_url when source_url is absent."""
        chunk = RetrievedChunk.from_raw_retriever_dict(raw_hybrid_chunk)

        assert chunk.source_url == "https://wildfrostwiki.com/Mimik"
        # Hybrid-specific fields in cypher_result
        assert chunk.cypher_result["rrf_score"] == 0.032
        assert "bm25" in chunk.cypher_result["retriever_scores"]

    def test_source_url_prefers_source_url_over_doc_source_url(self) -> None:
        """When both exist, source_url takes precedence."""
        raw = {
            "score": 1.0,
            "search_type": "vector",
            "rag_context": "some text",
            "source_url": "https://wildfrostwiki.com/Correct",
            "doc_source_url": "https://wildfrostwiki.com/Wrong",
        }
        chunk = RetrievedChunk.from_raw_retriever_dict(raw)
        assert chunk.source_url == "https://wildfrostwiki.com/Correct"

    def test_to_dict_json_serializable(self, raw_bm25_chunk: dict[str, Any]) -> None:
        """Verify to_dict output is JSON-serializable."""
        chunk = RetrievedChunk.from_raw_retriever_dict(raw_bm25_chunk)
        d = chunk.to_dict()

        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        # Known fields at top level
        assert d["score"] == 12.345
        assert d["search_type"] == "bm25"
        assert d["retrieved_text"] == "Title: Bombom\nText: Bombom is a Companion card..."
        assert d["source_url"] == "https://wildfrostwiki.com/Bombom"
        # cypher_result is nested (not flat)
        assert d["cypher_result"]["text"] == "Bombom is a Companion card..."
        assert "text" not in d  # NOT at top level

    def test_roundtrip_to_dict_from_dict(self, raw_bm25_chunk: dict[str, Any]) -> None:
        """Verify to_dict -> from_dict roundtrip preserves all fields."""
        original = RetrievedChunk.from_raw_retriever_dict(raw_bm25_chunk)
        restored = RetrievedChunk.from_dict(original.to_dict())

        assert restored.score == original.score
        assert restored.search_type == original.search_type
        assert restored.retrieved_text == original.retrieved_text
        assert restored.source_url == original.source_url
        assert restored.cypher_result == original.cypher_result

    def test_roundtrip_text2cypher(self, raw_text2cypher_chunk: dict[str, Any]) -> None:
        """Verify roundtrip for text2cypher chunks."""
        original = RetrievedChunk.from_raw_retriever_dict(raw_text2cypher_chunk)
        restored = RetrievedChunk.from_dict(original.to_dict())

        assert restored.score == original.score
        assert restored.cypher_result["card_name"] == "Foxee"
        assert restored.source_url is None

    def test_empty_chunk(self) -> None:
        """Minimal chunk with defaults."""
        chunk = RetrievedChunk(score=0.0, search_type="", retrieved_text="", source_url=None)
        d = chunk.to_dict()
        restored = RetrievedChunk.from_dict(d)
        assert restored.score == 0.0
        assert restored.cypher_result == {}


# ---------------------------------------------------------------------------
# CypherExecution tests
# ---------------------------------------------------------------------------


class TestCypherExecution:
    """Tests for the CypherExecution dataclass."""

    def test_success(self) -> None:
        """Verify successful cypher execution serialization."""
        ce = CypherExecution(
            cypher_query="MATCH (c:Card) WHERE c.attack = 5 RETURN c",
            cypher_execution_status="success",
            cypher_error_message=None,
        )
        d = ce.to_dict()
        assert d["cypher_query"] == "MATCH (c:Card) WHERE c.attack = 5 RETURN c"
        assert d["cypher_execution_status"] == "success"
        assert d["cypher_error_message"] is None

    def test_failure(self) -> None:
        """Verify failed cypher execution serialization."""
        ce = CypherExecution(
            cypher_query="INVALID CYPHER",
            cypher_execution_status="failed",
            cypher_error_message="SyntaxError: unexpected token",
        )
        d = ce.to_dict()
        assert d["cypher_execution_status"] == "failed"
        assert d["cypher_error_message"] == "SyntaxError: unexpected token"

    def test_null_cypher(self) -> None:
        """BM25 and hybrid retrievers have no per-query Cypher."""
        ce = CypherExecution(
            cypher_query=None,
            cypher_execution_status="success",
            cypher_error_message=None,
        )
        d = ce.to_dict()
        assert d["cypher_query"] is None

        restored = CypherExecution.from_dict(d)
        assert restored.cypher_query is None

    def test_roundtrip(self) -> None:
        """Verify to_dict -> from_dict roundtrip for CypherExecution."""
        original = CypherExecution(
            cypher_query="MATCH (c:Card {card_name: 'Foxee'}) RETURN c",
            cypher_execution_status="success",
            cypher_error_message=None,
        )
        restored = CypherExecution.from_dict(original.to_dict())
        assert restored.cypher_query == original.cypher_query
        assert restored.cypher_execution_status == original.cypher_execution_status
        assert restored.cypher_error_message == original.cypher_error_message

    def test_json_serializable(self) -> None:
        """Verify CypherExecution to_dict is JSON-serializable."""
        ce = CypherExecution(
            cypher_query="RETURN 1",
            cypher_execution_status="success",
            cypher_error_message=None,
        )
        serialized = json.dumps(ce.to_dict())
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# QueryResult tests
# ---------------------------------------------------------------------------


class TestQueryResult:
    """Tests for the QueryResult dataclass."""

    def test_to_dict_includes_cypher_execution(self, raw_bm25_chunk: dict[str, Any]) -> None:
        """Verify QueryResult to_dict includes cypher_execution field."""
        chunk = RetrievedChunk.from_raw_retriever_dict(raw_bm25_chunk)
        qr = QueryResult(
            query_id=1,
            query="What does Bombom do?",
            retrieved_chunks=[chunk],
            relevance_annotations=[{"chunk_index": 0, "is_relevant": True}],
        )

        d = qr.to_dict()
        assert d["query_id"] == 1
        assert d["query"] == "What does Bombom do?"
        assert "cypher_execution" in d
        assert d["cypher_execution"]["cypher_query"] is None
        assert d["cypher_execution"]["cypher_execution_status"] == "success"
        assert len(d["retrieved_chunks"]) == 1
        assert d["retrieved_chunks"][0]["retrieved_text"] == chunk.retrieved_text
        assert d["relevance_annotations"] == [{"chunk_index": 0, "is_relevant": True}]

    def test_bm25_cypher_execution_null_query(self, raw_bm25_chunk: dict[str, Any]) -> None:
        """BM25 results have cypher_execution with null cypher_query."""
        chunk = RetrievedChunk.from_raw_retriever_dict(raw_bm25_chunk)
        ce = CypherExecution(
            cypher_query=None,
            cypher_execution_status="success",
            cypher_error_message=None,
        )
        qr = QueryResult(
            query_id=1,
            query="What does Bombom do?",
            cypher_execution=ce,
            retrieved_chunks=[chunk],
        )
        d = qr.to_dict()
        assert d["cypher_execution"]["cypher_query"] is None
        assert d["cypher_execution"]["cypher_execution_status"] == "success"

    def test_text2cypher_cypher_execution_populated(
        self, raw_text2cypher_chunk: dict[str, Any]
    ) -> None:
        """Text2Cypher results have cypher_execution with populated cypher_query."""
        chunk = RetrievedChunk.from_raw_retriever_dict(raw_text2cypher_chunk)
        ce = CypherExecution(
            cypher_query="MATCH (c:Card) WHERE c.card_name = 'Foxee' RETURN c.card_name",
            cypher_execution_status="success",
            cypher_error_message=None,
        )
        qr = QueryResult(
            query_id=1,
            query="Find Foxee",
            cypher_execution=ce,
            retrieved_chunks=[chunk],
        )
        d = qr.to_dict()
        assert (
            d["cypher_execution"]["cypher_query"]
            == "MATCH (c:Card) WHERE c.card_name = 'Foxee' RETURN c.card_name"
        )

    def test_text2cypher_failure_zero_rows(self) -> None:
        """Text2Cypher failure: cypher_execution shows failure, retrieved_chunks empty."""
        ce = CypherExecution(
            cypher_query="MATCH (c:Card)-[:BELONGS_TO]->(t:Tribe) RETURN t.name",
            cypher_execution_status="failed",
            cypher_error_message="RelationshipType BELONGS_TO does not exist",
        )
        qr = QueryResult(
            query_id=2,
            query="What tribe is Sneezle in?",
            cypher_execution=ce,
            retrieved_chunks=[],
        )
        d = qr.to_dict()
        assert d["cypher_execution"]["cypher_execution_status"] == "failed"
        assert (
            d["cypher_execution"]["cypher_error_message"]
            == "RelationshipType BELONGS_TO does not exist"
        )
        assert d["retrieved_chunks"] == []

    def test_roundtrip(
        self, raw_bm25_chunk: dict[str, Any], raw_text2cypher_chunk: dict[str, Any]
    ) -> None:
        """Verify QueryResult roundtrip with multiple chunk types."""
        chunks = [
            RetrievedChunk.from_raw_retriever_dict(raw_bm25_chunk),
            RetrievedChunk.from_raw_retriever_dict(raw_text2cypher_chunk),
        ]
        ce = CypherExecution(
            cypher_query="RETURN 1",
            cypher_execution_status="success",
            cypher_error_message=None,
        )
        original = QueryResult(
            query_id=5,
            query="Compare Bombom and Foxee",
            cypher_execution=ce,
            retrieved_chunks=chunks,
            relevance_annotations=[],
        )

        restored = QueryResult.from_dict(original.to_dict())
        assert restored.query_id == original.query_id
        assert restored.query == original.query
        assert restored.cypher_execution.cypher_query == "RETURN 1"
        assert len(restored.retrieved_chunks) == 2
        assert restored.retrieved_chunks[0].source_url == "https://wildfrostwiki.com/Bombom"
        assert restored.retrieved_chunks[1].cypher_result["card_name"] == "Foxee"

    def test_empty_chunks(self) -> None:
        """Zero-shot mode: no retrieved chunks."""
        qr = QueryResult(query_id=1, query="What is Wildfrost?")
        d = qr.to_dict()
        restored = QueryResult.from_dict(d)
        assert restored.retrieved_chunks == []
        assert restored.relevance_annotations == []
        assert restored.cypher_execution.cypher_query is None

    def test_json_serializable(self, raw_bm25_chunk: dict[str, Any]) -> None:
        """Verify QueryResult to_dict is JSON-serializable."""
        chunk = RetrievedChunk.from_raw_retriever_dict(raw_bm25_chunk)
        qr = QueryResult(query_id=1, query="test", retrieved_chunks=[chunk])
        serialized = json.dumps(qr.to_dict())
        assert isinstance(serialized, str)
