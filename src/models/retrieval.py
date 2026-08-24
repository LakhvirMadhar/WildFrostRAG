"""
Typed schemas for retriever outputs.

These dataclasses define the save/load boundary — what gets written to results.json
and what downstream consumers (generation, auto-annotation, metrics) read back.

Retrievers keep returning raw List[Dict[str, Any]] internally. Conversion to typed
objects happens in evaluate_retrievers.py before saving.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """A single retrieval result from any retriever."""

    score: float
    search_type: str
    retrieved_text: str
    source_url: str | None
    cypher_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage.

        cypher_result is nested under its own key (not flattened).
        """
        return {
            "score": self.score,
            "search_type": self.search_type,
            "retrieved_text": self.retrieved_text,
            "source_url": self.source_url,
            "cypher_result": self.cypher_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievedChunk":
        """Deserialize from a dict (as saved by to_dict)."""
        return cls(
            score=data.get("score", 0.0),
            search_type=data.get("search_type", ""),
            retrieved_text=data.get("retrieved_text", ""),
            source_url=data.get("source_url"),
            cypher_result=data.get("cypher_result", {}),
        )

    @classmethod
    def from_raw_retriever_dict(cls, raw: dict[str, Any]) -> "RetrievedChunk":
        """Convert a raw retriever output dict to a typed RetrievedChunk.

        This handles the rename: rag_context -> retrieved_text,
        and consolidates source_url / doc_source_url.
        Remaining fields become cypher_result (the actual Cypher row data).
        """
        # Keys consumed by typed fields or filtered out (metadata-only)
        consumed_keys = {
            "score",
            "search_type",
            "rag_context",
            "source_url",
            "doc_source_url",
            "generated_cypher",
            "result_index",
            "no_results",
            "error",
        }
        return cls(
            score=raw.get("score", 0.0),
            search_type=raw.get("search_type", ""),
            retrieved_text=raw.get("rag_context", ""),
            source_url=raw.get("source_url") or raw.get("doc_source_url"),
            cypher_result={k: v for k, v in raw.items() if k not in consumed_keys},
        )


@dataclass
class CypherExecution:
    """Tracks the Cypher query that was executed and whether it succeeded."""

    cypher_query: str | None
    cypher_execution_status: str  # 'success' | 'failed'
    cypher_error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage."""
        return {
            "cypher_query": self.cypher_query,
            "cypher_execution_status": self.cypher_execution_status,
            "cypher_error_message": self.cypher_error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CypherExecution":
        """Deserialize from a dict (as saved by to_dict)."""
        return cls(
            cypher_query=data.get("cypher_query"),
            cypher_execution_status=data.get("cypher_execution_status", "success"),
            cypher_error_message=data.get("cypher_error_message"),
        )


@dataclass
class QueryResult:
    """Result for a single query across any retriever."""

    query_id: int
    query: str
    cypher_execution: CypherExecution = field(
        default_factory=lambda: CypherExecution(
            cypher_query=None,
            cypher_execution_status="success",
            cypher_error_message=None,
        )
    )
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    relevance_annotations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "cypher_execution": self.cypher_execution.to_dict(),
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "relevance_annotations": self.relevance_annotations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryResult":
        """Deserialize from a dict (as saved by to_dict)."""
        return cls(
            query_id=data["query_id"],
            query=data["query"],
            cypher_execution=CypherExecution.from_dict(data["cypher_execution"]),
            retrieved_chunks=[
                RetrievedChunk.from_dict(c) for c in data.get("retrieved_chunks", [])
            ],
            relevance_annotations=data.get("relevance_annotations", []),
        )
