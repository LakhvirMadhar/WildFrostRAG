"""Domain exception hierarchy for WildFrostRAG.

All domain-specific failures raise a subclass of WildFrostRAGError instead of
a bare ValueError/RuntimeError or a dict-shaped {"error": "..."} return, so
callers can catch by type and read structured attributes instead of parsing
a message string.
"""


class WildFrostRAGError(Exception):
    """Base class for all WildFrostRAG domain exceptions."""


class RetrievalError(WildFrostRAGError):
    """Base class for failures raised by a retriever's search() call."""

    def __init__(self, message: str, retriever_type: str) -> None:
        """Initialize with a message and the retriever's search_type identifier.

        Args:
            message: Human-readable description of the failure.
            retriever_type: The retriever's search_type identifier (e.g.
                "bm25", "text2cypher_llm").
        """
        self.retriever_type = retriever_type
        super().__init__(message)


class CypherGenerationError(RetrievalError):
    """Raised when the LLM fails to generate a Cypher query from a query."""

    def __init__(self, query: str, reason: str) -> None:
        """Initialize with the failed query and the reason generation failed.

        Args:
            query: The natural language query that failed to convert.
            reason: Why generation failed (e.g. the underlying LLM error).
        """
        self.query = query
        self.reason = reason
        super().__init__(
            f"Failed to generate Cypher query for {query!r}: {reason}",
            retriever_type="text2cypher_llm",
        )


class CypherExecutionError(RetrievalError):
    """Raised when a generated Cypher query fails to execute against Neo4j."""

    def __init__(self, cypher_query: str, reason: str) -> None:
        """Initialize with the failed query and the reason execution failed.

        Args:
            cypher_query: The Cypher query that failed.
            reason: Why execution failed (e.g. the underlying Neo4j driver error).
        """
        self.cypher_query = cypher_query
        self.reason = reason
        super().__init__(
            f"Cypher query execution failed: {reason}",
            retriever_type="text2cypher_llm",
        )


class ExperimentNotFoundError(WildFrostRAGError):
    """Raised when experiment_tracker can't find a referenced experiment."""

    def __init__(self, experiment_id: str, experiment_type: str) -> None:
        """Initialize with the unresolved experiment reference and its type.

        Args:
            experiment_id: The experiment reference that couldn't be
                resolved (e.g. "bm25/001" or "latest/bm25").
            experiment_type: Either "retrieval" or "generation".
        """
        self.experiment_id = experiment_id
        self.experiment_type = experiment_type
        super().__init__(f"No {experiment_type} experiment found for reference {experiment_id!r}")


class EmbeddingError(WildFrostRAGError):
    """Raised when embedding generation fails for a given provider."""

    def __init__(self, provider: str, reason: str) -> None:
        """Initialize with the failing provider and the reason it failed.

        Args:
            provider: The embedding provider key (e.g. "hf", "openai", "gemma").
            reason: Why embedding generation failed.
        """
        self.provider = provider
        self.reason = reason
        super().__init__(f"Embedding generation failed for provider {provider!r}: {reason}")
