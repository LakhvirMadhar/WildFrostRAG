"""Base class for Neo4j-based retrievers in WildFrostRAG.

This module provides a common base for different retrieval strategies using Neo4j.
"""

from typing import Any
from urllib.parse import urlparse
from neo4j import Driver, Record
from neo4j.graph import Node, Relationship, Path
from wildfrost_rag.utils.config import get_settings

# Types for Neo4j values before and after serialization
Neo4jValue = Node | Relationship | Path | list[Any] | str | int | float | bool | None
SerializedValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class BaseNeo4jRetriever:
    """Base class for Neo4j-based retrievers.

    Follows dependency injection pattern - driver is passed in, not created here.
    """

    # Subclasses can override to specify which result field to use as rag_context
    # If None, _format_result_as_text() is used as fallback
    rag_context_field: str | None = None

    def __init__(self, driver: Driver, neo4j_database: str | None = None) -> None:
        """Initialize the base Neo4j retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
        """
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.last_cypher_query: str | None = None

        # Extract port for logging (security: don't log full URI/host)
        # Note: We can't get URI directly from driver, so we'll get it from settings for logging
        uri = get_settings().neo4j.uri.get_secret_value()
        parsed_uri = urlparse(uri)
        self.port = parsed_uri.port or 7687  # Default Neo4j port

    @staticmethod
    def _serialize_value(value: Neo4jValue) -> SerializedValue:
        """Convert a Neo4j graph object to a JSON-serializable Python type.

        Handles Node, Relationship, and Path objects that the LLM-generated
        Cypher may return (e.g., `RETURN c` instead of `RETURN c.card_name`).
        """
        if isinstance(value, Node):
            props = {
                k: v for k, v in value.items() if k != "embedding" and not k.endswith("_embedding")
            }
            props["_labels"] = list(value.labels)
            return props
        if isinstance(value, Relationship):
            return {"_type": value.type, **dict(value.items())}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [BaseNeo4jRetriever._serialize_value(v) for v in value]
        return value

    def _record_to_dict(self, record: Record) -> dict[str, Any]:
        """Convert any Neo4j record to a flat dictionary.

        Handles any Cypher query result structure:
        - Single nodes: RETURN node, score
        - Multiple nodes: RETURN d, c, t, score
        - Scalars: RETURN count(*), name
        - Mixed: Any combination
        - Raw Node/Relationship/Path objects from LLM-generated Cypher

        For backward compatibility:
        - Variable named 'node' has properties extracted WITHOUT prefix (text, source_file)
        - Other node variables are prefixed (d_text, c_card_name) to avoid collisions

        Args:
            record: Neo4j record from query result

        Returns:
            Flattened dictionary with all properties
        """
        result = {}

        for key in record.keys():
            value = record[key]
            if value is None:
                continue

            # Handle Neo4j Node/Relationship objects - extract their properties
            if isinstance(value, (Node, Relationship)):
                for prop_key, prop_value in value.items():
                    if prop_key == "embedding" or prop_key.endswith("_embedding"):
                        continue  # Skip embedding vectors

                    serialized = self._serialize_value(prop_value)

                    # Backward compat: 'node' variable doesn't get prefixed
                    # Other variables (d, c, t, etc.) get prefixed to avoid collisions
                    if key == "node":
                        result[prop_key] = serialized
                    else:
                        result[f"{key}_{prop_key}"] = serialized
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, list):
                result[key] = self._serialize_value(value)
            else:
                # Scalar values (score, strings, ints, etc.)
                result[key] = value

        return result

    def _execute_query(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute a Neo4j query and return results using the shared driver.

        Handles ANY Cypher query structure - single nodes, multiple nodes,
        scalars, or mixed results.

        Args:
            query: Cypher query to execute
            params: Parameters for the query

        Returns:
            List of dictionaries containing query results
        """
        self.last_cypher_query = query
        with self.driver.session(database=self.neo4j_database) as session:
            results = session.run(query, params)
            return [self._record_to_dict(record) for record in results]

    def _format_result_as_text(self, result: dict[str, Any]) -> str:
        """Format a result dictionary as human-readable text for RAG context.

        Excludes metadata fields and formats the remaining fields as readable text.
        Override in subclasses for custom formatting.
        """
        exclude_fields = {
            "score",
            "search_type",
            "generated_cypher",
            "result_index",
            "rag_context",
        }
        lines = []

        for key, value in result.items():
            if key in exclude_fields:
                continue
            if value is None or value == "":
                continue
            if key.endswith("_embedding") or key == "embedding":
                continue

            formatted_key = key.replace("_", " ").title()
            lines.append(f"{formatted_key}: {value}")

        return "\n".join(lines)

    def _add_metadata(
        self, results: list[dict[str, Any]], search_type: str
    ) -> list[dict[str, Any]]:
        """Add metadata to search results and ensure rag_context field exists.

        Args:
            results: List of retrieval results
            search_type: Type of search performed (e.g., 'vector', 'fulltext', 'bm25')

        Returns:
            Results with added metadata and standardized rag_context field
        """
        for result in results:
            result["search_type"] = search_type

            # Standardize RAG context - ensure every result has this field
            if "rag_context" not in result:
                # Use configured field if set, otherwise format all properties
                if self.rag_context_field and self.rag_context_field in result:
                    result["rag_context"] = result[self.rag_context_field]
                else:
                    result["rag_context"] = self._format_result_as_text(result)

        return results
