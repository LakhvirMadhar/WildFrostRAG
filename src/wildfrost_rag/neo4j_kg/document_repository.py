"""Repository for reading Document nodes from Neo4j.

Owns the raw index-search Cypher queries that used to live inline in
Neo4jVectorSearch, Neo4jFullTextSearch, and BM25Retriever (T4.3) - each of
those retrievers now depends on this repository instead of executing Cypher
itself, matching the DI pattern already used for the driver.
"""

from typing import Any

from neo4j import Driver

from wildfrost_rag.neo4j_kg.record_utils import record_to_dict


class DocumentRepository:
    """Reads Document nodes from Neo4j - vector search, fulltext search, bulk load."""

    def __init__(self, driver: Driver, neo4j_database: str | None = None) -> None:
        """Initialize the repository.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
        """
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.last_cypher_query: str | None = None

    def _run(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.last_cypher_query = query
        with self.driver.session(database=self.neo4j_database) as session:
            results = session.run(query, params)
            return [record_to_dict(record) for record in results]

    def vector_search(
        self, index_name: str, query_embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Vector similarity search against a Document vector index.

        Args:
            index_name: Name of the Neo4j vector index to query
            query_embedding: Embedded query vector
            k: Number of top results to return

        Returns:
            List of raw result dicts (node properties + score)
        """
        query = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """
        params = {"index_name": index_name, "query_embedding": query_embedding, "k": k}
        return self._run(query, params)

    def fulltext_search(self, index_name: str, query_text: str, k: int) -> list[dict[str, Any]]:
        """Lucene-based full-text search against a Document fulltext index.

        Args:
            index_name: Name of the Neo4j fulltext index to query
            query_text: Query string (already preprocessed by the caller, e.g. stop words removed)
            k: Number of top results to return

        Returns:
            List of raw result dicts (node properties + score)
        """
        query = """
        CALL db.index.fulltext.queryNodes($index_name, $query)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $k
        """
        params = {"index_name": index_name, "query": query_text, "k": k}
        return self._run(query, params)

    def load_all_documents(self, label: str) -> list[tuple[str, dict[str, Any]]]:
        """Load every Document (or Document-labeled) node's text and properties.

        Used by BM25Retriever to build its in-memory index - loads the whole
        corpus rather than searching, so it returns raw (text, node_properties)
        pairs instead of the score-annotated shape the search methods return.

        Args:
            label: Node label to load (BM25Retriever's index_name doubles as
                the label it queries, e.g. "Document")

        Returns:
            List of (text, node_properties_without_embedding) tuples
        """
        query = f"""
        MATCH (d:{label})
        WHERE d.text IS NOT NULL
        RETURN d.text AS text, d
        """
        self.last_cypher_query = query
        with self.driver.session(database=self.neo4j_database) as session:
            results = session.run(query)
            return [
                (record["text"], {k: v for k, v in record["d"].items() if k != "embedding"})
                for record in results
            ]
