"""Repository for Document reads enriched with Card/entity graph data.

Owns the graph-traversal-enriched Cypher queries that used to live inline in
VectorThenCypherRetriever and FulltextThenCypherRetriever (T4.3). Each query
is a single atomic Cypher statement (index lookup piped directly into
GRAPH_TRAVERSAL_QUERY), so it stays one repository call per search rather
than two separate round trips.
"""

from typing import Any

from neo4j import Driver

from wildfrost_rag.neo4j_kg.record_utils import record_to_dict
from wildfrost_rag.neo4j_kg.traversal_patterns import GRAPH_TRAVERSAL_QUERY


class CardRepository:
    """Reads Documents enriched with their owning Card/Bell/Charm/etc. graph data."""

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

    def vector_search_with_enrichment(
        self, index_name: str, query_embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Vector search against a Document index, enriched with graph data.

        Args:
            index_name: Name of the Neo4j vector index to query
            query_embedding: Embedded query vector
            k: Number of top results to return

        Returns:
            List of raw result dicts (doc + score + enrichment fields)
        """
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
        YIELD node as doc, score
        {GRAPH_TRAVERSAL_QUERY}
        """
        params = {"index_name": index_name, "query_embedding": query_embedding, "k": k}
        return self._run(query, params)

    def fulltext_search_with_enrichment(
        self, index_name: str, query_text: str, k: int
    ) -> list[dict[str, Any]]:
        """Fulltext search against a Document index, enriched with graph data.

        Args:
            index_name: Name of the Neo4j fulltext index to query
            query_text: Query string (already preprocessed by the caller, e.g. stop words removed)
            k: Number of top results to return

        Returns:
            List of raw result dicts (doc + score + enrichment fields)
        """
        query = f"""
        CALL db.index.fulltext.queryNodes($index_name, $query)
        YIELD node as doc, score
        {GRAPH_TRAVERSAL_QUERY}
        LIMIT $k
        """
        params = {"index_name": index_name, "query": query_text, "k": k}
        return self._run(query, params)
