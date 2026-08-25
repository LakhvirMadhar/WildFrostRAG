"""VectorThenCypherRetriever for WildFrostRAG.

Combines vector similarity search with graph traversal to enrich results
with related Card, Tribe, CardType, Keyword, Stat, and other graph data.

The name "VectorThenCypher" makes the order explicit:
1. Vector search FIRST (find relevant documents)
2. Cypher traversal SECOND (enrich with graph data)
"""

from typing import Any
from collections.abc import Callable

from neo4j import Driver

from utils.config import get_settings
from rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever
from rag.retrievers.traversal_patterns import GRAPH_TRAVERSAL_QUERY


class VectorThenCypherRetriever(BaseNeo4jRetriever):
    """Retriever that combines vector search with graph traversal enrichment.

    This is the "Graph RAG" approach: use semantic similarity to find relevant
    Document nodes, then traverse the graph to enrich results with structured
    Card/Tribe/CardType/Keyword/Stat data.

    Flow:
        1. Embed query
        2. Vector search finds relevant Document nodes
        3. Cypher traversal enriches with graph data
        4. Return combined results with rag_context
    """

    def __init__(
        self,
        driver: Driver,
        embed_fn: Callable[[str], list[float]],
        neo4j_database: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """Initialize the VectorThenCypherRetriever.

        Args:
            driver: Neo4j driver instance
            embed_fn: Function that encodes a query string into a list of floats
            neo4j_database: Optional database name
            index_name: Vector index name (default: from settings)
        """
        super().__init__(driver, neo4j_database)
        self._embed_fn = embed_fn
        self.index_name = index_name or get_settings().vector_index_name

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search using vector similarity + graph traversal.

        Args:
            query: Natural language query
            k: Number of results to return

        Returns:
            List of enriched results with Card/Tribe/CardType/Keyword/Stat data
        """
        query_embedding = self._embed_fn(query)

        combined_query = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
        YIELD node as doc, score
        {GRAPH_TRAVERSAL_QUERY}
        """

        params = {
            "index_name": self.index_name,
            "query_embedding": query_embedding,
            "k": k,
        }

        results = self._execute_query(combined_query, params)
        return self._add_metadata(results, "vector_then_cypher")
