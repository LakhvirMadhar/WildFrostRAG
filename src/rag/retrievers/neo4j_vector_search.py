"""Neo4j-based retrieval implementations for WildFrostRAG.

This module provides various retrieval strategies using Neo4j:
- Vector search (semantic similarity)
- Future implementations: BM25, Hybrid, Text2Cypher, Graph RAG
"""

from typing import Any
from collections.abc import Callable
from neo4j import Driver
from utils.config import settings
from rag.retrievers.base_neo4j_retriever import BaseNeo4jRetriever


class Neo4jVectorSearch(BaseNeo4jRetriever):
    """Implements semantic similarity retrieval using Neo4j vector indexes.

    This corresponds to the 'Cosine Similarity' approach in the research goals.
    """

    def __init__(
        self,
        driver: Driver,
        embed_fn: Callable[[str], list[float]],
        neo4j_database: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """Initialize the Neo4j vector search retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            embed_fn: Function that encodes a query string into a list of floats.
                      Use get_query_embed_fn() from embeddings.query_embedders.
            neo4j_database: Optional database name (default: None uses default database)
            index_name: Optional vector index name (default: uses settings.vector_index_name)
        """
        super().__init__(driver, neo4j_database)
        self._embed_fn = embed_fn
        self.index_name = index_name or settings.vector_index_name

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant document chunks from Neo4j based on semantic similarity.

        This method performs vector similarity search using Neo4j's built-in vector indexing
        capabilities, which implements cosine similarity between embedded queries and stored
        document vectors.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        # Step 1: Embed the user's query
        query_embedding = self._embed_fn(query)

        # Step 2: Perform vector similarity search
        search_query = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """

        params = {
            "index_name": self.index_name,
            "query_embedding": query_embedding,
            "k": k,
        }

        results = self._execute_query(search_query, params)
        return self._add_metadata(results, "vector")
