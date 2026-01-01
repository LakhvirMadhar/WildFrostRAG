"""
Neo4j-based full-text search retriever for WildFrostRAG.

This module implements lexical retrieval using Neo4j's built-in full-text search
capabilities, which are based on Apache Lucene.
"""

from typing import List, Dict, Any
from src.utils.config import settings
from .base_neo4j_retriever import BaseNeo4jRetriever


class Neo4jFullTextSearch(BaseNeo4jRetriever):
    """
    Implements lexical similarity retrieval using Neo4j's full-text search.
    This corresponds to the 'BM25' (or lexical search) approach in the research goals,
    using Neo4j's Lucene-based full-text search as a proxy.
    """

    def __init__(self):
        """
        Initialize the Neo4j full-text search retriever.
        """
        super().__init__()
        self.index_name = settings.fulltext_index_name

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant document chunks from Neo4j based on lexical similarity.

        This method performs full-text search using Neo4j's built-in full-text indexing
        capabilities, which implements Lucene-based search algorithms.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        # Perform full-text search
        search_query = f"""
        CALL db.index.fulltext.queryNodes($index_name, $query)
        YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        LIMIT $k
        """

        params = {
            "index_name": self.index_name,
            "query": query,
            "k": k
        }

        results = self._execute_query(search_query, params)

        # Add search type to distinguish from other retrieval methods
        for result in results:
            result["search_type"] = "fulltext"

        return results