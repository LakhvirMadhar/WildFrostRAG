"""
Neo4j-based full-text search retriever for WildFrostRAG.

This module implements lexical retrieval using Neo4j's built-in full-text search
capabilities, which are based on Apache Lucene.
"""

from typing import List, Dict, Any, Optional
from neo4j import Driver
from src.utils.config import settings
from src.utils.logger import logger
from .base_neo4j_retriever import BaseNeo4jRetriever


class Neo4jFullTextSearch(BaseNeo4jRetriever):
    """
    Implements lexical similarity retrieval using Neo4j's full-text search.
    This corresponds to the 'BM25' (or lexical search) approach in the research goals,
    using Neo4j's Lucene-based full-text search as a proxy.
    """

    def __init__(self, driver: Driver, neo4j_database: Optional[str] = None):
        """
        Initialize the Neo4j full-text search retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
        """
        super().__init__(driver, neo4j_database)
        self.index_name = settings.fulltext_index_name

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant document chunks from Neo4j based on lexical similarity.

        This method performs full-text search using Neo4j's built-in full-text indexing
        capabilities, which implements Lucene-based search algorithms.

        Note: The fulltext index must already exist (created during pipeline setup).
              If index doesn't exist, Neo4j will raise an error.

        Args:
            query: Input query string to search for
            k: Number of top results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with their metadata and scores
        """
        # Perform full-text search (index must already exist)
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

        try:
            results = self._execute_query(search_query, params)
            return self._add_metadata(results, 'fulltext')
        except Exception as e:
            logger.error(
                f"Fulltext search failed. Index '{self.index_name}' may not exist. "
                f"Run 'python -m scripts.ingest_data --no-chunking' to create it. "
                f"Error: {e}"
            )
            raise