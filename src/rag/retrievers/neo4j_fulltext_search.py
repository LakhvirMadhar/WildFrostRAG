"""
Neo4j-based full-text search retriever for WildFrostRAG.

This module implements lexical retrieval using Neo4j's built-in full-text search
capabilities, which are based on Apache Lucene.
"""

from typing import List, Dict, Any
from neo4j import GraphDatabase
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

    def _ensure_fulltext_index_exists(self):
        """
        Ensure the fulltext index exists, create it if it doesn't.
        """
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            with driver.session() as session:
                # Check if the index exists
                check_query = """
                CALL db.indexes() YIELD name, type, state, populationProgress, uniqueness, entityCount, labelsOrTypes, properties
                WHERE name = $index_name
                RETURN count(*) AS indexCount
                """

                result = session.run(check_query, index_name=self.index_name)
                index_exists = result.single()["indexCount"] > 0

                if not index_exists:
                    # Create the fulltext index
                    create_query = f"""
                    CALL db.index.fulltext.createNodeIndex(
                        $index_name,
                        ['Document'],  // Label to index
                        ['text']       // Properties to index
                    )
                    """
                    session.run(create_query, index_name=self.index_name)

                    # Wait for index to be populated
                    wait_query = f"""
                    CALL db.indexes() YIELD name, state
                    WHERE name = $index_name AND state = 'ONLINE'
                    RETURN count(*) AS readyCount
                    """

                    # Wait until the index is ready
                    import time
                    max_wait_time = 30  # seconds
                    wait_time = 0
                    while wait_time < max_wait_time:
                        result = session.run(wait_query, index_name=self.index_name)
                        if result.single()["readyCount"] > 0:
                            break
                        time.sleep(1)
                        wait_time += 1
        finally:
            driver.close()

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
        # Ensure the fulltext index exists
        self._ensure_fulltext_index_exists()

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