"""
Base class for Neo4j-based retrievers in WildFrostRAG.

This module provides a common base for different retrieval strategies using Neo4j.
"""

from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from neo4j import Driver
from src.utils.config import settings


class BaseNeo4jRetriever:
    """
    Base class for Neo4j-based retrievers.
    Follows dependency injection pattern - driver is passed in, not created here.
    """

    def __init__(self, driver: Driver, neo4j_database: Optional[str] = None):
        """
        Initialize the base Neo4j retriever.

        Args:
            driver: Neo4j driver instance (created externally, managed by application)
            neo4j_database: Optional database name (default: None uses default database)
        """
        self.driver = driver
        self.neo4j_database = neo4j_database

        # Extract port for logging (security: don't log full URI/host)
        # Note: We can't get URI directly from driver, so we'll get it from settings for logging
        uri = settings.neo4j_uri.get_secret_value()
        parsed_uri = urlparse(uri)
        self.port = parsed_uri.port or 7687  # Default Neo4j port

    def _execute_query(self, query: str, params: dict) -> List[Dict[str, Any]]:
        """
        Execute a Neo4j query and return results using the shared driver.

        Args:
            query: Cypher query to execute
            params: Parameters for the query

        Returns:
            List of dictionaries containing query results
        """
        with self.driver.session(database=self.neo4j_database) as session:
            results = session.run(query, params)

            retrieved_chunks = []
            for record in results:
                node = record["node"]
                # Start with score
                chunk_dict = {
                    "score": record["score"],
                }
                # Flatten all node properties into the dict
                # This includes 'text', 'source_file', etc.
                for key, value in node.items():
                    if key != "embedding":  # Exclude the large vector
                        chunk_dict[key] = value

                retrieved_chunks.append(chunk_dict)

        return retrieved_chunks

    def _add_metadata(self, results: List[Dict[str, Any]], search_type: str) -> List[Dict[str, Any]]:
        """
        Add metadata to search results.

        Args:
            results: List of retrieval results
            search_type: Type of search performed (e.g., 'vector', 'fulltext', 'bm25')

        Returns:
            Results with added metadata
        """
        for result in results:
            result['search_type'] = search_type
        return results