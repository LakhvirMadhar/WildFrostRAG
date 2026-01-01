"""
Base class for Neo4j-based retrievers in WildFrostRAG.

This module provides a common base for different retrieval strategies using Neo4j.
"""

from typing import List, Dict, Any
from neo4j import GraphDatabase
from src.utils.config import settings


class BaseNeo4jRetriever:
    """
    Base class for Neo4j-based retrievers.
    Provides common functionality for connecting to Neo4j.
    """
    
    def __init__(self):
        """
        Initialize the base Neo4j retriever.
        """
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password.get_secret_value()

    def _execute_query(self, query: str, params: dict) -> List[Dict[str, Any]]:
        """
        Execute a Neo4j query and return results.
        
        Args:
            query: Cypher query to execute
            params: Parameters for the query
            
        Returns:
            List of dictionaries containing query results
        """
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            with driver.session() as session:
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
                        if key != "embedding": # Exclude the large vector
                            chunk_dict[key] = value

                    retrieved_chunks.append(chunk_dict)

            return retrieved_chunks
        finally:
            driver.close()