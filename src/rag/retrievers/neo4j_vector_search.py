"""
Neo4j-based retrieval implementations for WildFrostRAG.

This module provides various retrieval strategies using Neo4j:
- Vector search (semantic similarity)
- Future implementations: BM25, Hybrid, Text2Cypher, Graph RAG
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from src.utils.config import settings


class Neo4jVectorSearch:
    """
    Implements semantic similarity retrieval using Neo4j vector indexes.
    This corresponds to the 'Cosine Similarity' approach in the research goals.
    """

    _instance = None
    _embedding_model = None

    def __init__(self):
        """
        Initialize the Neo4j vector search retriever.

        Sets up connection parameters and prepares the embedding model.
        """
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password.get_secret_value()
        self.index_name = settings.vector_index_name

    @classmethod
    def get_embedding_model(cls):
        """
        Lazy load the embedding model to avoid overhead on import.

        Returns:
            Loaded SentenceTransformer model instance
        """
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformer(settings.embedding_model_name)
        return cls._embedding_model

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant document chunks from Neo4j based on semantic similarity.

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
        model = self.get_embedding_model()
        query_embedding = model.encode(query).tolist()

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        try:
            with driver.session() as session:
                # Step 2: Perform vector similarity search
                search_query = f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
                YIELD node, score
                RETURN node, score
                ORDER BY score DESC
                """

                results = session.run(
                    search_query,
                    index_name=self.index_name,
                    query_embedding=query_embedding,
                    k=k
                )

                # Step 3: Extract properties
                retrieved_chunks = []
                for record in results:
                    node = record["node"]
                    # Start with score
                    chunk_dict = {
                        "score": record["score"],
                    }
                    # Flatten all node properties into the dict
                    # This includes 'text', 'source_file', 'header1', etc.
                    for key, value in node.items():
                        if key != "embedding": # Exclude the large vector
                            chunk_dict[key] = value

                    retrieved_chunks.append(chunk_dict)

            return retrieved_chunks
        finally:
            driver.close()
