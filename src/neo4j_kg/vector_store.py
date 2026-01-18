"""
Neo4j vector store operations for WildFrostRAG.

This module handles ingestion of document embeddings into Neo4j and
vector similarity search operations.
"""

import numpy as np
from typing import List, Dict, Any
from neo4j import GraphDatabase
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from src.utils.config import settings
from src.utils.logger import logger


def ingest_embeddings_into_neo4j(
    document_chunks: List[Document],
    embeddings: np.ndarray,
    chunk_label: str = "Document",
    text_property: str = "text",
    embedding_property: str = "embedding"
) -> None:
    """
    Ingest document chunks and their embeddings into Neo4j.

    This function creates Document nodes with text content, embeddings,
    and metadata from the chunks. Uses MERGE to avoid duplicates.

    Args:
        document_chunks: List of LangChain Document objects
        embeddings: NumPy array of embeddings matching the chunks
        chunk_label: Node label to use in Neo4j (default: "Document")
        text_property: Property name for text content (default: "text")
        embedding_property: Property name for embedding vector (default: "embedding")

    Raises:
        ValueError: If the number of chunks and embeddings don't match
    """
    if len(document_chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(document_chunks)} chunks but {len(embeddings)} embeddings"
        )

    logger.info(
        f"Ingesting {len(document_chunks)} document chunks into Neo4j database"
    )

    driver = GraphDatabase.driver(settings.neo4j_uri.get_secret_value(), auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))

    try:
        driver.verify_connectivity()
        logger.info("Connection to Neo4j successful")

        with driver.session() as session:
            # Cypher query with MERGE to avoid duplicates
            cypher_query = f"""
            UNWIND $data AS item
            MERGE (d:{chunk_label} {{
                {text_property}: item.text
            }})
            ON CREATE SET
                d.{embedding_property} = item.embedding,
                d.source_file = item.source_file
            """

            # Prepare data for ingestion
            data_to_ingest = [
                {
                    "text": chunk.page_content,
                    "embedding": embedding.tolist(),
                    "source_file": chunk.metadata.get('source', 'unknown')
                }
                for chunk, embedding in zip(document_chunks, embeddings)
            ]

            session.run(cypher_query, parameters={"data": data_to_ingest})
            logger.info("Data ingestion complete")

    finally:
        driver.close()


def get_retrieved_chunks(
    query: str,
    embedding_model: SentenceTransformer,
    index_name: str = "document-embeddings",
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant document chunks using vector search.

    Args:
        query: User's search query
        embedding_model: Pre-loaded SentenceTransformer model for query embedding
        index_name: Name of the vector index to query
        k: Number of results to return

    Returns:
        List of dictionaries containing retrieved chunks with their metadata and scores

    Example:
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> results = get_retrieved_chunks(
        ...     query="What is Azul Candle?",
        ...     embedding_model=model,
        ...     k=5
        ... )
        >>> for result in results:
        ...     print(f"Score: {result['score']}, Text: {result['text'][:50]}...")
    """
    logger.info(f"Retrieving top-{k} chunks for query: '{query}'")

    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()

    driver = GraphDatabase.driver(settings.neo4j_uri.get_secret_value(), auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))

    try:
        with driver.session() as session:
            # Vector similarity search query
            search_query = f"""
            CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
            """

            results = session.run(
                search_query,
                index_name=index_name,
                query_embedding=query_embedding,
                k=k
            )

            # Extract results
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

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            return retrieved_chunks

    finally:
        driver.close()


def link_documents_to_cards() -> int:
    """
    Link Document nodes to Card nodes in the knowledge graph.

    This function creates HAS_DOCUMENT relationships between Card nodes
    and Document nodes by matching the source_file property of Documents
    to the card_name property of Cards.

    Returns:
        Number of relationships created

    Example:
        After ingesting documents and cards:
        >>> count = link_documents_to_cards()
        >>> print(f"Created {count} Card-Document relationships")
    """
    logger.info("Linking Document nodes to Card nodes...")

    driver = GraphDatabase.driver(settings.neo4j_uri.get_secret_value(), auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))

    try:
        with driver.session() as session:
            # Match exactly on the filename property we now store on the Card node
            # This handles cases where card_name has special chars (e.g. "Lil' Gazi" -> "Lil Gazi.html")
            link_query = """
            MATCH (d:Document)
            MATCH (c:Card)
            WHERE d.source_file ENDS WITH c.filename
            MERGE (c)-[:HAS_DOCUMENT]->(d)
            RETURN count(*) as relationships_created
            """

            result = session.run(link_query)
            record = result.single()
            count = record["relationships_created"] if record else 0

            logger.info(f"Created {count} Card-Document relationships")
            return count

    finally:
        driver.close()
