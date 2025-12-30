"""
Neo4j vector store operations for WildFrostRAG.

This module handles ingestion of embeddings into Neo4j, creation of vector
indices, and vector similarity search operations.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from src.utils.logger import logger


def ingest_embeddings_into_neo4j(
    uri: str,
    user: str,
    password: str,
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
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
        user: Neo4j username
        password: Neo4j password
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
        f"Ingesting {len(document_chunks)} document chunks into Neo4j at {uri}"
    )
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
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
                d.source_file = item.source_file,
                d.header1 = item.header1,
                d.header2 = item.header2,
                d.header3 = item.header3
            """
            
            # Prepare data for ingestion
            data_to_ingest = [
                {
                    "text": chunk.page_content,
                    "embedding": embedding.tolist(),
                    "source_file": chunk.metadata.get('source', 'unknown'),
                    "header1": chunk.metadata.get('Header 1', ''),
                    "header2": chunk.metadata.get('Header 2', ''),
                    "header3": chunk.metadata.get('Header 3', '')
                }
                for chunk, embedding in zip(document_chunks, embeddings)
            ]
            
            session.run(cypher_query, parameters={"data": data_to_ingest})
            logger.info("Data ingestion complete")
            
    finally:
        driver.close()


def create_vector_index(
    uri: str,
    user: str,
    password: str,
    index_name: str,
    embedding_dimension: int,
    node_label: str = "Document",
    embedding_property: str = "embedding",
    similarity_function: str = "cosine"
) -> None:
    """
    Create a vector index in Neo4j for similarity search.
    
    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        index_name: Name for the vector index
        embedding_dimension: Dimensionality of embedding vectors
        node_label: Node label to index (default: "Document")
        embedding_property: Property containing embeddings (default: "embedding")
        similarity_function: Similarity metric to use (default: "cosine")
    
    Note:
        If the index already exists, this function will skip creation
        and log a message.
    """
    logger.info(f"Creating vector index '{index_name}' in Neo4j")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Check if index already exists
            index_exists_query = "SHOW INDEXES YIELD name WHERE name = $name"
            if session.run(index_exists_query, name=index_name).single():
                logger.info(f"Vector index '{index_name}' already exists. Skipping creation.")
                return
            
            # Create the vector index
            create_query = f"""
            CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
            FOR (d:{node_label}) ON (d.{embedding_property})
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {embedding_dimension},
                `vector.similarity_function`: "{similarity_function}"
              }}
            }}
            """
            
            session.run(create_query)
            logger.info(f"Vector index '{index_name}' successfully created")
            
    finally:
        driver.close()


def get_retrieved_chunks(
    query: str,
    uri: str,
    user: str,
    password: str,
    embedding_model: SentenceTransformer,
    index_name: str = "document-embeddings",
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant document chunks using vector search.
    
    Args:
        query: User's search query
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        embedding_model: Pre-loaded SentenceTransformer model for query embedding
        index_name: Name of the vector index to query
        k: Number of results to return
    
    Returns:
        List of dictionaries containing 'text' and 'score' for each result
    
    Example:
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> results = get_retrieved_chunks(
        ...     query="What is Azul Candle?",
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password",
        ...     embedding_model=model,
        ...     k=5
        ... )
        >>> for result in results:
        ...     print(f"Score: {result['score']}, Text: {result['text'][:50]}...")
    """
    logger.info(f"Retrieving top-{k} chunks for query: '{query}'")
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Vector similarity search query
            search_query = f"""
            CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
            YIELD node, score
            RETURN node.text AS text, score
            ORDER BY score DESC
            """
            
            results = session.run(
                search_query,
                index_name=index_name,
                query_embedding=query_embedding,
                k=k
            )
            
            # Extract results
            retrieved_chunks = [
                {"text": record["text"], "score": record["score"]}
                for record in results
            ]
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            return retrieved_chunks
            
    finally:
        driver.close()


def wait_for_index_population(seconds: int = 5) -> None:
    """
    Wait for Neo4j vector index to be fully populated.
    
    After creating a vector index, Neo4j needs time to populate it.
    This is a simple helper to add a delay.
    
    Args:
        seconds: Number of seconds to wait (default: 5)
    """
    logger.info(f"Waiting {seconds} seconds for index to be fully populated...")
    time.sleep(seconds)
    logger.info("Wait complete")


def link_documents_to_cards(
    uri: str,
    user: str,
    password: str
) -> int:
    """
    Link Document nodes to Card nodes in the knowledge graph.
    
    This function creates HAS_DOCUMENT relationships between Card nodes
    and Document nodes by matching the source_file property of Documents
    to the card_name property of Cards.
    
    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
    
    Returns:
        Number of relationships created
    
    Example:
        After ingesting documents and cards:
        >>> count = link_documents_to_cards(
        ...     uri="bolt://localhost:7687",
        ...     user="neo4j",
        ...     password="password"
        ... )
        >>> print(f"Created {count} Card-Document relationships")
    """
    logger.info("Linking Document nodes to Card nodes...")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
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
