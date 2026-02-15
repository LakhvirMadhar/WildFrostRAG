"""
Neo4j index management for WildFrostRAG.

This module handles creation and management of Neo4j indexes:
- Vector indexes for semantic similarity search
- Full-text indexes for lexical keyword search
"""

import time
from neo4j import GraphDatabase
from src.utils.config import settings
from src.utils.logger import logger


def create_vector_index(
    index_name: str,
    embedding_dimension: int,
    node_label: str = "Document",
    embedding_property: str = "embedding",
    similarity_function: str = "cosine"
) -> None:
    """
    Create a vector index in Neo4j for similarity search.

    Args:
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

    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
    )

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


def create_fulltext_index(
    session,
    index_name: str,
    node_label: str = "Document",
    text_property: str = "text"
) -> None:
    """
    Create a full-text index in Neo4j for lexical search.

    Uses Neo4j's Lucene-based full-text indexing capabilities.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)
        index_name: Name for the full-text index
        node_label: Node label to index (default: "Document")
        text_property: Property containing text content (default: "text")

    Note:
        If the index already exists, this function will skip creation
        and log a message.
    """
    logger.info(f"Creating full-text index '{index_name}' in Neo4j")

    # Check if index already exists
    index_exists_query = "SHOW INDEXES YIELD name WHERE name = $name"
    if session.run(index_exists_query, name=index_name).single():
        logger.info(f"Full-text index '{index_name}' already exists. Skipping creation.")
        return

    # Create the full-text index using Neo4j 5.x syntax
    create_query = f"""
    CREATE FULLTEXT INDEX `{index_name}` IF NOT EXISTS
    FOR (n:{node_label}) ON EACH [n.{text_property}]
    """

    session.run(create_query)
    logger.info(f"Full-text index '{index_name}' successfully created")


def wait_for_index_population(seconds: int = 5) -> None:
    """
    Wait for Neo4j indexes to be fully populated.

    After creating an index, Neo4j needs time to populate it.
    This is a simple helper to add a delay.

    Args:
        seconds: Number of seconds to wait (default: 5)
    """
    logger.info(f"Waiting {seconds} seconds for index to be fully populated...")
    time.sleep(seconds)
    logger.info("Wait complete")
