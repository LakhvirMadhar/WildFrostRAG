"""
Neo4j vector store operations for WildFrostRAG.

This module handles ingestion of document embeddings into Neo4j and
vector similarity search operations.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from src.utils.config import settings
from src.utils.logger import logger


def ingest_documents_into_neo4j(
    document_chunks: List[Document],
    url_lookup: Optional[Dict[str, str]] = None,
    chunk_label: str = "Document",
    text_property: str = "text"
) -> None:
    """
    Ingest document chunks into Neo4j as Document nodes.

    This function creates Document nodes with text content and metadata only.
    No embeddings are included - use add_embeddings.py to add embeddings later.
    Uses MERGE to avoid duplicates.

    Args:
        document_chunks: List of LangChain Document objects
        url_lookup: Optional dict mapping filename to wiki URL
        chunk_label: Node label to use in Neo4j (default: "Document")
        text_property: Property name for text content (default: "text")
    """
    logger.info(
        f"Ingesting {len(document_chunks)} document chunks into Neo4j database"
    )

    if url_lookup is None:
        url_lookup = {}

    driver = GraphDatabase.driver(settings.neo4j_uri.get_secret_value(), auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))

    try:
        driver.verify_connectivity()
        logger.info("Connection to Neo4j successful")

        with driver.session() as session:
            # Cypher query - text, metadata, and source URL
            cypher_query = f"""
            UNWIND $data AS item
            MERGE (d:{chunk_label} {{
                {text_property}: item.text
            }})
            ON CREATE SET
                d.source_file = item.source_file,
                d.title = item.title,
                d.source_url = item.source_url
            """

            # Prepare data without embeddings
            data_to_ingest = []
            for chunk in document_chunks:
                full_path = chunk.metadata.get('source', 'unknown')

                # Extract filename and make path relative to data/
                filename = Path(full_path).name if full_path else ''

                # Convert to relative path (data/structured_outputs/...)
                source_file = full_path
                if full_path and 'data' in full_path:
                    data_idx = full_path.find('data')
                    source_file = full_path[data_idx:].replace('\\', '/')

                # Derive title from filename (without extension)
                title = Path(filename).stem if filename else 'unknown'

                source_url = url_lookup.get(filename, '')

                data_to_ingest.append({
                    "text": chunk.page_content,
                    "source_file": source_file,
                    "title": title,
                    "source_url": source_url
                })

            session.run(cypher_query, parameters={"data": data_to_ingest})
            logger.info("Data ingestion complete")

    finally:
        driver.close()


def create_embedding_index(
    property_name: str,
    index_name: str,
    dimension: int
) -> None:
    """
    Create a vector index for a specific embedding property.

    This allows storing multiple embedding providers' vectors on the same
    Document nodes and querying each provider's index independently.

    Args:
        property_name: Property name on Document nodes (e.g., "hf_embedding", "openai_embedding")
        index_name: Name for the vector index (e.g., "document-embeddings-hf")
        dimension: Vector dimensionality (e.g., 384 for HF, 1536 for OpenAI)

    Example:
        >>> create_embedding_index(
        ...     property_name="openai_embedding",
        ...     index_name="document-embeddings-openai",
        ...     dimension=1536
        ... )
    """
    logger.info(f"Creating vector index '{index_name}' for property '{property_name}' (dim={dimension})")

    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
    )

    try:
        driver.verify_connectivity()
        logger.info("Connection to Neo4j successful")

        with driver.session() as session:
            # Create vector index using Neo4j 5.x syntax
            create_index_query = f"""
            CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
            FOR (n:Document)
            ON n.{property_name}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """

            session.run(create_index_query)
            logger.info(f"Vector index '{index_name}' created successfully")

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
            search_query = """
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


def link_documents_to_crowns() -> int:
    """
    Link Crown nodes to the Crowns Document node.

    Both Crown and Cursed Crown nodes get linked to the same Crowns.html document.

    Returns:
        Number of relationships created
    """
    logger.info("Linking Crown nodes to Document...")

    driver = GraphDatabase.driver(settings.neo4j_uri.get_secret_value(), auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))

    try:
        with driver.session() as session:
            link_query = """
            MATCH (d:Document)
            WHERE d.source_file ENDS WITH 'Crowns.html'
            MATCH (crown:Crown)
            MERGE (crown)-[:HAS_DOCUMENT]->(d)
            RETURN count(*) as relationships_created
            """

            result = session.run(link_query)
            record = result.single()
            count = record["relationships_created"] if record else 0

            logger.info(f"Created {count} Crown-Document relationships")
            return count

    finally:
        driver.close()


def link_documents_to_stats() -> int:
    """
    Link Stat nodes to the Stats Document node.

    All Stat nodes get linked to the same Stats.html document.

    Returns:
        Number of relationships created
    """
    logger.info("Linking Stat nodes to Document...")

    driver = GraphDatabase.driver(settings.neo4j_uri.get_secret_value(), auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()))

    try:
        with driver.session() as session:
            link_query = """
            MATCH (d:Document)
            WHERE d.source_file ENDS WITH 'Stats.html'
            MATCH (stat:Stat)
            MERGE (stat)-[:HAS_DOCUMENT]->(d)
            RETURN count(*) as relationships_created
            """

            result = session.run(link_query)
            record = result.single()
            count = record["relationships_created"] if record else 0

            logger.info(f"Created {count} Stat-Document relationships")
            return count

    finally:
        driver.close()
