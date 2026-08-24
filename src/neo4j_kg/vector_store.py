"""Neo4j vector store operations for WildFrostRAG.

This module handles ingestion of document embeddings into Neo4j and
vector similarity search operations.

Pipeline functions (ingest, link) accept a Neo4j Session via dependency
injection — the caller (ingest_data.py) manages driver lifecycle.
"""

from pathlib import Path
from typing import Any
from neo4j import GraphDatabase, Session
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from neo4j_kg.query_utils import single_value
from utils.config import settings
from utils.logger import logger


def ingest_documents_into_neo4j(
    session: Session,
    document_chunks: list[Document],
    url_lookup: dict[str, str] | None = None,
    chunk_label: str = "Document",
    text_property: str = "text",
) -> None:
    """Ingest document chunks into Neo4j as Document nodes.

    This function creates Document nodes with text content and metadata only.
    No embeddings are included - use add_embeddings.py to add embeddings later.
    Uses MERGE to avoid duplicates.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)
        document_chunks: List of LangChain Document objects
        url_lookup: Optional dict mapping filename to wiki URL
        chunk_label: Node label to use in Neo4j (default: "Document")
        text_property: Property name for text content (default: "text")
    """
    logger.info(f"Ingesting {len(document_chunks)} document chunks into Neo4j database")

    if url_lookup is None:
        url_lookup = {}

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
        full_path = chunk.metadata.get("source", "unknown")

        # Extract filename and make path relative to data/
        filename = Path(full_path).name if full_path else ""

        # Convert to relative path (data/structured_outputs/...)
        source_file = full_path
        if full_path and "data" in full_path:
            data_idx = full_path.find("data")
            source_file = full_path[data_idx:].replace("\\", "/")

        # Derive title from filename (without extension)
        title = Path(filename).stem if filename else "unknown"

        source_url = url_lookup.get(filename, "")

        data_to_ingest.append(
            {
                "text": chunk.page_content,
                "source_file": source_file,
                "title": title,
                "source_url": source_url,
            }
        )

    session.run(cypher_query, parameters={"data": data_to_ingest})
    logger.info("Data ingestion complete")


def create_embedding_index(property_name: str, index_name: str, dimension: int) -> None:
    """Create a vector index for a specific embedding property.

    This allows storing multiple embedding providers' vectors on the same
    Document nodes and querying each provider's index independently.

    Args:
        property_name: Property name on Document nodes (e.g., "hf_embedding", "openai_embedding")
        index_name: Name for the vector index (e.g., "document-embeddings-hf")
        dimension: Vector dimensionality (e.g., 384 for HF, 1536 for OpenAI)
    """
    logger.info(
        f"Creating vector index '{index_name}' for property '{property_name}' (dim={dimension})"
    )

    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
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
    k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve the top-k most relevant document chunks using vector search.

    Args:
        query: User's search query
        embedding_model: Pre-loaded SentenceTransformer model for query embedding
        index_name: Name of the vector index to query
        k: Number of results to return

    Returns:
        List of dictionaries containing retrieved chunks with their metadata and scores
    """
    logger.info(f"Retrieving top-{k} chunks for query: '{query}'")

    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()

    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
    )

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
                k=k,
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
                    if key != "embedding":  # Exclude the large vector
                        chunk_dict[key] = value

                retrieved_chunks.append(chunk_dict)

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            return retrieved_chunks

    finally:
        driver.close()


def link_documents_to_cards(session: Session) -> int:
    """Link Document nodes to Card nodes in the knowledge graph.

    Creates HAS_DOCUMENT relationships between Card nodes and Document nodes
    by matching the source_file property of Documents to the card_name property of Cards.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking Document nodes to Card nodes...")

    link_query = """
    MATCH (d:Document)
    MATCH (c:Card)
    WHERE d.source_file ENDS WITH ('/' + c.filename)
    MERGE (c)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) as relationships_created
    """

    result = session.run(link_query)
    record = result.single()
    count = record["relationships_created"] if record else 0
    logger.info(f"Created {count} Card-Document relationships (by filename)")

    # Fallback: match by URL for multi-phase cards where filename differs
    link_by_url = """
    MATCH (c:Card)
    WHERE c.url IS NOT NULL AND NOT (c)-[:HAS_DOCUMENT]->(:Document)
    MATCH (d:Document) WHERE d.source_url = c.url
    MERGE (c)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) as relationships_created
    """

    result = session.run(link_by_url)
    record = result.single()
    url_count = record["relationships_created"] if record else 0
    if url_count > 0:
        logger.info(f"Created {url_count} Card-Document relationships (by URL fallback)")

    total = count + url_count
    logger.info(f"Total: {total} Card-Document relationships")
    return total


def link_documents_to_crowns(session: Session) -> int:
    """Link Crown nodes to the Crowns Document node.

    Both Crown and Cursed Crown nodes get linked to the same Crowns.html document.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking Crown nodes to Document...")

    link_query = """
    MATCH (d:Document)
    WHERE d.source_file ENDS WITH 'crowns/Crowns.html'
    MATCH (crown:Crown)
    MERGE (crown)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) as relationships_created
    """

    result = session.run(link_query)
    record = result.single()
    count = record["relationships_created"] if record else 0

    logger.info(f"Created {count} Crown-Document relationships")
    return count


def link_documents_to_stats(session: Session) -> int:
    """Link Stat nodes to their individual stat page Documents.

    Each Stat matches its own Document via the filename property.
    Additionally, all stats link to the summary Stats.html for overview context.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking Stat nodes to Documents...")

    total = 0

    # Per-stat Documents (individual pages with detailed mechanics)
    per_stat_query = """
    MATCH (d:Document)
    MATCH (stat:Stat)
    WHERE stat.filename IS NOT NULL
      AND d.source_file ENDS WITH ('/' + stat.filename)
    MERGE (stat)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) as relationships_created
    """
    result = session.run(per_stat_query)
    record = result.single()
    per_stat_count = record["relationships_created"] if record else 0
    total += per_stat_count

    # Summary Stats.html (overview table with all stats)
    summary_query = """
    MATCH (d:Document)
    WHERE d.source_file ENDS WITH 'stats/Stats.html'
    MATCH (stat:Stat)
    MERGE (stat)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) as relationships_created
    """
    result = session.run(summary_query)
    record = result.single()
    summary_count = record["relationships_created"] if record else 0
    total += summary_count

    logger.info(
        f"Created {total} Stat-Document relationships ({per_stat_count} per-stat + {summary_count} summary)"
    )
    return total


def link_documents_to_charms(session: Session) -> int:
    """Link Charm nodes to their individual charm page Documents.

    Each Charm matches its own Document via the filename property
    (same pattern as link_documents_to_cards). Additionally, all charms
    link to the summary Charms.html for overview context.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking Charm nodes to Documents...")

    total = 0

    # Per-charm Documents (individual pages with Strategy sections)
    per_charm_query = """
    MATCH (d:Document)
    MATCH (charm:Charm)
    WHERE charm.filename IS NOT NULL
      AND d.source_file ENDS WITH ('/' + charm.filename)
    MERGE (charm)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) as relationships_created
    """
    result = session.run(per_charm_query)
    record = result.single()
    per_charm_count = record["relationships_created"] if record else 0
    total += per_charm_count
    logger.info(f"  Linked {per_charm_count} charms to individual Documents")

    # Summary Charms.html (overview table with all charms)
    summary_query = """
    MATCH (d:Document)
    WHERE d.source_file ENDS WITH 'charms/Charms.html'
    MATCH (charm:Charm)
    MERGE (charm)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) as relationships_created
    """
    result = session.run(summary_query)
    record = result.single()
    summary_count = record["relationships_created"] if record else 0
    total += summary_count
    logger.info(f"  Linked {summary_count} charms to summary Charms.html")

    logger.info(f"Created {total} total Charm-Document relationships")
    return total


def link_documents_to_shades(session: Session) -> int:
    """Link shade Card nodes to the Shades.html overview Document.

    Individual shade cards already link to their own page Documents via
    link_documents_to_cards(). This additionally links them to the aggregate
    Shades page which contains summoning mechanics and summon conditions.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking shade cards to Shades overview Document...")

    query = """
    MATCH (d:Document)
    WHERE d.source_file ENDS WITH 'shades/Shades.html'
    MATCH (c:Card)-[:HAS_CARD_TYPE]->(ct:CardType {name: 'shades'})
    MERGE (c)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) AS created
    """
    result = session.run(query)
    count = single_value(result, "created")
    logger.info(f"Linked {count} shade cards to Shades overview document")
    return count


def link_documents_to_map(session: Session) -> int:
    """Link Map, Zone, and MapEvent nodes to the Map Document node.

    Fight nodes are NOT linked here - they get their own fight page documents
    via link_documents_to_fights().

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking map nodes to Document...")

    total = 0
    for label in ["Map", "Zone", "MapEvent"]:
        query = f"""
        MATCH (d:Document)
        WHERE d.source_file ENDS WITH 'maps/Map.html'
        MATCH (n:{label})
        MERGE (n)-[:HAS_DOCUMENT]->(d)
        RETURN count(*) as created
        """
        result = session.run(query)
        count = single_value(result, "created")
        total += count
        logger.info(f"  Linked {count} {label} nodes to Map document")

    logger.info(f"Created {total} map-Document relationships")
    return total


def link_documents_to_fights(session: Session) -> int:
    """Link Fight nodes to their individual fight page Document nodes.

    Each Fight has a page_name property (e.g., "Infernoko_Fight") that
    corresponds to the source_file of its Document (e.g., "Infernoko_Fight.html").

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking fight nodes to their Documents...")

    query = """
    MATCH (f:Fight)
    WHERE f.page_name IS NOT NULL
    MATCH (d:Document)
    WHERE d.source_file ENDS WITH ('fights/' + f.page_name + '.html')
    MERGE (f)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) AS created
    """
    result = session.run(query)
    count = single_value(result, "created")
    logger.info(f"Linked {count} Fight nodes to their Documents")
    return count


def link_documents_to_shops(session: Session) -> int:
    """Link Shop nodes to their wiki page Documents.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking Shop nodes to Documents...")

    total = 0
    shop_docs = [
        ("The Woolly Snail", "shops/The_Woolly_Snail.html"),
        ("Charm Merchant", "shops/Charm_Merchant.html"),
    ]
    for shop_name, doc_suffix in shop_docs:
        query = """
        MATCH (d:Document)
        WHERE d.source_file ENDS WITH $doc_suffix
        MATCH (s:Shop {name: $shop_name})
        MERGE (s)-[:HAS_DOCUMENT]->(d)
        RETURN count(*) AS created
        """
        result = session.run(query, doc_suffix=doc_suffix, shop_name=shop_name)
        count = single_value(result, "created")
        total += count

    logger.info(f"Linked {total} Shop-Document relationships")
    return total


def link_documents_to_bling(session: Session) -> int:
    """Link Bling node to the Bling wiki page Document.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking Bling node to Document...")

    query = """
    MATCH (d:Document)
    WHERE d.source_file ENDS WITH 'bling/Bling.html'
    MATCH (b:Bling)
    MERGE (b)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) AS created
    """
    result = session.run(query)
    count = single_value(result, "created")
    logger.info(f"Linked {count} Bling-Document relationships")
    return count


def link_documents_to_bells(session: Session) -> int:
    """Link Bell nodes to their individual bell page Documents.

    Bells with individual wiki pages match their own Document via filename.
    Additionally, all bells link to the summary Bells.html for overview context.

    Args:
        session: Active Neo4j session (caller manages driver lifecycle)

    Returns:
        Number of relationships created
    """
    logger.info("Linking Bell nodes to Documents...")

    total = 0

    # Per-bell Documents (individual pages with detailed effects)
    per_bell_query = """
    MATCH (d:Document)
    MATCH (b:Bell)
    WHERE b.filename IS NOT NULL
      AND d.source_file ENDS WITH ('/' + b.filename)
    MERGE (b)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) AS created
    """
    result = session.run(per_bell_query)
    per_bell_count = single_value(result, "created")
    total += per_bell_count

    # Summary Bells.html (overview table with all bells)
    summary_query = """
    MATCH (d:Document)
    WHERE d.source_file ENDS WITH 'bells/Bells.html'
    MATCH (b:Bell)
    MERGE (b)-[:HAS_DOCUMENT]->(d)
    RETURN count(*) AS created
    """
    result = session.run(summary_query)
    summary_count = single_value(result, "created")
    total += summary_count

    logger.info(
        f"Created {total} Bell-Document relationships ({per_bell_count} per-bell + {summary_count} summary)"
    )
    return total
