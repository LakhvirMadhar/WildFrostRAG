#!/usr/bin/env python3
"""
Add embedding properties to existing Document nodes in Neo4j.

This script allows adding embeddings from multiple providers to the same
Document nodes, enabling multi-embedder testing without data duplication.

Usage:
    poetry run python -m scripts.add_embeddings --embedder hf
    poetry run python -m scripts.add_embeddings --embedder openai
"""

import argparse
import asyncio
from pathlib import Path
import sys
import time
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import ollama
from src.utils.config import settings
from src.utils.logger import logger
from src.neo4j_kg.vector_store import create_embedding_index


async def main():
    parser = argparse.ArgumentParser(
        description="Add embedding properties to existing Document nodes"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        required=True,
        help="Embedder name (e.g., 'hf', 'openai'). Must be defined in config.py"
    )
    args = parser.parse_args()

    # Get embedder config
    if args.embedder not in settings.embedding_configs:
        logger.error(f"Unknown embedder: {args.embedder}")
        logger.error(f"Available embedders: {list(settings.embedding_configs.keys())}")
        exit(1)

    embedder_config = settings.embedding_configs[args.embedder]
    property_name = embedder_config["property_name"]
    index_name = embedder_config["index_name"]
    model_name = embedder_config["model"]
    dimension = embedder_config["dimension"]

    logger.info(f"Adding embeddings for provider: {args.embedder}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Property name: {property_name}")
    logger.info(f"Index name: {index_name}")
    logger.info(f"Dimension: {dimension}")

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        settings.neo4j_uri.get_secret_value(),
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value())
    )

    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully")

        with driver.session() as session:
            # Check if property already exists
            check_query = f"""
            MATCH (d:Document)
            WHERE d.{property_name} IS NOT NULL
            RETURN count(d) as count
            """

            result = session.run(check_query)
            record = result.single()
            existing_count = record["count"] if record else 0

            if existing_count > 0:
                logger.info(f"Property '{property_name}' already exists on {existing_count} documents")
                logger.info("Skipping embedding generation (already exists)")
                driver.close()
                return

            # Get all Document nodes
            get_docs_query = """
            MATCH (d:Document)
            RETURN d.text as text, elementId(d) as element_id
            ORDER BY element_id
            """

            results = session.run(get_docs_query)
            documents = [(record["text"], record["element_id"]) for record in results]

            if not documents:
                logger.error("No Document nodes found in Neo4j")
                exit(1)

            logger.info(f"Found {len(documents)} documents to process")

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}...")
        if args.embedder == "hf":
            # HuggingFace sentence transformers
            embedding_model = SentenceTransformer(model_name)
            logger.info("HuggingFace model loaded")
        elif args.embedder == "openai":
            # OpenAI embeddings
            if not settings.openai_api_key:
                logger.error("OpenAI API key not found in settings")
                exit(1)
            embedding_model = OpenAI(api_key=settings.openai_api_key.get_secret_value())
            logger.info("OpenAI client initialized")
        elif args.embedder == "gemma":
            # Ollama embeddings (no model loading needed)
            embedding_model = None  # Ollama is accessed via API
            logger.info("Using Ollama for Gemma embeddings")
        else:
            logger.error(f"Unsupported embedder type: {args.embedder}")
            exit(1)

        # Create async client for Gemma if needed
        async_client = None
        if args.embedder == "gemma":
            async_client = ollama.AsyncClient()

        # Generate embeddings and update nodes
        logger.info("Generating embeddings and updating nodes...")
        batch_size = 50
        total_updated = 0

        # Progress bar for all documents
        with tqdm(total=len(documents), desc="Embedding documents", unit="doc") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_texts = [doc[0] for doc in batch]
                batch_ids = [doc[1] for doc in batch]

                # Generate embeddings
                batch_start = time.time()
                if args.embedder == "hf":
                    embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
                    embeddings_list = [emb.tolist() for emb in embeddings]
                elif args.embedder == "openai":
                    response = embedding_model.embeddings.create(
                        input=batch_texts,
                        model=model_name
                    )
                    embeddings_list = [data.embedding for data in response.data]
                elif args.embedder == "gemma":
                    response = await async_client.embed(model=model_name, input=batch_texts)
                    embeddings_list = response['embeddings']

                batch_time = time.time() - batch_start
                logger.info(f"Batch {i//batch_size + 1}: {len(batch_texts)} docs in {batch_time:.2f}s ({len(batch_texts)/batch_time:.1f} docs/sec)")

                # Update nodes
                with driver.session() as session:
                    for element_id, embedding in zip(batch_ids, embeddings_list):
                        update_query = f"""
                        MATCH (d:Document)
                        WHERE elementId(d) = $element_id
                        SET d.{property_name} = $embedding
                        """
                        session.run(update_query, element_id=element_id, embedding=embedding)
                        total_updated += 1
                        pbar.update(1)

        logger.info(f"✓ Successfully added {property_name} to {total_updated} documents")

        # Create vector index
        logger.info(f"Creating vector index: {index_name}...")
        create_embedding_index(
            property_name=property_name,
            index_name=index_name,
            dimension=dimension
        )
        logger.info(f"✓ Vector index '{index_name}' created")

        logger.info("")
        logger.info("="*60)
        logger.info("Embedding addition complete!")
        logger.info("="*60)
        logger.info(f"Provider: {args.embedder}")
        logger.info(f"Documents updated: {total_updated}")
        logger.info(f"Property: {property_name}")
        logger.info(f"Index: {index_name}")
        logger.info("")
        logger.info("You can now use this embedder in batch retrieval:")
        logger.info(f"  - type: vector")
        logger.info(f"    embedder: \"{args.embedder}\"")

    finally:
        driver.close()


if __name__ == "__main__":
    asyncio.run(main())
