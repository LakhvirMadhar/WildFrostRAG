#!/usr/bin/env python3
"""Add embedding properties to existing Document nodes in Neo4j.

This script allows adding embeddings from multiple providers to the same
Document nodes, enabling multi-embedder testing without data duplication.

Usage:
    poetry run python -m scripts.add_embeddings --embedder hf
    poetry run python -m scripts.add_embeddings --embedder openai
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
import time

from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from neo4j import GraphDatabase, Driver
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import ollama
from utils.config import get_settings
from utils.logger import logger
from neo4j_kg.vector_store import create_embedding_index


@dataclass
class EmbedderConfig:
    """Configuration for an embedding provider."""

    name: str
    property_name: str
    index_name: str
    model_name: str
    dimension: int


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add embedding properties to existing Document nodes"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        required=True,
        help="Embedder name (e.g., 'hf', 'openai'). Must be defined in config.py",
    )
    return parser.parse_args()


def get_embedder_config(embedder_name: str) -> EmbedderConfig:
    """Validate embedder name and return its configuration."""
    settings = get_settings()
    if embedder_name not in settings.embedding.embedding_configs:
        logger.error(f"Unknown embedder: {embedder_name}")
        logger.error(f"Available embedders: {list(settings.embedding.embedding_configs.keys())}")
        sys.exit(1)

    config = settings.embedding.embedding_configs[embedder_name]
    return EmbedderConfig(
        name=embedder_name,
        property_name=config["property_name"],
        index_name=config["index_name"],
        model_name=config["model"],
        dimension=config["dimension"],
    )


def check_existing_embeddings(driver: Driver, property_name: str) -> int:
    """Check if embeddings already exist for the given property."""
    with driver.session() as session:
        query = f"""
        MATCH (d:Document)
        WHERE d.{property_name} IS NOT NULL
        RETURN count(d) as count
        """
        result = session.run(query)
        record = result.single()
        return record["count"] if record else 0


def get_documents(driver: Driver) -> list[tuple[str, str]]:
    """Fetch all Document nodes from Neo4j."""
    with driver.session() as session:
        query = """
        MATCH (d:Document)
        RETURN d.text as text, elementId(d) as element_id
        ORDER BY element_id
        """
        results = session.run(query)
        return [(record["text"], record["element_id"]) for record in results]


def load_embedding_model(
    embedder_name: str, model_name: str
) -> SentenceTransformer | OpenAI | None:
    """Load the appropriate embedding model based on provider."""
    logger.info(f"Loading embedding model: {model_name}...")

    if embedder_name == "hf":
        model = SentenceTransformer(model_name)
        logger.info("HuggingFace model loaded")
        return model

    if embedder_name == "openai":
        settings = get_settings()
        if not settings.openai.api_key:
            logger.error("OpenAI API key not found in settings")
            sys.exit(1)
        client = OpenAI(api_key=settings.openai.api_key.get_secret_value())
        logger.info("OpenAI client initialized")
        return client

    if embedder_name == "gemma":
        logger.info("Using Ollama for Gemma embeddings")
        return None  # Ollama is accessed via API

    logger.error(f"Unsupported embedder type: {embedder_name}")
    sys.exit(1)


async def generate_batch_embeddings(
    embedder_name: str,
    model: SentenceTransformer | OpenAI | None,
    model_name: str,
    texts: list[str],
    async_client: ollama.AsyncClient | None = None,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    if embedder_name == "hf":
        if not isinstance(model, SentenceTransformer):
            raise TypeError("HF embedder requires SentenceTransformer model")
        embeddings = model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    if embedder_name == "openai":
        if not isinstance(model, OpenAI):
            raise TypeError("OpenAI embedder requires OpenAI client")
        response = model.embeddings.create(input=texts, model=model_name)
        return [data.embedding for data in response.data]

    if embedder_name == "gemma":
        if async_client is None:
            raise ValueError("Gemma embedder requires ollama.AsyncClient")
        gemma_response = await async_client.embed(model=model_name, input=texts)
        embeddings_result: list[list[float]] = gemma_response["embeddings"]
        return embeddings_result

    raise ValueError(f"Unknown embedder: {embedder_name}")


def update_documents(
    driver: Driver,
    element_ids: list[str],
    embeddings: list[list[float]],
    property_name: str,
) -> int:
    """Update Document nodes with embeddings."""
    updated = 0
    with driver.session() as session:
        for element_id, embedding in zip(element_ids, embeddings, strict=False):
            query = f"""
            MATCH (d:Document)
            WHERE elementId(d) = $element_id
            SET d.{property_name} = $embedding
            """
            session.run(query, element_id=element_id, embedding=embedding)
            updated += 1
    return updated


def log_summary(config: EmbedderConfig, total_updated: int) -> None:
    """Log completion summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Embedding addition complete!")
    logger.info("=" * 60)
    logger.info(f"Provider: {config.name}")
    logger.info(f"Documents updated: {total_updated}")
    logger.info(f"Property: {config.property_name}")
    logger.info(f"Index: {config.index_name}")
    logger.info("")
    logger.info("You can now use this embedder in batch retrieval:")
    logger.info("  - type: vector")
    logger.info(f'    embedder: "{config.name}"')


async def main() -> None:
    """Add embeddings from a specified provider to Document nodes."""
    args = parse_args()
    config = get_embedder_config(args.embedder)

    logger.info(f"Adding embeddings for provider: {config.name}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Property name: {config.property_name}")
    logger.info(f"Index name: {config.index_name}")
    logger.info(f"Dimension: {config.dimension}")

    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j.uri.get_secret_value(),
        auth=(settings.neo4j.username, settings.neo4j.password.get_secret_value()),
    )

    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully")

        # Check for existing embeddings
        existing_count = check_existing_embeddings(driver, config.property_name)
        if existing_count > 0:
            logger.info(
                f"Property '{config.property_name}' already exists on {existing_count} documents"
            )
            logger.info("Skipping embedding generation (already exists)")
            return

        # Get documents
        documents = get_documents(driver)
        if not documents:
            logger.error("No Document nodes found in Neo4j")
            sys.exit(1)
        logger.info(f"Found {len(documents)} documents to process")

        # Load model
        model = load_embedding_model(config.name, config.model_name)
        async_client = ollama.AsyncClient() if config.name == "gemma" else None

        # Process in batches
        logger.info("Generating embeddings and updating nodes...")
        batch_size = 50
        total_updated = 0

        with tqdm(total=len(documents), desc="Embedding documents", unit="doc") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_texts = [doc[0] for doc in batch]
                batch_ids = [doc[1] for doc in batch]

                batch_start = time.time()
                embeddings = await generate_batch_embeddings(
                    config.name, model, config.model_name, batch_texts, async_client
                )
                batch_time = time.time() - batch_start

                logger.info(
                    f"Batch {i // batch_size + 1}: {len(batch_texts)} docs in "
                    f"{batch_time:.2f}s ({len(batch_texts) / batch_time:.1f} docs/sec)"
                )

                updated = update_documents(driver, batch_ids, embeddings, config.property_name)
                total_updated += updated
                pbar.update(len(batch))

        logger.info(f"Successfully added {config.property_name} to {total_updated} documents")

        # Create vector index
        logger.info(f"Creating vector index: {config.index_name}...")
        create_embedding_index(
            property_name=config.property_name,
            index_name=config.index_name,
            dimension=config.dimension,
        )
        logger.info(f"Vector index '{config.index_name}' created")

        log_summary(config, total_updated)

    finally:
        driver.close()


if __name__ == "__main__":
    asyncio.run(main())
