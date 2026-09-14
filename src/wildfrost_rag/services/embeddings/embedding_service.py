"""Embedding generation and vector-index creation for Document nodes.

Safe to re-run after a partial failure: only Documents still missing the
target property are processed, rather than skipping the whole run if any
document already has it.

All Neo4j reads/writes are delegated to VectorRepository - this service only
does provider selection, model loading, and batch orchestration.
"""

import time

import ollama
from neo4j import Driver
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from wildfrost_rag.clients.openai_client import call_openai_embeddings
from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.configs.embedding import EmbedderProviderConfig
from wildfrost_rag.core.exceptions import EmbeddingError
from wildfrost_rag.core.logger import logger
from wildfrost_rag.repositories.vector_store import VectorRepository
from wildfrost_rag.core.embedder_type import EmbedderType

_BATCH_SIZE = 50


class EmbeddingService:
    """Generates embeddings for Document nodes and builds their vector index."""

    def _get_embedder_config(self, embedder: EmbedderType) -> EmbedderProviderConfig:
        """Look up an embedder's configuration from settings."""
        return get_settings().embedding.embedding_configs[embedder]

    def _load_embedding_model(
        self, embedder: EmbedderType, model_name: str
    ) -> SentenceTransformer | None:
        """Load the appropriate embedding model based on provider."""
        logger.info(f"Loading embedding model: {model_name}...")

        if embedder is EmbedderType.HF:
            model = SentenceTransformer(model_name)
            logger.info("HuggingFace model loaded")
            return model

        if embedder is EmbedderType.OPENAI:
            if not get_settings().openai.api_key:
                raise EmbeddingError(provider=embedder.value, reason="OpenAI API key not set")
            return None

        # EmbedderType.GEMMA: accessed via the Ollama API, no local model to load.
        logger.info("Using Ollama for Gemma embeddings")
        return None

    async def _generate_batch_embeddings(
        self,
        embedder: EmbedderType,
        model: SentenceTransformer | None,
        model_name: str,
        texts: list[str],
        async_client: ollama.AsyncClient | None,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if embedder is EmbedderType.HF:
            if not isinstance(model, SentenceTransformer):
                raise EmbeddingError(
                    provider=embedder.value, reason="Missing SentenceTransformer model"
                )
            embeddings = model.encode(texts, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]

        if embedder is EmbedderType.OPENAI:
            return await call_openai_embeddings(texts, model=model_name)

        # EmbedderType.GEMMA
        if async_client is None:
            raise EmbeddingError(provider=embedder.value, reason="Missing ollama.AsyncClient")
        gemma_response = await async_client.embed(model=model_name, input=texts)
        gemma_embeddings: list[list[float]] = gemma_response["embeddings"]
        return gemma_embeddings

    async def add_embeddings(self, driver: Driver, embedder: EmbedderType) -> int:
        """Generate and store embeddings for every Document missing this provider's property.

        Safe to re-run after a partial failure: only documents still missing
        the property are processed, rather than skipping the whole run if
        any document already has it.

        Returns:
            Number of Document nodes updated.
        """
        config = self._get_embedder_config(embedder)
        logger.info(f"Embedding provider: {embedder.value}, model: {config.model}")

        repository = VectorRepository(driver)
        documents = repository.documents_missing_property(config.property_name)
        if not documents:
            logger.info(f"All Documents already have '{config.property_name}' - nothing to do")
            return 0
        logger.info(f"{len(documents)} documents need '{config.property_name}'")

        model = self._load_embedding_model(embedder, config.model)
        async_client = ollama.AsyncClient() if embedder is EmbedderType.GEMMA else None

        total_updated = 0
        with tqdm(total=len(documents), desc="Embedding documents", unit="doc") as pbar:
            for i in range(0, len(documents), _BATCH_SIZE):
                batch = documents[i : i + _BATCH_SIZE]
                batch_texts = [doc.text for doc in batch]
                batch_ids = [doc.element_id for doc in batch]

                batch_start = time.time()
                embeddings = await self._generate_batch_embeddings(
                    embedder, model, config.model, batch_texts, async_client
                )
                batch_time = time.time() - batch_start
                rate = len(batch_texts) / batch_time if batch_time > 0 else float("inf")
                logger.info(
                    f"Batch {i // _BATCH_SIZE + 1}: {len(batch_texts)} docs in "
                    f"{batch_time:.2f}s ({rate:.1f} docs/sec)"
                )

                total_updated += repository.set_document_embeddings(
                    batch_ids, embeddings, config.property_name
                )
                pbar.update(len(batch))

        logger.info(f"Added '{config.property_name}' to {total_updated} documents")
        return total_updated

    def create_vector_index(self, driver: Driver, embedder: EmbedderType) -> None:
        """Create the vector index for the given provider's embedding property."""
        config = self._get_embedder_config(embedder)
        VectorRepository(driver).create_embedding_index(
            property_name=config.property_name,
            index_name=config.index_name,
            dimension=config.dimension,
        )
