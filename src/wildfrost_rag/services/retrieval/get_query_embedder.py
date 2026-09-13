"""Query-time embedding functions for WildFrostRAG retrievers.

Provides a factory that returns the correct embedding function for each
embedder provider (hf, gemma, openai). Each function takes a query string
and returns a list of floats — ready to pass to Neo4j vector search.

Models are cached at module level so they're loaded once per process.
"""

from collections.abc import Callable

import ollama
from openai import APIError, OpenAI
from sentence_transformers import SentenceTransformer

from wildfrost_rag.core.exceptions import EmbeddingError
from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger
from wildfrost_rag.services.embeddings.embedder_type import EmbedderType


class _EmbedderCache:
    """Typed cache for lazily-loaded embedding models/clients."""

    hf_model: SentenceTransformer | None = None
    openai_client: OpenAI | None = None


_cache = _EmbedderCache()


def get_query_embed_fn(embedder: str) -> Callable[[str], list[float]]:
    """Return an embedding function for the given embedder provider.

    Args:
        embedder: Provider name — must be a key in settings.embedding.embedding_configs
                  (e.g. "hf", "gemma", "openai")

    Returns:
        A function: (query: str) -> list[float]
    """
    try:
        embedder_type = EmbedderType(embedder)
    except ValueError as exc:
        available = [member.value for member in EmbedderType]
        raise ValueError(f"Unknown embedder: {embedder}. Available: {available}") from exc

    config = get_settings().embedding.embedding_configs[embedder_type.value]
    model_name = config["model"]

    if embedder_type is EmbedderType.HF:
        return _make_hf_embed_fn(model_name)

    if embedder_type is EmbedderType.GEMMA:
        return _make_ollama_embed_fn(model_name)

    return _make_openai_embed_fn(model_name)


def _make_hf_embed_fn(model_name: str) -> Callable[[str], list[float]]:
    """Build embed function for HuggingFace SentenceTransformer models."""

    def embed(query: str) -> list[float]:
        if _cache.hf_model is None:
            logger.info(f"Loading HF embedding model: {model_name}")
            _cache.hf_model = SentenceTransformer(model_name)
        result: list[float] = _cache.hf_model.encode(query).tolist()
        return result

    return embed


def _make_ollama_embed_fn(model_name: str) -> Callable[[str], list[float]]:
    """Build embed function for Ollama-served models (e.g. Gemma)."""

    def embed(query: str) -> list[float]:
        response = ollama.embed(model=model_name, input=[query])
        embeddings: list[float] = response["embeddings"][0]
        return embeddings

    return embed


def _make_openai_embed_fn(model_name: str) -> Callable[[str], list[float]]:
    """Build embed function for OpenAI embedding models.

    Stays synchronous: embed_fn is a shared sync Callable[[str], list[float]]
    contract used by every retriever, and some call it from inside an
    already-running event loop, where an async client call isn't safe.
    """

    def embed(query: str) -> list[float]:
        if _cache.openai_client is None:
            logger.info(f"Initializing OpenAI client for model: {model_name}")
            settings = get_settings()
            if settings.openai.api_key is None:
                raise ValueError("OPENAI_API_KEY not configured")
            _cache.openai_client = OpenAI(api_key=settings.openai.api_key.get_secret_value())
        try:
            response = _cache.openai_client.embeddings.create(input=[query], model=model_name)
        except APIError as e:
            raise EmbeddingError(provider="openai", reason=str(e)) from e
        return response.data[0].embedding

    return embed
