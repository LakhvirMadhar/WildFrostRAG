"""Query-time embedding functions for WildFrostRAG retrievers.

Provides a factory that returns the correct embedding function for each
embedder provider (hf, gemma, openai). Each function takes a query string
and returns a list of floats — ready to pass to Neo4j vector search.

Models are cached at module level so they're loaded once per process.
"""

from collections.abc import Callable

import ollama
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from utils.config import settings
from utils.logger import logger


class _EmbedderCache:
    """Typed cache for lazily-loaded embedding models/clients."""

    hf_model: SentenceTransformer | None = None
    openai_client: OpenAI | None = None


_cache = _EmbedderCache()


def get_query_embed_fn(embedder: str) -> Callable[[str], list[float]]:
    """Return an embedding function for the given embedder provider.

    Args:
        embedder: Provider name — must be a key in settings.embedding_configs
                  (e.g. "hf", "gemma", "openai")

    Returns:
        A function: (query: str) -> list[float]
    """
    if embedder not in settings.embedding_configs:
        raise ValueError(
            f"Unknown embedder: {embedder}. Available: {list(settings.embedding_configs.keys())}"
        )

    config = settings.embedding_configs[embedder]
    model_name = config["model"]

    if embedder == "hf":
        return _make_hf_embed_fn(model_name)

    if embedder == "gemma":
        return _make_ollama_embed_fn(model_name)

    if embedder == "openai":
        return _make_openai_embed_fn(model_name)

    raise ValueError(f"No query embed function implemented for embedder: {embedder}")


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
    """Build embed function for OpenAI embedding models."""

    def embed(query: str) -> list[float]:
        if _cache.openai_client is None:
            logger.info(f"Initializing OpenAI client for model: {model_name}")
            if settings.openai_api_key is None:
                raise ValueError("OPENAI_API_KEY not configured")
            _cache.openai_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        response = _cache.openai_client.embeddings.create(input=[query], model=model_name)
        return response.data[0].embedding

    return embed
