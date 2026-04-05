"""
Query-time embedding functions for WildFrostRAG retrievers.

Provides a factory that returns the correct embedding function for each
embedder provider (hf, gemma, openai). Each function takes a query string
and returns a list of floats — ready to pass to Neo4j vector search.

Models are cached at module level so they're loaded once per process.
"""

from typing import Callable

import ollama
from sentence_transformers import SentenceTransformer

from utils.config import settings
from utils.logger import logger

# Module-level cache: embedder name -> loaded model/client
_model_cache: dict[str, object] = {}


def get_query_embed_fn(embedder: str) -> Callable[[str], list[float]]:
    """
    Return an embedding function for the given embedder provider.

    Args:
        embedder: Provider name — must be a key in settings.embedding_configs
                  (e.g. "hf", "gemma", "openai")

    Returns:
        A function: (query: str) -> list[float]
    """
    if embedder not in settings.embedding_configs:
        raise ValueError(
            f"Unknown embedder: {embedder}. "
            f"Available: {list(settings.embedding_configs.keys())}"
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
        if "hf" not in _model_cache:
            logger.info(f"Loading HF embedding model: {model_name}")
            _model_cache["hf"] = SentenceTransformer(model_name)
        model = _model_cache["hf"]
        return model.encode(query).tolist()
    return embed


def _make_ollama_embed_fn(model_name: str) -> Callable[[str], list[float]]:
    """Build embed function for Ollama-served models (e.g. Gemma)."""
    def embed(query: str) -> list[float]:
        response = ollama.embed(model=model_name, input=[query])
        return response["embeddings"][0]
    return embed


def _make_openai_embed_fn(model_name: str) -> Callable[[str], list[float]]:
    """Build embed function for OpenAI embedding models."""
    def embed(query: str) -> list[float]:
        if "openai" not in _model_cache:
            from openai import OpenAI
            logger.info(f"Initializing OpenAI client for model: {model_name}")
            _model_cache["openai"] = OpenAI(
                api_key=settings.openai_api_key.get_secret_value()
            )
        client = _model_cache["openai"]
        response = client.embeddings.create(input=[query], model=model_name)
        return response.data[0].embedding
    return embed
