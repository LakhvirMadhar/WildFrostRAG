"""Centralized async OpenAI client for WildFrostRAG.

This module provides:
- Low-level API calls (call_openai_api, call_openai_api_structured, call_openai_embeddings)
- High-level generation functions (generate_zero_shot, generate_rag)

All functions are async with built-in rate limiting via semaphore.
"""

import asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from utils.config import get_settings
from utils.logger import logger
from prompts.prompt_utils import VersionedPrompt, format_prompt_tuple


# =============================================================================
# Lazy-initialized client and semaphore
# =============================================================================

_client: AsyncOpenAI | None = None
_semaphore: asyncio.Semaphore | None = None


def _get_client() -> AsyncOpenAI:
    """Get or create the singleton AsyncOpenAI client."""
    global _client
    if _client is None:
        settings = get_settings()
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY not configured")
        _client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    return _client


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the singleton semaphore for rate limiting."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(get_settings().llm_semaphore_limit)
    return _semaphore


# =============================================================================
# Low-level API calls
# =============================================================================


async def call_openai_api(
    messages: list[ChatCompletionMessageParam],
    model: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> str:
    """Async chat completion with rate limiting.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Model name (defaults to settings.openai_model_name)
        temperature: Temperature (defaults to settings.openai_temperature)
        seed: Random seed (defaults to settings.openai_seed)

    Returns:
        The generated response text
    """
    async with _get_semaphore():
        client = _get_client()
        settings = get_settings()
        try:
            response = await client.chat.completions.create(
                model=model or settings.openai_model_name,
                messages=messages,
                temperature=temperature if temperature is not None else settings.openai_temperature,
                seed=seed if seed is not None else settings.openai_seed,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("OpenAI returned empty response content")
            return content
        except Exception as e:
            logger.error(f"Error in call_openai_api: {e}")
            raise


async def call_openai_api_structured[T: BaseModel](
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    model: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> T:
    """Async chat with Pydantic structured output and rate limiting.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        response_model: Pydantic model class for the response
        model: Model name (defaults to settings.openai_model_name)
        temperature: Temperature (defaults to settings.openai_temperature)
        seed: Random seed (defaults to settings.openai_seed)

    Returns:
        Parsed Pydantic model instance
    """
    async with _get_semaphore():
        client = _get_client()
        settings = get_settings()
        try:
            response = await client.beta.chat.completions.parse(
                model=model or settings.openai_model_name,
                messages=messages,
                response_format=response_model,
                temperature=temperature if temperature is not None else settings.openai_temperature,
                seed=seed if seed is not None else settings.openai_seed,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("OpenAI returned empty parsed response")
            return parsed
        except Exception as e:
            logger.error(f"Error in call_openai_api_structured: {e}")
            raise


async def call_openai_embeddings(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """Async embeddings with rate limiting.

    Args:
        texts: List of texts to embed
        model: Embedding model name (defaults to settings.embedding_configs["openai"]["model"])

    Returns:
        List of embedding vectors
    """
    async with _get_semaphore():
        client = _get_client()
        try:
            response = await client.embeddings.create(
                input=texts,
                model=model or get_settings().embedding_configs["openai"]["model"],
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error in call_openai_embeddings: {e}")
            raise


# =============================================================================
# High-level generation functions
# =============================================================================


async def generate_zero_shot(query: str, system_prompt: VersionedPrompt) -> str:
    """Generate a zero-shot response (no context).

    Args:
        query: The user query
        system_prompt: VersionedPrompt containing the system prompt

    Returns:
        The generated response text
    """
    return await call_openai_api(
        messages=[
            {"role": "system", "content": system_prompt.prompt_tuple[0]},
            {"role": "user", "content": query},
        ]
    )


async def generate_rag(
    query: str,
    context: str,
    system_prompt: VersionedPrompt,
    rag_prompt: VersionedPrompt,
) -> str:
    """Generate a RAG response using provided context.

    Args:
        query: The user query
        context: The retrieved context (concatenated chunks)
        system_prompt: VersionedPrompt containing the system prompt
        rag_prompt: VersionedPrompt for formatting the user message with context

    Returns:
        The generated response text
    """
    user_message = format_prompt_tuple(rag_prompt.prompt_tuple, query=query, context=context)
    return await call_openai_api(
        messages=[
            {"role": "system", "content": system_prompt.prompt_tuple[0]},
            {"role": "user", "content": user_message},
        ]
    )
