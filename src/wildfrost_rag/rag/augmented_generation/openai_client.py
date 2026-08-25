"""Centralized async OpenAI client for WildFrostRAG.

This module provides:
- Low-level API calls (call_openai_api, call_openai_api_structured, call_openai_embeddings)
- High-level generation functions (generate_zero_shot, generate_rag)

All functions are async with built-in rate limiting via semaphore.
"""

import asyncio
from openai import AsyncOpenAI, APIError, AuthenticationError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from wildfrost_rag.core.exceptions import (
    EmbeddingError,
    LLMAuthenticationError,
    LLMError,
    LLMMalformedResponseError,
    LLMRateLimitError,
)
from wildfrost_rag.utils.config import get_settings
from wildfrost_rag.utils.logger import logger
from wildfrost_rag.prompts.prompt_utils import VersionedPrompt, format_prompt_tuple


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
        if settings.openai.api_key is None:
            raise ValueError("OPENAI_API_KEY not configured")
        _client = AsyncOpenAI(api_key=settings.openai.api_key.get_secret_value())
    return _client


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the singleton semaphore for rate limiting."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(get_settings().openai.llm_semaphore_limit)
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
        model: Model name (defaults to settings.openai.model_name)
        temperature: Temperature (defaults to settings.openai.temperature)
        seed: Random seed (defaults to settings.openai.seed)

    Returns:
        The generated response text
    """
    async with _get_semaphore():
        client = _get_client()
        settings = get_settings()
        resolved_model = model or settings.openai.model_name
        try:
            response = await client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature if temperature is not None else settings.openai.temperature,
                seed=seed if seed is not None else settings.openai.seed,
            )
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded in call_openai_api: {e}")
            raise LLMRateLimitError(str(e), model=resolved_model) from e
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed in call_openai_api: {e}")
            raise LLMAuthenticationError(str(e), model=resolved_model) from e
        except APIError as e:
            logger.error(f"OpenAI API error in call_openai_api: {e}")
            raise LLMError(str(e), model=resolved_model) from e

        content = response.choices[0].message.content
        if content is None:
            raise LLMMalformedResponseError(
                "OpenAI returned empty response content", model=resolved_model
            )
        return content


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
        model: Model name (defaults to settings.openai.model_name)
        temperature: Temperature (defaults to settings.openai.temperature)
        seed: Random seed (defaults to settings.openai.seed)

    Returns:
        Parsed Pydantic model instance
    """
    async with _get_semaphore():
        client = _get_client()
        settings = get_settings()
        resolved_model = model or settings.openai.model_name
        try:
            response = await client.beta.chat.completions.parse(
                model=resolved_model,
                messages=messages,
                response_format=response_model,
                temperature=temperature if temperature is not None else settings.openai.temperature,
                seed=seed if seed is not None else settings.openai.seed,
            )
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded in call_openai_api_structured: {e}")
            raise LLMRateLimitError(str(e), model=resolved_model) from e
        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed in call_openai_api_structured: {e}")
            raise LLMAuthenticationError(str(e), model=resolved_model) from e
        except APIError as e:
            logger.error(f"OpenAI API error in call_openai_api_structured: {e}")
            raise LLMError(str(e), model=resolved_model) from e

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise LLMMalformedResponseError(
                "OpenAI returned empty parsed response", model=resolved_model
            )
        return parsed


async def call_openai_embeddings(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """Async embeddings with rate limiting.

    Args:
        texts: List of texts to embed
        model: Embedding model name (defaults to
            settings.embedding.embedding_configs["openai"]["model"])

    Returns:
        List of embedding vectors
    """
    async with _get_semaphore():
        client = _get_client()
        resolved_model = model or get_settings().embedding.embedding_configs["openai"]["model"]
        try:
            response = await client.embeddings.create(input=texts, model=resolved_model)
            return [item.embedding for item in response.data]
        except APIError as e:
            logger.error(f"OpenAI API error in call_openai_embeddings: {e}")
            raise EmbeddingError(provider="openai", reason=str(e)) from e


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
