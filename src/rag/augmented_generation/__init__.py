"""__init__.py for augmented_generation package.

This module exports the OpenAI client functions for generation.
"""

from .openai_client import (
    call_openai_api,
    call_openai_api_structured,
    call_openai_embeddings,
    generate_zero_shot,
    generate_rag,
)

__all__ = [
    "call_openai_api",
    "call_openai_api_structured",
    "call_openai_embeddings",
    "generate_zero_shot",
    "generate_rag",
]
