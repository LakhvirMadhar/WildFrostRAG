"""
__init__.py for augmented_generation package.

This module exports the main classes from the generation pipeline.
"""

from .call_llm_generation import LLMGenerator, GenerationPipeline

__all__ = ['LLMGenerator', 'GenerationPipeline']