"""
Prompt utilities for versioned prompt management.

This module provides utilities for managing versioned prompts across the codebase.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VersionedPrompt:
    """
    A versioned prompt template with metadata.

    This class standardizes prompt versioning across the codebase, making it easy
    to track which prompt version was used for each experiment.

    Attributes:
        prompt_version_name: Name matching the variable name (e.g., "TEXT2CYPHER_PROMPT_V1")
        prompt_tuple: Tuple with template string and parameter names
                     Format: (template_string, param1, param2, ...)

    Example:
        >>> TEXT2CYPHER_PROMPT_V1 = VersionedPrompt(
        ...     prompt_version_name="TEXT2CYPHER_PROMPT_V1",
        ...     prompt_tuple=(
        ...         "Convert {query} using {schema}",
        ...         "query",
        ...         "schema"
        ...     )
        ... )
    """
    prompt_version_name: str
    prompt_tuple: Tuple[str, ...]


def format_prompt_tuple(prompt_tuple: Tuple[str, ...], **kwargs) -> str:
    """
    Format a prompt tuple with provided keyword arguments.

    Prompt tuples follow the pattern: (template_string, param1, param2, ...)
    where template_string contains {param1}, {param2}, etc. placeholders.

    Args:
        prompt_tuple: Tuple where first element is template string,
                     remaining elements are expected parameter names
        **kwargs: Keyword arguments to fill in the template

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If required parameters are missing

    Example:
        >>> PROMPT = ("Hello {name}, you are {age} years old", "name", "age")
        >>> format_prompt_tuple(PROMPT, name="Alice", age=30)
        "Hello Alice, you are 30 years old"
    """
    template = prompt_tuple[0]
    expected_params = prompt_tuple[1:]

    # Validate that all expected parameters are provided
    missing_params = [param for param in expected_params if param not in kwargs]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")

    # Format the template with provided kwargs
    return template.format(**kwargs)
