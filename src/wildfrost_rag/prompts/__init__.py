"""Prompt loading for WildFrostRAG.

Template text is authored in this package's modules and version-controlled
in MLflow's Prompt Registry (see sync_prompts.py). load_prompt resolves a
"name" or "name:version" reference to a specific registered PromptVersion.
"""

from wildfrost_rag.prompts.prompt_utils import load_prompt

__all__ = ["load_prompt"]
