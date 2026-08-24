"""Prompt registry for WildFrostRAG.

Auto-discovers all VersionedPrompt instances from prompt modules.
Adding a new prompt to any module automatically registers it.
"""

from prompts import system_prompts, text2cypher_prompts, taxonomy_prompts
from prompts.prompt_utils import VersionedPrompt
from utils.logger import logger

PROMPT_REGISTRY: dict[str, VersionedPrompt] = {}

for _module in [system_prompts, text2cypher_prompts, taxonomy_prompts]:
    for _name, _obj in vars(_module).items():
        if isinstance(_obj, VersionedPrompt):
            PROMPT_REGISTRY[_name] = _obj


def get_prompt(prompt_name: str) -> VersionedPrompt:
    """Get a prompt by name from the registry.

    Args:
        prompt_name: Name of the prompt (e.g., "SYSTEM_PROMPT_V1").

    Returns:
        The VersionedPrompt object.
    """
    if prompt_name not in PROMPT_REGISTRY:
        logger.error(f"Unknown prompt: {prompt_name}. Available: {list(PROMPT_REGISTRY.keys())}")
        raise ValueError(f"Unknown prompt: {prompt_name}")
    return PROMPT_REGISTRY[prompt_name]
