"""Loading prompts tracked in MLflow's Prompt Registry.

Prompt template text is authored in this package's modules (system_prompts.py,
text2cypher_prompts.py, taxonomy_prompts.py) and version-controlled in git;
sync_prompts.py mirrors each one into MLflow so it becomes a real, queryable
PromptVersion that a run can be linked to. load_prompt resolves a reference
against that registry, not against the Python constants directly.
"""

from typing import Any

import mlflow
import mlflow.genai as genai
from mlflow.entities.model_registry.prompt_version import PromptVersion

from wildfrost_rag.core.config import get_settings


def render_text_prompt(prompt: PromptVersion, **kwargs: Any) -> str:  # noqa: ANN401
    """Render a text (non-chat) prompt version, asserting it isn't chat-style.

    PromptVersion.format() also supports chat-style templates (a list of
    role/content messages), which none of this project's prompts are -
    the assert catches a prompt being re-registered as chat-style without
    updating the call site that expects a plain string.
    """
    rendered = prompt.format(**kwargs)
    assert isinstance(rendered, str), f"{prompt.name} must be a text prompt, not chat-style"
    return rendered


def load_prompt(reference: str) -> PromptVersion:
    """Load a registered prompt version by "name" or "name:version" reference.

    "name" alone resolves to that prompt's latest registered version.

    Args:
        reference: e.g. "system_prompt" (latest) or "system_prompt:2" (pinned).

    Returns:
        The matching PromptVersion, with .format(**kwargs) to render it and
        .uri to record which exact version was used.
    """
    mlflow.set_tracking_uri(get_settings().mlflow.tracking_uri)
    name, _, version = reference.partition(":")
    prompt: PromptVersion = genai.load_prompt(name, version=int(version) if version else None)
    return prompt
