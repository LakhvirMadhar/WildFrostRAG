"""Mirror this project's prompt templates into MLflow's Prompt Registry.

The Python constants in system_prompts.py/text2cypher_prompts.py/
taxonomy_prompts.py are the reviewable, git-diffable source of truth. Run
sync_local_prompts_to_mlflow() after editing one of them so MLflow gets a
new registered version to point to; unchanged templates are left alone
rather than re-registered as a no-op version.
"""

import mlflow
import mlflow.genai as genai

from wildfrost_rag.core.config import get_settings
from wildfrost_rag.core.logger import logger
from wildfrost_rag.domain.prompt_name import PromptName
from wildfrost_rag.prompts import system_prompts, taxonomy_prompts, text2cypher_prompts

CURRENT_TEMPLATES: dict[PromptName, str] = {
    PromptName.SYSTEM_PROMPT: system_prompts.SYSTEM_PROMPT,
    PromptName.RAG_PROMPT: system_prompts.RAG_PROMPT,
    PromptName.TEXT2CYPHER_PROMPT: text2cypher_prompts.TEXT2CYPHER_PROMPT,
    PromptName.TAXONOMY_SYSTEM_PROMPT: taxonomy_prompts.TAXONOMY_SYSTEM_PROMPT,
    PromptName.TAXONOMY_USER_PROMPT: taxonomy_prompts.TAXONOMY_USER_PROMPT,
}


def sync_local_prompts_to_mlflow() -> None:
    """Register a new MLflow prompt version for each template that changed."""
    mlflow.set_tracking_uri(get_settings().mlflow.tracking_uri)

    for name, template in CURRENT_TEMPLATES.items():
        latest = genai.load_prompt(name, allow_missing=True)
        if latest is not None and latest.template == template:
            logger.info(f"{name}: unchanged (still v{latest.version})")
            continue

        registered = genai.register_prompt(name=name, template=template)
        logger.info(f"{name}: registered v{registered.version}")


if __name__ == "__main__":
    sync_local_prompts_to_mlflow()
