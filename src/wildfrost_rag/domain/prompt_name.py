"""Single source of truth for the names prompts are registered under in MLflow."""

from enum import StrEnum


class PromptName(StrEnum):
    """Every prompt this project registers, by its MLflow Prompt Registry name."""

    SYSTEM_PROMPT = "system_prompt"
    RAG_PROMPT = "rag_prompt"
    TEXT2CYPHER_PROMPT = "text2cypher_prompt"
    TAXONOMY_SYSTEM_PROMPT = "taxonomy_system_prompt"
    TAXONOMY_USER_PROMPT = "taxonomy_user_prompt"
