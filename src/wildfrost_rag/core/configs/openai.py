"""OpenAI API settings and per-use-case LLM hyperparameters."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """OpenAI API and per-use-case LLM configuration."""

    # --- Setting: credential, loaded from the environment ---
    api_key: SecretStr | None = None

    # --- Hyperparameters: tunable, affect generation quality/determinism ---
    model_name: str = "gpt-4.1-nano"  # Default model for generation
    temperature: float = 0.0  # For deterministic responses
    seed: int = 42  # Random seed for reproducibility

    text2cypher_temperature: float = 1.0  # gpt-5-mini only supports temperature=1
    text2cypher_model: str = "gpt-5-mini"

    taxonomy_temperature: float = 0.3  # Slightly creative for categorization
    taxonomy_model: str = "gpt-4o-mini"

    # --- Operational tuning: concurrency, not a research variable ---
    llm_semaphore_limit: int = 50  # Max concurrent LLM calls

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
