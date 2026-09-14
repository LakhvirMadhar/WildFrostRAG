"""Web scraping target configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ScrapingSettings(BaseSettings):
    """Web scraping target configuration."""

    wildfrost_wiki_base_url: str = "https://wildfrostwiki.com"
    max_concurrent_requests: int = 50

    model_config = SettingsConfigDict(
        env_prefix="SCRAPING_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
