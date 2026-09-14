"""Web scraping target configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ScrapingSettings(BaseSettings):
    """Web scraping target configuration."""

    wildfrost_wiki_base_url: str = "https://wildfrostwiki.com"
    max_concurrent_requests: int = 50

    # Special pages for tribe enrichment
    companions_page_url: str = "https://wildfrostwiki.com/Companions"
    items_page_url: str = "https://wildfrostwiki.com/Items"

    model_config = SettingsConfigDict(
        env_prefix="SCRAPING_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
