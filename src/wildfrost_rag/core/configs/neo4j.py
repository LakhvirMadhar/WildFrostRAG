"""Neo4j connection settings - credentials loaded from the environment."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jSettings(BaseSettings):
    """Neo4j connection configuration."""

    uri: SecretStr
    username: str
    password: SecretStr

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
