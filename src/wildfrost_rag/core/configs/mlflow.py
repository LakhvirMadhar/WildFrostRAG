"""MLflow tracking backend settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MlflowSettings(BaseSettings):
    """MLflow tracking backend configuration."""

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "wildfrost_rag"

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
