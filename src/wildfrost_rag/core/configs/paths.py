"""Directory paths for data, outputs, and schemas.

data_dir/outputs_dir and their subdirectories are computed properties
derived from project_root, not independently-defaulted fields - a
Pydantic field default referencing another field is baked in at class
definition time, so it would silently ignore an overridden
PATH_PROJECT_ROOT instead of tracking it.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PathSettings(BaseSettings):
    """Directory paths for data, outputs, and schemas."""

    # core/configs/paths.py -> core/configs -> core -> wildfrost_rag -> src -> repo root
    project_root: Path = Path(__file__).parent.parent.parent.parent.parent

    model_config = SettingsConfigDict(
        env_prefix="PATH_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def data_dir(self) -> Path:
        """Directory for scraped/processed data."""
        return self.project_root / "data"

    @property
    def outputs_dir(self) -> Path:
        """Directory for experiment outputs."""
        return self.project_root / "outputs"

    @property
    def structured_outputs_dir(self) -> Path:
        """Directory for structured card data by CardType."""
        return self.data_dir / "structured_outputs"

    @property
    def raw_htmls_dir(self) -> Path:
        """Directory for raw scraped HTML files."""
        return self.data_dir / "raw_htmls"

    @property
    def schemas_dir(self) -> Path:
        """Directory for JSON card type schemas."""
        return self.data_dir / "schemas"
