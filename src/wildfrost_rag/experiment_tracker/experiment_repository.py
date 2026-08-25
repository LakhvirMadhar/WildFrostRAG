"""File-system persistence for the experiment registry YAML file."""

from pathlib import Path
from typing import Any

import yaml

from wildfrost_rag.utils.logger import logger


class ExperimentRepository:
    """Owns all raw read/write access to the experiment registry YAML file.

    Takes the registry file path via the constructor - dependency injection,
    matching the pattern used elsewhere in this codebase (driver/config
    injected, never self-constructed internally).
    """

    def __init__(self, registry_path: Path) -> None:
        """Initialize the repository for a given registry file path.

        Args:
            registry_path: Path to the registry YAML file
        """
        self.registry_path = registry_path
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        """Create the registry file with an empty structure if it doesn't exist."""
        if not self.registry_path.exists():
            initial_data: dict[str, Any] = {"current_run": 1, "runs": {}}
            self.save(initial_data)
            logger.info(f"Created experiment registry at {self.registry_path}")

    def load(self) -> dict[str, Any]:
        """Load the registry contents from the YAML file."""
        with open(self.registry_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def save(self, data: dict[str, Any]) -> None:
        """Save the registry contents to the YAML file."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
