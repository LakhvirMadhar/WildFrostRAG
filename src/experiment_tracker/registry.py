"""Experiment registry for tracking all experiments in a central YAML file.

This provides MLflow-like functionality:
- Register experiments as they're created
- Query experiments by type, parameters, etc.
- Resolve shortcuts like "latest/bm25" or "current" run
"""

import yaml
from pathlib import Path
from typing import Any

from models.experiment import GenerationRecord, RetrievalRecord
from utils.config import get_settings
from utils.logger import logger


class ExperimentRegistry:
    """Centralized registry for all experiments.

    The registry is stored in outputs/experiments.yaml and tracks:
    - All retrieval experiments
    - All generation experiments
    - Current run number
    - Experiment metadata for quick lookup
    """

    def __init__(self, registry_path: Path | None = None) -> None:
        """Initialize the experiment registry.

        Args:
            registry_path: Path to registry file (default: outputs/experiments.yaml)
        """
        self.registry_path = registry_path or (
            get_settings().paths.outputs_dir / "experiments.yaml"
        )
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Create registry file if it doesn't exist."""
        if not self.registry_path.exists():
            initial_data = {"current_run": 1, "runs": {}}
            self._save_registry(initial_data)
            logger.info(f"Created experiment registry at {self.registry_path}")

    def _load_registry(self) -> dict[str, Any]:
        """Load registry from YAML file."""
        with open(self.registry_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Save registry to YAML file."""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_current_run(self) -> int:
        """Get the current run number."""
        registry = self._load_registry()
        current_run = registry.get("current_run", 1)
        if not isinstance(current_run, int):
            logger.error(f"Expected int for current_run, got {type(current_run)}")
            raise TypeError(f"Expected int for current_run, got {type(current_run)}")
        return current_run

    def increment_run(self) -> int:
        """Increment to next run number.

        Returns:
            New run number
        """
        registry = self._load_registry()
        current: int = registry.get("current_run", 1)
        new_run = current + 1
        registry["current_run"] = new_run
        self._save_registry(registry)
        logger.info(f"Incremented run number: {current} -> {new_run}")
        return new_run

    def register_retrieval(
        self,
        run_num: int,
        retriever_type: str,
        experiment_id: str,
        config: dict[str, Any],
    ) -> None:
        """Register a retrieval experiment in the registry.

        Args:
            run_num: Run number
            retriever_type: Type of retriever
            experiment_id: Experiment ID (e.g., "001")
            config: Full experiment config
        """
        registry = self._load_registry()

        # Ensure run exists
        if "runs" not in registry:
            registry["runs"] = {}
        if run_num not in registry["runs"]:
            registry["runs"][run_num] = {"retrievals": {}, "generations": {}}

        # Register retrieval
        retrieval_ref = f"{retriever_type}/{experiment_id}"
        registry["runs"][run_num]["retrievals"][retrieval_ref] = {
            "timestamp": config.get("timestamp"),
            "retriever_type": retriever_type,
            "chunking": config.get("chunking"),
            "description": config.get("description", ""),
            "total_queries": config.get("total_queries"),
            "successful_queries": config.get("successful_queries"),
        }

        self._save_registry(registry)
        logger.info(f"Registered retrieval: run_{run_num}/{retrieval_ref}")

    def register_generation(self, run_num: int, generation_id: str, config: dict[str, Any]) -> None:
        """Register a generation experiment in the registry.

        Args:
            run_num: Run number
            generation_id: Generation experiment ID (e.g., "001")
            config: Full experiment config
        """
        registry = self._load_registry()

        # Ensure run exists
        if "runs" not in registry:
            registry["runs"] = {}
        if run_num not in registry["runs"]:
            registry["runs"][run_num] = {"retrievals": {}, "generations": {}}

        # Register generation
        gen_ref = f"gen/{generation_id}"
        registry["runs"][run_num]["generations"][gen_ref] = {
            "timestamp": config.get("timestamp"),
            "retrieval_reference": config.get("retrieval_reference"),
            "system_prompt_version": config.get("prompts", {}).get("system_prompt_version"),
            "description": config.get("description", ""),
            "total_queries": config.get("total_queries"),
            "successful_queries": config.get("successful_queries"),
        }

        self._save_registry(registry)
        logger.info(f"Registered generation: run_{run_num}/{gen_ref}")

    def resolve_retrieval_reference(self, run_num: int, reference: str) -> str | None:
        """Resolve a retrieval reference, handling shortcuts.

        Args:
            run_num: Run number
            reference: Reference string (e.g., "bm25/001", "latest/bm25")

        Returns:
            Resolved reference (e.g., "bm25/001") or None if not found
        """
        # Check for "latest" shortcut
        if reference.startswith("latest/"):
            retriever_type = reference.split("/", 1)[1]
            return self.get_latest_retrieval(run_num, retriever_type)

        # Otherwise return as-is (exact reference)
        return reference

    def get_latest_retrieval(self, run_num: int, retriever_type: str) -> str | None:
        """Get the latest retrieval experiment for a given retriever type.

        Args:
            run_num: Run number
            retriever_type: Type of retriever (e.g., "bm25", "vector")

        Returns:
            Latest retrieval reference (e.g., "bm25/003") or None
        """
        registry = self._load_registry()

        if "runs" not in registry or run_num not in registry["runs"]:
            return None

        retrievals = registry["runs"][run_num].get("retrievals", {})

        # Filter by retriever type and sort by timestamp
        matching = [
            (ref, data) for ref, data in retrievals.items() if ref.startswith(f"{retriever_type}/")
        ]

        if not matching:
            return None

        # Sort by timestamp (most recent first)
        matching.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)
        result = matching[0][0]
        if not isinstance(result, str):
            logger.error(f"Expected str for retrieval reference, got {type(result)}")
            raise TypeError(f"Expected str for retrieval reference, got {type(result)}")
        return result

    def list_retrievals(
        self, run_num: int, retriever_type: str | None = None
    ) -> list[RetrievalRecord]:
        """List all retrieval experiments for a run.

        Args:
            run_num: Run number
            retriever_type: Optional filter by retriever type

        Returns:
            List of typed retrieval experiment records
        """
        registry = self._load_registry()

        if "runs" not in registry or run_num not in registry["runs"]:
            return []

        retrievals = registry["runs"][run_num].get("retrievals", {})

        # Filter by type if specified
        if retriever_type:
            retrievals = {
                ref: data
                for ref, data in retrievals.items()
                if ref.startswith(f"{retriever_type}/")
            }

        result = [RetrievalRecord.from_registry_data(ref, data) for ref, data in retrievals.items()]

        # Sort by timestamp (most recent first)
        result.sort(key=lambda r: r.timestamp or "", reverse=True)
        return result

    def list_generations(self, run_num: int) -> list[GenerationRecord]:
        """List all generation experiments for a run.

        Args:
            run_num: Run number

        Returns:
            List of typed generation experiment records
        """
        registry = self._load_registry()

        if "runs" not in registry or run_num not in registry["runs"]:
            return []

        generations = registry["runs"][run_num].get("generations", {})

        result = [
            GenerationRecord.from_registry_data(ref, data) for ref, data in generations.items()
        ]

        # Sort by timestamp (most recent first)
        result.sort(key=lambda r: r.timestamp or "", reverse=True)
        return result

    def search_experiments(
        self,
        run_num: int | None = None,
        experiment_type: str | None = None,
        **filters: str | int | float | bool | None,
    ) -> list[RetrievalRecord | GenerationRecord]:
        """Search for experiments matching criteria.

        Args:
            run_num: Optional filter by run number
            experiment_type: Optional filter by type ("retrieval" or "generation")
            **filters: Additional filters (e.g., retriever_type="bm25", chunking=False)

        Returns:
            List of matching typed experiment records
        """
        registry = self._load_registry()
        results: list[RetrievalRecord | GenerationRecord] = []

        runs_to_search = [run_num] if run_num else registry.get("runs", {}).keys()

        for rnum in runs_to_search:
            if rnum not in registry.get("runs", {}):
                continue

            run_data = registry["runs"][rnum]

            # Search retrievals
            if experiment_type in [None, "retrieval"]:
                for ref, data in run_data.get("retrievals", {}).items():
                    if self._matches_filters(data, filters):
                        results.append(RetrievalRecord.from_registry_data(ref, data, rnum))

            # Search generations
            if experiment_type in [None, "generation"]:
                for ref, data in run_data.get("generations", {}).items():
                    if self._matches_filters(data, filters):
                        results.append(GenerationRecord.from_registry_data(ref, data, rnum))

        return results

    def _matches_filters(
        self, data: dict[str, Any], filters: dict[str, str | int | float | bool | None]
    ) -> bool:
        """Check if experiment data matches all filters."""
        for key, value in filters.items():
            if data.get(key) != value:
                return False
        return True
