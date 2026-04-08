"""Data adapters for experiment results.

This module provides a unified interface for accessing experiment data,
enabling the GUI to work with any experiment type without code changes.

The adapter pattern abstracts away differences in data storage:
- Retrieval experiments: results.json with retrieved chunks
- Generation experiments: results.json with LLM responses
- Text2Cypher experiments: additional cypher_queries.json
- Hybrid experiments: individual retriever scores in chunks

Usage:
    adapter = get_adapter(experiment_path)
    queries = adapter.get_queries()
    adapter.save_annotation(query_id, annotation)
"""

import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logger import logger
from utils.config import settings


@dataclass
class QueryResult:
    """Standardized query result for GUI display."""

    query_id: int
    query: str
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    response: str | None = None
    cypher_query: str | None = None
    cypher_execution_status: str | None = None
    relevance_annotations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentMetadata:
    """Standardized experiment metadata."""

    experiment_type: str  # 'retrieval' or 'generation'
    experiment_id: str
    run_number: int
    timestamp: str
    retriever_type: str | None = None
    chunking: bool | None = None
    description: str = ""
    total_queries: int = 0
    successful_queries: int = 0

    # Generation-specific
    retrieval_reference: str | None = None
    system_prompt_version: str | None = None
    llm_model: str | None = None

    # Text2Cypher-specific
    text2cypher_prompt_version: str | None = None

    # Embedder info (for vector experiments)
    embedder: str | None = None

    # Raw config for any additional fields
    raw_config: dict[str, Any] = field(default_factory=dict)


class ExperimentDataAdapter(ABC):
    """Abstract base class for experiment data adapters.

    GUI components only interact with this interface, making them
    agnostic to the underlying data storage format.
    """

    def __init__(self, experiment_path: Path) -> None:
        """Initialize adapter with experiment directory path.

        Args:
            experiment_path: Path to experiment directory (e.g., outputs/run_1/retrievals/bm25/001)
        """
        self.experiment_path = Path(experiment_path)
        self._config: dict[str, Any] | None = None
        self._results: list[dict[str, Any]] | None = None
        self._annotations: dict[str, Any] | None = None

    @abstractmethod
    def get_experiment_type(self) -> str:
        """Return experiment type ('retrieval' or 'generation')."""
        pass

    def get_metadata(self) -> ExperimentMetadata:
        """Load and return experiment metadata from config.json.

        Returns:
            ExperimentMetadata object with all config info
        """
        config = self._load_config()

        return ExperimentMetadata(
            experiment_type=config.get("experiment_type", "unknown"),
            experiment_id=config.get("retrieval_id")
            or config.get("generation_id", "unknown"),
            run_number=config.get("run_number", 0),
            timestamp=config.get("timestamp", ""),
            retriever_type=config.get("retriever_type"),
            chunking=config.get("chunking"),
            description=config.get("description", ""),
            total_queries=config.get("total_queries", 0),
            successful_queries=config.get("successful_queries", 0),
            retrieval_reference=config.get("retrieval_reference"),
            system_prompt_version=config.get("prompts", {}).get(
                "system_prompt_version"
            ),
            llm_model=config.get("llm_model"),
            text2cypher_prompt_version=config.get("text2cypher_prompt_version"),
            embedder=config.get("embedder"),
            raw_config=config,
        )

    def get_queries(self) -> list[QueryResult]:
        """Load and return all queries with their results.

        Returns:
            List of QueryResult objects
        """
        results = self._load_results()
        annotations = self._load_annotations()

        query_results = []
        for result in results:
            query_id = result.get("query_id", 0)

            # Get annotations for this query
            query_annotations = annotations.get(str(query_id), {})
            relevance_annotations = query_annotations.get("relevance_annotations", [])

            query_result = QueryResult(
                query_id=query_id,
                query=result.get("query", ""),
                retrieved_chunks=result.get("retrieved_chunks", []),
                response=result.get("response"),
                relevance_annotations=relevance_annotations,
            )
            query_results.append(query_result)

        return query_results

    def get_query_by_id(self, query_id: int) -> QueryResult | None:
        """Get a single query by ID."""
        queries = self.get_queries()
        for q in queries:
            if q.query_id == query_id:
                return q
        return None

    def load_annotations(self) -> dict[str, Any]:
        """Load all annotations from annotations.json.

        Returns:
            Dict mapping query_id to annotation data
        """
        return self._load_annotations()

    def save_annotation(self, query_id: int, annotation: dict[str, Any]) -> None:
        """Save annotation for a specific query.

        Args:
            query_id: The query ID to annotate
            annotation: Dict with annotation data (validation, open_coding, axial_coding, etc.)
        """
        annotations = self._load_annotations()

        # Update annotation for this query
        annotations[str(query_id)] = {
            **annotations.get(str(query_id), {}),
            **annotation,
            "updated_at": datetime.now().isoformat(),
        }

        self._save_annotations(annotations)
        logger.debug(f"Saved annotation for query {query_id}")

    def save_chunk_relevance(
        self,
        query_id: int,
        chunk_idx: int,
        is_relevant: bool,
        auto_populated: bool = False,
    ) -> None:
        """Save relevance annotation for a specific chunk.

        Args:
            query_id: The query ID
            chunk_idx: Index of the chunk
            is_relevant: Whether the chunk is relevant
            auto_populated: Whether this annotation was auto-populated (URL matching)
        """
        annotations = self._load_annotations()

        # Ensure query entry exists
        if str(query_id) not in annotations:
            annotations[str(query_id)] = {}

        # Get or create relevance annotations list
        relevance = annotations[str(query_id)].get("relevance_annotations", [])

        # Find and update or add entry for this chunk
        chunk_id = f"chunk_{chunk_idx}"
        found = False
        for ann in relevance:
            if ann.get("chunk_id") == chunk_id:
                ann["is_relevant"] = is_relevant
                ann["updated_at"] = datetime.now().isoformat()
                if auto_populated:
                    ann["auto_populated"] = True
                found = True
                break

        if not found:
            entry = {
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "is_relevant": is_relevant,
                "created_at": datetime.now().isoformat(),
            }
            if auto_populated:
                entry["auto_populated"] = True
            relevance.append(entry)

        annotations[str(query_id)]["relevance_annotations"] = relevance
        annotations[str(query_id)]["updated_at"] = datetime.now().isoformat()

        self._save_annotations(annotations)

    def _load_config(self) -> dict[str, Any]:
        """Load config.json."""
        if self._config is None:
            config_path = self.experiment_path / "config.json"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    self._config = json.load(f)
            else:
                logger.warning(f"Config not found: {config_path}")
                self._config = {}
        return self._config

    def _load_results(self) -> list[dict[str, Any]]:
        """Load results.json."""
        if self._results is None:
            results_path = self.experiment_path / "results.json"
            if results_path.exists():
                with open(results_path, encoding="utf-8") as f:
                    self._results = json.load(f)
            else:
                logger.warning(f"Results not found: {results_path}")
                self._results = []
        return self._results

    def _load_annotations(self) -> dict[str, Any]:
        """Load annotations.json."""
        if self._annotations is None:
            annotations_path = self.experiment_path / "annotations.json"
            if annotations_path.exists():
                with open(annotations_path, encoding="utf-8") as f:
                    self._annotations = json.load(f)
            else:
                self._annotations = {}
        return self._annotations

    def _save_annotations(self, annotations: dict[str, Any]) -> None:
        """Save annotations to annotations.json."""
        annotations_path = self.experiment_path / "annotations.json"
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        self._annotations = annotations


class RetrievalDataAdapter(ExperimentDataAdapter):
    """Adapter for standard retrieval experiments.

    Works with: bm25, vector, fulltext, and hybrid retrievers.
    """

    def get_experiment_type(self) -> str:
        """Return the experiment type identifier."""
        return "retrieval"


class Text2CypherDataAdapter(ExperimentDataAdapter):
    """Adapter for Text2Cypher experiments.

    Extends retrieval adapter with Cypher query information.
    Reads cypher_execution directly from results.json (no separate cypher_queries.json).
    """

    def get_experiment_type(self) -> str:
        """Return the experiment type identifier."""
        return "retrieval"

    def get_queries(self) -> list[QueryResult]:
        """Load queries with Cypher query information from cypher_execution in results."""
        queries = super().get_queries()
        results = self._load_results()

        # Build map from query_id to cypher_execution
        cypher_map: dict[int, dict[str, Any]] = {}
        for result in results:
            ce = result.get("cypher_execution")
            if ce:
                cypher_map[result["query_id"]] = ce

        # Enrich query results with cypher info
        for query in queries:
            ce = cypher_map.get(query.query_id)
            if ce:
                query.cypher_query = ce.get("cypher_query")
                query.cypher_execution_status = ce.get("cypher_execution_status")

        return queries


class GenerationDataAdapter(ExperimentDataAdapter):
    """Adapter for generation experiments.

    Works with LLM generation results that include responses.
    """

    def get_experiment_type(self) -> str:
        """Return the experiment type identifier."""
        return "generation"


class ExperimentRegistry:
    """Registry for discovering and listing experiments.

    Reads from experiments.yaml and provides methods to filter
    and retrieve experiment paths.
    """

    def __init__(self, outputs_dir: Path | None = None) -> None:
        """Initialize registry.

        Args:
            outputs_dir: Path to outputs directory. Defaults to settings.outputs_dir.
        """
        if outputs_dir is None:
            outputs_dir = settings.outputs_dir

        self.outputs_dir = Path(outputs_dir)
        self._registry: dict[str, Any] | None = None

    @property
    def registry(self) -> dict[str, Any]:
        """Load and return the registry data."""
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry

    def _load_registry(self) -> dict[str, Any]:
        """Load experiments.yaml."""
        registry_path = self.outputs_dir / "experiments.yaml"
        if registry_path.exists():
            with open(registry_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {"current_run": 0, "runs": {}}

    def get_current_run(self) -> int:
        """Get current run number."""
        current_run = self.registry.get("current_run", 0)
        if not isinstance(current_run, int):
            logger.error(f"Expected int for current_run, got {type(current_run)}")
            raise TypeError(f"Expected int for current_run, got {type(current_run)}")
        return current_run

    def get_all_runs(self) -> list[int]:
        """Get list of all run numbers."""
        return sorted(self.registry.get("runs", {}).keys())

    def get_experiments(
        self,
        run_num: int | None = None,
        experiment_type: str | None = None,
        retriever_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get experiments matching filters.

        Args:
            run_num: Filter by run number. None for all runs.
            experiment_type: Filter by type ('retrieval' or 'generation').
            retriever_type: Filter by retriever type (bm25, vector, etc.).

        Returns:
            List of experiment info dicts with 'path', 'type', 'id', etc.
        """
        results = []
        runs = self.registry.get("runs", {})

        # Determine which runs to check
        if run_num is not None:
            runs_to_check = {run_num: runs.get(run_num, {})}
        else:
            runs_to_check = runs

        for rn, run_data in runs_to_check.items():
            # Process retrievals
            if experiment_type is None or experiment_type == "retrieval":
                for exp_id, exp_info in run_data.get("retrievals", {}).items():
                    # Filter by retriever type
                    if (
                        retriever_type
                        and exp_info.get("retriever_type") != retriever_type
                    ):
                        continue

                    # Build path
                    retriever_name = exp_id.split("/")[0]
                    exp_num = exp_id.split("/")[1]
                    path = (
                        self.outputs_dir
                        / f"run_{rn}"
                        / "retrievals"
                        / retriever_name
                        / exp_num
                    )

                    results.append(
                        {
                            "path": path,
                            "type": "retrieval",
                            "id": exp_id,
                            "run_number": rn,
                            **exp_info,
                        }
                    )

            # Process generations
            if experiment_type is None or experiment_type == "generation":
                for exp_id, exp_info in run_data.get("generations", {}).items():
                    exp_num = exp_id.split("/")[1] if "/" in exp_id else exp_id
                    path = self.outputs_dir / f"run_{rn}" / "generation" / exp_num

                    results.append(
                        {
                            "path": path,
                            "type": "generation",
                            "id": exp_id,
                            "run_number": rn,
                            **exp_info,
                        }
                    )

        return results

    def get_retrievers(self, run_num: int | None = None) -> list[str]:
        """Get list of unique retriever types."""
        experiments = self.get_experiments(run_num=run_num, experiment_type="retrieval")
        retrievers = set()
        for exp in experiments:
            if exp.get("retriever_type"):
                retrievers.add(exp["retriever_type"])
        return sorted(retrievers)

    def get_embedders(self, run_num: int | None = None) -> list[str]:
        """Get list of unique embedders used."""
        experiments = self.get_experiments(run_num=run_num)
        embedders = set()
        for exp in experiments:
            # Check for embedder in registry info
            embedder = exp.get("embedder")
            if embedder:
                embedders.add(embedder)
            # Also check retriever type for vector variants
            retriever = exp.get("retriever_type", "")
            if retriever.startswith("vector_"):
                embedders.add(retriever.replace("vector_", ""))
        return sorted(embedders)


def get_adapter(experiment_path: Path) -> ExperimentDataAdapter:
    """Factory function to get appropriate adapter for an experiment.

    Automatically detects experiment type from config.json and
    returns the correct adapter class.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        Appropriate ExperimentDataAdapter subclass
    """
    experiment_path = Path(experiment_path)
    config_path = experiment_path / "config.json"

    if not config_path.exists():
        logger.warning(
            f"No config.json found at {experiment_path}, using default RetrievalDataAdapter"
        )
        return RetrievalDataAdapter(experiment_path)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    experiment_type = config.get("experiment_type", "")
    retriever_type = config.get("retriever_type", "")

    # Select adapter based on experiment type and retriever
    if experiment_type == "generation":
        return GenerationDataAdapter(experiment_path)
    elif retriever_type == "text2cypher":
        return Text2CypherDataAdapter(experiment_path)
    else:
        return RetrievalDataAdapter(experiment_path)
