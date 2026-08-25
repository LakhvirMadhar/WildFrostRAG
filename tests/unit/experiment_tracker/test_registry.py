"""End-to-end round-trip tests for ExperimentRegistry via ExperimentRepository.

No Neo4j, no mocks - a real temp directory backs the registry file.
"""

from pathlib import Path

from wildfrost_rag.experiment_tracker.registry import ExperimentRegistry
from wildfrost_rag.models.experiment import GenerationRecord, RetrievalRecord
from wildfrost_rag.models.experiment_config import (
    EmbeddingConfig,
    GenerationConfig,
    PromptVersions,
    QueryStats,
    RetrievalConfig,
)


def _make_retrieval_config(
    retrieval_id: str = "bm25/001", timestamp: str = "2026-01-01T00:00:00"
) -> RetrievalConfig:
    return RetrievalConfig(
        retrieval_id=retrieval_id,
        run_number=1,
        timestamp=timestamp,
        retriever_type="bm25",
        chunking=False,
        description="Baseline BM25",
        query_stats=QueryStats(total=10, successful=10, failed=0),
        embedding=EmbeddingConfig(model="all-MiniLM-L6-v2"),
    )


def _make_generation_config() -> GenerationConfig:
    return GenerationConfig(
        generation_id="gen/001",
        run_number=1,
        timestamp="2026-01-01T01:00:00",
        retrieval_reference="bm25/001",
        llm_model="gpt-4.1-nano",
        prompts=PromptVersions(system_prompt_version="SYSTEM_PROMPT_V1"),
        query_stats=QueryStats(total=10, successful=9, failed=1),
    )


def test_register_and_list_retrieval(tmp_path: Path) -> None:
    """A registered retrieval experiment is returned by list_retrievals."""
    registry = ExperimentRegistry(registry_path=tmp_path / "experiments.yaml")

    registry.register_retrieval(1, "bm25", "001", _make_retrieval_config())

    retrievals = registry.list_retrievals(1)
    assert len(retrievals) == 1
    assert isinstance(retrievals[0], RetrievalRecord)
    assert retrievals[0].reference == "bm25/001"
    assert retrievals[0].retriever_type == "bm25"
    assert retrievals[0].total_queries == 10


def test_register_and_list_generation(tmp_path: Path) -> None:
    """A registered generation experiment is returned by list_generations."""
    registry = ExperimentRegistry(registry_path=tmp_path / "experiments.yaml")

    registry.register_generation(1, "001", _make_generation_config())

    generations = registry.list_generations(1)
    assert len(generations) == 1
    assert isinstance(generations[0], GenerationRecord)
    assert generations[0].system_prompt_version == "SYSTEM_PROMPT_V1"
    assert generations[0].successful_queries == 9


def test_resolve_latest_shortcut(tmp_path: Path) -> None:
    """The latest/<type> shortcut resolves to the most recently registered match."""
    registry = ExperimentRegistry(registry_path=tmp_path / "experiments.yaml")
    registry.register_retrieval(1, "bm25", "001", _make_retrieval_config())
    registry.register_retrieval(
        1,
        "bm25",
        "002",
        _make_retrieval_config(retrieval_id="bm25/002", timestamp="2026-01-02T00:00:00"),
    )

    resolved = registry.resolve_retrieval_reference(1, "latest/bm25")

    assert resolved == "bm25/002"


def test_search_experiments_across_types(tmp_path: Path) -> None:
    """search_experiments() with no filters returns both retrieval and generation records."""
    registry = ExperimentRegistry(registry_path=tmp_path / "experiments.yaml")
    registry.register_retrieval(1, "bm25", "001", _make_retrieval_config())
    registry.register_generation(1, "001", _make_generation_config())

    results = registry.search_experiments()

    assert len(results) == 2
    types = {r.type for r in results}
    assert types == {"retrieval", "generation"}


def test_increment_run_persists_across_instances(tmp_path: Path) -> None:
    """A run increment is visible to a new ExperimentRegistry pointed at the same file."""
    registry_path = tmp_path / "experiments.yaml"
    registry = ExperimentRegistry(registry_path=registry_path)
    assert registry.get_current_run() == 1

    registry.increment_run()

    reopened = ExperimentRegistry(registry_path=registry_path)
    assert reopened.get_current_run() == 2
