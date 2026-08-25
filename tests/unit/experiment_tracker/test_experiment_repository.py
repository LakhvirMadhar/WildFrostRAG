"""Tests for ExperimentRepository - raw YAML file I/O, no Neo4j needed."""

from pathlib import Path

from wildfrost_rag.experiment_tracker.experiment_repository import ExperimentRepository


def test_creates_registry_file_if_missing(tmp_path: Path) -> None:
    """Constructing a repository against a missing path creates it with an empty structure."""
    registry_path = tmp_path / "experiments.yaml"
    assert not registry_path.exists()

    repo = ExperimentRepository(registry_path)

    assert registry_path.exists()
    data = repo.load()
    assert data == {"current_run": 1, "runs": {}}


def test_does_not_overwrite_existing_registry_file(tmp_path: Path) -> None:
    """An already-existing registry file is left untouched on construction."""
    registry_path = tmp_path / "experiments.yaml"
    repo = ExperimentRepository(registry_path)
    repo.save({"current_run": 5, "runs": {1: {"retrievals": {}, "generations": {}}}})

    repo_again = ExperimentRepository(registry_path)

    assert repo_again.load()["current_run"] == 5


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    """Data saved to the repository is returned unchanged by a subsequent load."""
    registry_path = tmp_path / "experiments.yaml"
    repo = ExperimentRepository(registry_path)

    payload = {
        "current_run": 2,
        "runs": {
            1: {
                "retrievals": {"bm25/001": {"timestamp": "2026-01-01T00:00:00"}},
                "generations": {},
            }
        },
    }
    repo.save(payload)

    assert repo.load() == payload


def test_two_repositories_share_the_same_file(tmp_path: Path) -> None:
    """A second repository instance pointed at the same path sees the first's writes."""
    registry_path = tmp_path / "experiments.yaml"
    repo_a = ExperimentRepository(registry_path)
    repo_b = ExperimentRepository(registry_path)

    repo_a.save({"current_run": 7, "runs": {}})

    assert repo_b.load()["current_run"] == 7
