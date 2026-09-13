"""Unit tests for get_or_create_run against a real temp SQLite MLflow backend.

Uses a real MLflow store (not mocked) since the whole point of this function
is its search-then-reuse-or-create logic - mocking mlflow.search_runs would
just be testing the mock. A tmp_path-scoped SQLite file keeps this isolated
from the project's real mlflow.db.
"""

import re
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
from git import InvalidGitRepositoryError, Repo
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

from wildfrost_rag.domain.experiment_type import ExperimentType
from wildfrost_rag.services.evaluation import mlflow_tracking


@pytest.fixture(autouse=True)
def _isolated_mlflow_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point MLflow at a throwaway SQLite file for this test only."""
    monkeypatch.setattr(mlflow_tracking, "TRACKING_URI", f"sqlite:///{tmp_path}/test_mlflow.db")


def test_get_or_create_run_creates_a_new_run_when_none_exists() -> None:
    """A fresh run name gets a brand-new MLflow run with a real run_id."""
    with mlflow_tracking.get_or_create_run("bm25/001") as run:
        run_id = run.info.run_id

    assert run_id


def test_get_or_create_run_reuses_the_existing_run_by_name() -> None:
    """Calling with the same run name twice resumes the same run, not a new one."""
    with mlflow_tracking.get_or_create_run("bm25/002") as first_run:
        first_run_id = first_run.info.run_id
        mlflow.log_param("k", 10)

    with mlflow_tracking.get_or_create_run("bm25/002") as second_run:
        second_run_id = second_run.info.run_id
        mlflow.log_metric("avg_hit_at_1", 0.9)

    assert first_run_id == second_run_id


def test_get_or_create_run_does_not_reuse_a_differently_named_run() -> None:
    """Two distinct run names get two distinct runs."""
    with mlflow_tracking.get_or_create_run("bm25/003") as run_a:
        run_a_id = run_a.info.run_id

    with mlflow_tracking.get_or_create_run("vector_hf/001") as run_b:
        run_b_id = run_b.info.run_id

    assert run_a_id != run_b_id


def test_get_or_create_run_tags_the_real_git_commit_sha() -> None:
    """The run is tagged with the actual current commit.

    MLflow's own autodetection doesn't fire for how this project is really
    invoked (verified: dagster asset materialize resolves sys.argv[0] to the
    installed dagster console script, not a repo file, so MLflow finds no
    enclosing git repo to detect).
    """
    with mlflow_tracking.get_or_create_run("bm25/004") as run:
        run_id = run.info.run_id

    tagged_sha = mlflow.get_run(run_id).data.tags.get(MLFLOW_GIT_COMMIT)
    actual_sha = Repo(Path(__file__).parent, search_parent_directories=True).head.commit.hexsha

    assert tagged_sha == actual_sha
    assert re.fullmatch(r"[0-9a-f]{40}", tagged_sha)


def test_get_or_create_run_prefers_the_git_commit_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GIT_COMMIT env var wins over inspecting the repo directly.

    This is what makes it work in a container: a Docker build can bake the
    SHA in as an env var, since a built image commonly has neither a .git
    directory nor the git binary at runtime.
    """
    monkeypatch.setenv("GIT_COMMIT", "baked-in-build-sha")

    with mlflow_tracking.get_or_create_run("bm25/005") as run:
        run_id = run.info.run_id

    assert mlflow.get_run(run_id).data.tags.get(MLFLOW_GIT_COMMIT) == "baked-in-build-sha"


def test_get_or_create_run_skips_the_tag_when_not_in_a_git_repo() -> None:
    """No enclosing git repo (e.g. a container with .git excluded) -> no tag, not a crash."""
    with patch(
        "wildfrost_rag.services.evaluation.mlflow_tracking.Repo",
        side_effect=InvalidGitRepositoryError,
    ):
        with mlflow_tracking.get_or_create_run("bm25/006") as run:
            run_id = run.info.run_id

    assert MLFLOW_GIT_COMMIT not in mlflow.get_run(run_id).data.tags


def _log_retrieval_run(run_name: str, run_number: int, retriever_type: str) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_number", run_number)
        mlflow.log_params({"retriever_type": retriever_type})


def _log_generation_run(run_name: str, run_number: int) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_number", run_number)
        mlflow.log_params({"retrieval_reference": "bm25/001"})


def test_search_experiments_filters_by_retriever_type() -> None:
    """retriever_type='bm25' matches only the bm25 run, not the vector_hf one."""
    mlflow.set_tracking_uri(mlflow_tracking.TRACKING_URI)
    mlflow.set_experiment(mlflow_tracking.EXPERIMENT_NAME)
    _log_retrieval_run("bm25/010", run_number=1, retriever_type="bm25")
    _log_retrieval_run("vector_hf/010", run_number=1, retriever_type="vector_hf")

    runs = mlflow_tracking.search_experiments(retriever_type="bm25")

    assert runs["tags.mlflow.runName"].tolist() == ["bm25/010"]


def test_search_experiments_distinguishes_retrieval_from_generation() -> None:
    """Generation runs never log retriever_type; retrieval runs always do."""
    mlflow.set_tracking_uri(mlflow_tracking.TRACKING_URI)
    mlflow.set_experiment(mlflow_tracking.EXPERIMENT_NAME)
    _log_retrieval_run("bm25/011", run_number=1, retriever_type="bm25")
    _log_generation_run("gen/011", run_number=1)

    retrieval_runs = mlflow_tracking.search_experiments(experiment_type=ExperimentType.RETRIEVAL)
    generation_runs = mlflow_tracking.search_experiments(experiment_type=ExperimentType.GENERATION)

    assert retrieval_runs["tags.mlflow.runName"].tolist() == ["bm25/011"]
    assert generation_runs["tags.mlflow.runName"].tolist() == ["gen/011"]


def test_search_experiments_filters_by_run_number() -> None:
    """run_number=1 matches only runs logged under this project's run 1."""
    mlflow.set_tracking_uri(mlflow_tracking.TRACKING_URI)
    mlflow.set_experiment(mlflow_tracking.EXPERIMENT_NAME)
    _log_retrieval_run("bm25/012", run_number=1, retriever_type="bm25")
    _log_retrieval_run("bm25/013", run_number=2, retriever_type="bm25")

    runs = mlflow_tracking.search_experiments(run_number=1)

    assert runs["tags.mlflow.runName"].tolist() == ["bm25/012"]
