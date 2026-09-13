"""Unit tests for get_or_create_run against a real temp SQLite MLflow backend.

Uses a real MLflow store (not mocked) since the whole point of this function
is its search-then-reuse-or-create logic - mocking mlflow.search_runs would
just be testing the mock. A tmp_path-scoped SQLite file keeps this isolated
from the project's real mlflow.db.
"""

from pathlib import Path

import mlflow
import pytest

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
