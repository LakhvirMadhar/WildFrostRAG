"""Shared MLflow setup: one experiment, one run per retrieval or generation experiment.

retrieval_results and retrieval_metrics are separate Dagster assets (and
separate CLI invocations), but both need to log into the *same* MLflow run
for a given retriever experiment (e.g. "bm25/007") - config from one,
metrics from the other. get_or_create_run finds the existing run by name
instead of creating a second one. Generation experiments (e.g. "gen/007")
share the same MLflow experiment, distinguished by run name prefix.
"""

import os
from pathlib import Path

import mlflow
import pandas as pd
from git import InvalidGitRepositoryError, NoSuchPathError, Repo
from mlflow import ActiveRun
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

from wildfrost_rag.domain.experiment_type import ExperimentType

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wildfrost_rag"


def _get_git_commit_sha() -> str | None:
    """Get the current git commit SHA, or None if unavailable.

    MLflow can tag this automatically, but only when the process's entry
    point (sys.argv[0]) is a script inside the git repo. Real usage here
    goes through `dagster asset materialize`/`dagster dev`, where
    sys.argv[0] resolves to the installed dagster console script instead -
    verified this actually fails silently (no tag) for that invocation, so
    the SHA is captured explicitly instead of relying on autodetection.

    Checks GIT_COMMIT first so a container build can bake the SHA in as an
    env var (no .git directory or git binary needed at runtime - a built
    image commonly has neither) before falling back to inspecting the repo
    directly for local/dev use, anchored at this file's own location so it
    works regardless of the caller's current working directory.
    """
    env_sha = os.environ.get("GIT_COMMIT")
    if env_sha:
        return env_sha

    try:
        repo = Repo(Path(__file__).parent, search_parent_directories=True)
        return repo.head.commit.hexsha
    except (InvalidGitRepositoryError, NoSuchPathError, ValueError):
        return None


def get_or_create_run(run_name: str) -> ActiveRun:
    """Return a context manager for the MLflow run named run_name, reusing it if it exists.

    Tags the run with the current git commit SHA (see _get_git_commit_sha for
    why this can't rely on MLflow's own autodetection).

    Args:
        run_name: Unique identifier for this retrieval experiment, e.g. "bm25/007"
            (matches the retriever_type/experiment_id already used for the
            on-disk experiment directory name).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    existing_runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format="list",
    )
    run = (
        mlflow.start_run(run_id=existing_runs[0].info.run_id)
        if existing_runs
        else mlflow.start_run(run_name=run_name)
    )

    commit_sha = _get_git_commit_sha()
    if commit_sha:
        mlflow.set_tag(MLFLOW_GIT_COMMIT, commit_sha)

    return run


def search_experiments(
    experiment_type: ExperimentType | None = None,
    retriever_type: str | None = None,
    chunking: bool | None = None,
    run_number: int | None = None,
) -> pd.DataFrame:
    """Search logged retrieval/generation runs.

    experiment_type filters on whether a retriever_type param was logged,
    since only retrieval runs ever log one - see the retriever_type IS
    [NOT] NULL clauses below for why, rather than a dedicated type tag.

    Args:
        experiment_type: ExperimentType.RETRIEVAL or .GENERATION, or None for both
        retriever_type: e.g. "bm25" (retrieval runs only)
        chunking: whether chunking was enabled (retrieval runs only)
        run_number: this project's own run grouping (e.g. 1)

    Returns:
        Matching runs, one row per run
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    clauses = []
    if experiment_type is ExperimentType.GENERATION:
        # Generation runs never log a retriever_type param; retrieval runs always do.
        # MLflow's filter DSL has no "NOT LIKE" (verified: only LIKE/ILIKE/=/!=/IS [NOT] NULL),
        # so this checks actual data shape instead of the run-name prefix convention.
        clauses.append("params.retriever_type IS NULL")
    elif experiment_type is ExperimentType.RETRIEVAL:
        clauses.append("params.retriever_type IS NOT NULL")
    if retriever_type is not None:
        clauses.append(f"params.retriever_type = '{retriever_type}'")
    if chunking is not None:
        clauses.append(f"params.chunking = '{chunking}'")
    if run_number is not None:
        clauses.append(f"tags.run_number = '{run_number}'")

    return mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME], filter_string=" and ".join(clauses)
    )
