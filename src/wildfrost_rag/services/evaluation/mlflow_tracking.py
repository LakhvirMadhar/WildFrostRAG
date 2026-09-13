"""Shared MLflow setup: one experiment, one run per retrieval experiment.

retrieval_results and retrieval_metrics are separate Dagster assets (and
separate CLI invocations), but both need to log into the *same* MLflow run
for a given retriever experiment (e.g. "bm25/007") - config from one,
metrics from the other. get_or_create_run finds the existing run by name
instead of creating a second one.
"""

import os
from pathlib import Path

import mlflow
from git import InvalidGitRepositoryError, NoSuchPathError, Repo
from mlflow import ActiveRun
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wildfrost_rag_retrieval"


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
