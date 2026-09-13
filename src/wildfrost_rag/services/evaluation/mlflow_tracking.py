"""Shared MLflow setup: one experiment, one run per retrieval experiment.

retrieval_results and retrieval_metrics are separate Dagster assets (and
separate CLI invocations), but both need to log into the *same* MLflow run
for a given retriever experiment (e.g. "bm25/007") - config from one,
metrics from the other. get_or_create_run finds the existing run by name
instead of creating a second one.
"""

import mlflow
from mlflow import ActiveRun

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wildfrost_rag_retrieval"


def get_or_create_run(run_name: str) -> ActiveRun:
    """Return a context manager for the MLflow run named run_name, reusing it if it exists.

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
    if existing_runs:
        return mlflow.start_run(run_id=existing_runs[0].info.run_id)
    return mlflow.start_run(run_name=run_name)
