"""Dagster assets for retrieval metrics and generation taxonomy."""

from pathlib import Path

from dagster import Config, asset

from wildfrost_rag.services.evaluation.metrics_service import calculate_metrics, save_metrics
from wildfrost_rag.services.evaluation.taxonomy import generate_taxonomy_from_annotations


class RetrievalMetricsConfig(Config):
    """Which k values to compute Hit@k/Precision@k/Recall@k for."""

    k_values: list[int] | None = None


@asset
def retrieval_metrics(retrieval_results: str, config: RetrievalMetricsConfig) -> str:
    """Compute retrieval metrics for the experiment retrieval_results just produced.

    Returns the metrics.json path written.
    """
    experiment_path = Path(retrieval_results)
    metrics_data = calculate_metrics(experiment_path, config.k_values)
    return str(save_metrics(experiment_path, metrics_data))


class GenerationTaxonomyConfig(Config):
    """Which generation experiment's manual annotations to build a taxonomy from."""

    generation_experiment_path: str


@asset
async def generation_taxonomy(config: GenerationTaxonomyConfig) -> None:
    """Generate an axial-codes taxonomy from a generation experiment's manual annotations.

    Not downstream of retrieval_results: generation experiments (LLM answers,
    manually annotated with open codes) aren't produced by any asset in this
    graph yet, so this asset is parameterized entirely by its own config.
    """
    await generate_taxonomy_from_annotations(Path(config.generation_experiment_path))
