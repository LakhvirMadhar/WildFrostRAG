"""Single source of truth for the experiment-type identifiers."""

from enum import StrEnum


class ExperimentType(StrEnum):
    """Every kind of experiment tracked in MLflow."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"
