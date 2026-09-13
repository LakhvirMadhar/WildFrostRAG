"""Unit tests for coupling_analysis.

compute_fan_in_fan_out runs the real grimp graph builder against this
project's own real package rather than mocking grimp, since the whole
point of this function is grimp's real graph-building behavior. It's
computed once per test session (module-scoped fixture) since building
the graph is real, non-trivial work - two separate tests each rebuilding
it from scratch would double the cost for no benefit.
"""

import pytest

from wildfrost_rag.services.architecture.coupling_analysis import (
    compute_fan_in_fan_out,
    format_top_n,
)


def test_format_top_n_orders_by_count_descending() -> None:
    """The highest counts appear first, regardless of input dict order."""
    counts = {"a": 3, "b": 10, "c": 1}

    result = format_top_n(counts, top=3)

    lines = result.splitlines()
    assert "b" in lines[0]
    assert "a" in lines[1]
    assert "c" in lines[2]


def test_format_top_n_respects_top_limit() -> None:
    """Only the requested number of rows are returned."""
    counts = {f"module_{i}": i for i in range(10)}

    result = format_top_n(counts, top=3)

    assert len(result.splitlines()) == 3


@pytest.fixture(scope="module")
def real_fan_in_fan_out() -> tuple[dict[str, int], dict[str, int]]:
    """Build the real wildfrost_rag import graph once, shared by both tests below."""
    return compute_fan_in_fan_out("wildfrost_rag")


def test_compute_fan_in_fan_out_finds_real_known_high_fan_in_module(
    real_fan_in_fan_out: tuple[dict[str, int], dict[str, int]],
) -> None:
    """core.logger is imported by more of this project's own modules than almost anything else.

    A real, known fact about this codebase - if this ever stops being true,
    either the codebase's structure changed a lot, or something's wrong
    with how the graph is being built.
    """
    fan_in, _ = real_fan_in_fan_out

    assert "wildfrost_rag.core.logger" in fan_in
    assert fan_in["wildfrost_rag.core.logger"] > 30


def test_compute_fan_in_fan_out_excludes_third_party_imports(
    real_fan_in_fan_out: tuple[dict[str, int], dict[str, int]],
) -> None:
    """Third-party packages this project imports never show up as graph nodes.

    Confirms grimp only scans wildfrost_rag's own modules, the property
    that makes it a better fit than pydeps for an internal-coupling
    question (see docs/pydeps_lessons/04_grimp_a_better_tool_for_the_numbers.md).
    """
    fan_in, fan_out = real_fan_in_fan_out

    all_modules = set(fan_in) | set(fan_out)
    assert not any(m == "mlflow" or m.startswith("mlflow.") for m in all_modules)
    assert not any(m == "pandas" or m.startswith("pandas.") for m in all_modules)
    assert not any(m == "neo4j" or m.startswith("neo4j.") for m in all_modules)
