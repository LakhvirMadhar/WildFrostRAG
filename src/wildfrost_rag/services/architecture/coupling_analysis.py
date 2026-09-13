"""Fan-in/fan-out analysis using the same import graph import-linter checks.

Fan-in (how many modules import this one) and fan-out (how many modules
this one imports) are two different signals: high fan-in on a
foundational module (core/, domain/) is expected and healthy - it means
the rest of the codebase correctly depends on one shared, stable base.
High fan-out on a single file is the more actionable one - it means that
file has to know about a large number of other things to do its job,
which is a real, quantified version of "this file might be doing too
much."

Uses grimp (the graph library import-linter's own layer/cycle checks are
built on) rather than a general-purpose visualizer like pydeps, because
grimp only counts imports of this project's own modules - a file
importing five third-party libraries isn't an SRP signal, a file
importing five different internal service modules might be. See
docs/pydeps_lessons/04_grimp_a_better_tool_for_the_numbers.md for the
full reasoning and how this compares to pydeps' numbers.
"""

import grimp


def compute_fan_in_fan_out(package: str) -> tuple[dict[str, int], dict[str, int]]:
    """Build the import graph for `package` and count direct fan-in/fan-out per module.

    Args:
        package: Dotted package name to analyze (e.g. "wildfrost_rag").

    Returns:
        (fan_in, fan_out) - each a dict of module name -> count of direct
        imports of/by modules within the same graph.
    """
    graph = grimp.build_graph(package)
    modules = graph.modules

    fan_in = {m: len(graph.find_modules_that_directly_import(m)) for m in modules}
    fan_out = {m: len(graph.find_modules_directly_imported_by(m)) for m in modules}
    return fan_in, fan_out


def format_top_n(counts: dict[str, int], top: int) -> str:
    """Render the top N modules by count as an aligned text table."""
    ranked = sorted(counts.items(), key=lambda item: -item[1])[:top]
    return "\n".join(f"  {count:3d}  {module}" for module, count in ranked)
