#!/usr/bin/env python3
"""Check first-party package coupling under src/ - no third-party scanning.

Parses only the .py files under src/ with the ast module (no imports are
executed, no third-party or stdlib modules are ever resolved/traversed),
extracts each file's import statements, keeps only the ones that point at
another top-level package under src/, and reports:
- an ASCII layered diagram (most-composed packages at the top, infra floor
  at the bottom)
- which packages are imported by the most others (fan-in)
- the package -> package edge list
- any import cycles between top-level packages
- optionally, a rendered graph via the graphviz package (--svg)

Usage:
    poetry run python -m scripts.check_coupling
    poetry run python -m scripts.check_coupling --svg docs/migration_plan/coupling.svg
"""

import argparse
import ast
import sys
from pathlib import Path

import graphviz

SRC_DIR = Path(__file__).parent.parent / "src" / "wildfrost_rag"


def _top_level_packages(src_dir: Path) -> set[str]:
    """Every immediate subdirectory/module of src/ that could be imported."""
    packages = set()
    for entry in src_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith("__"):
            packages.add(entry.name)
        elif entry.suffix == ".py" and entry.stem != "__init__":
            packages.add(entry.stem)
    return packages


def _module_package(file_path: Path, src_dir: Path) -> str:
    """Top-level package name for a file under src/ (e.g. rag/retrievers/x.py -> rag)."""
    return file_path.relative_to(src_dir).parts[0]


def _resolve_relative_import(node: ast.ImportFrom, file_path: Path, src_dir: Path) -> str | None:
    """Resolve a relative import (from .foo import Bar) to a dotted module path."""
    if node.level == 0:
        return node.module
    # node.level == 1 means "current package", 2 means "parent package", etc.
    package_parts = list(file_path.relative_to(src_dir).parent.parts)
    ascend = node.level - 1
    if ascend > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - ascend] if ascend else package_parts
    if node.module:
        base_parts = [*base_parts, *node.module.split(".")]
    return ".".join(base_parts) if base_parts else None


def _extract_imports(file_path: Path, src_dir: Path) -> set[str]:
    """Extract the set of dotted module names this file imports (absolute or resolved-relative)."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative_import(node, file_path, src_dir)
            if resolved:
                modules.add(resolved)

    return modules


def build_package_graph(src_dir: Path) -> dict[str, set[str]]:
    """Build a package -> {packages it imports} graph, first-party only.

    Absolute imports are written relative to src_dir's own parent package
    (e.g. `from wildfrost_rag.rag...` when src_dir is .../src/wildfrost_rag),
    so strip that root name before resolving the target package. Relative
    imports (`from .foo import`) are already resolved relative to src_dir by
    _resolve_relative_import and never carry that prefix.
    """
    known_packages = _top_level_packages(src_dir)
    root_name = src_dir.name
    graph: dict[str, set[str]] = {pkg: set() for pkg in known_packages}

    for file_path in src_dir.rglob("*.py"):
        source_package = _module_package(file_path, src_dir)
        if source_package not in known_packages:
            continue

        for module in _extract_imports(file_path, src_dir):
            unprefixed = module.removeprefix(f"{root_name}.")
            target_package = unprefixed.split(".")[0]
            if target_package in known_packages and target_package != source_package:
                graph[source_package].add(target_package)

    return graph


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find cycles in the package graph via DFS (white/gray/black coloring)."""
    white, gray, black = 0, 1, 2
    color = dict.fromkeys(graph, white)
    cycles: list[list[str]] = []

    def visit(node: str, path: list[str]) -> None:
        color[node] = gray
        path.append(node)
        for neighbor in sorted(graph.get(node, ())):
            if color[neighbor] == gray:
                cycle_start = path.index(neighbor)
                cycles.append([*path[cycle_start:], neighbor])
            elif color[neighbor] == white:
                visit(neighbor, path)
        path.pop()
        color[node] = black

    for node in sorted(graph):
        if color[node] == white:
            visit(node, [])

    return cycles


def render_svg(graph: dict[str, set[str]], output_path: Path) -> None:
    """Render the package graph to SVG via the graphviz package."""
    dot = graphviz.Digraph(graph_attr={"rankdir": "LR"}, node_attr={"shape": "box"})
    for source, targets in sorted(graph.items()):
        for target in sorted(targets):
            dot.edge(source, target)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(dot.pipe(format="svg"))


def compute_depths(graph: dict[str, set[str]]) -> dict[str, int]:
    """Depth of each package: 0 for a leaf (no first-party imports), else 1 + max(depth of what it imports).

    Assumes an acyclic graph (skip/ignore if find_cycles() found any - depth
    is undefined for a cycle).
    """
    depths: dict[str, int] = {}

    def depth(node: str, visiting: set[str]) -> int:
        if node in depths:
            return depths[node]
        if node in visiting:
            return 0  # cycle guard; find_cycles() is the source of truth for cycles
        visiting.add(node)
        targets = graph.get(node, ())
        result = 0 if not targets else 1 + max(depth(t, visiting) for t in targets)
        visiting.discard(node)
        depths[node] = result
        return result

    for node in graph:
        depth(node, set())
    return depths


def _boxed_row(names: list[str]) -> list[str]:
    """Render a horizontal row of boxed package names (three lines: top/mid/bottom).

    Plain ASCII (+/-/|), not unicode box-drawing chars - Windows terminals
    default to cp1252, which can't encode them.
    """
    widths = [len(n) + 2 for n in names]
    top = "  ".join("+" + "-" * w + "+" for w in widths)
    mid = "  ".join(f"| {n} |" for n in names)
    bottom = "  ".join("+" + "-" * w + "+" for w in widths)
    return [top, mid, bottom]


def print_layered_diagram(graph: dict[str, set[str]]) -> None:
    """Print an ASCII layered diagram.

    Highest depth (most composed) at the top, depth 0 (infra floor - no
    first-party imports) at the bottom.
    """
    depths = compute_depths(graph)
    max_depth = max(depths.values(), default=0)

    layers: dict[int, list[str]] = {d: [] for d in range(max_depth + 1)}
    for pkg, d in depths.items():
        layers[d].append(pkg)

    print("\nLayered package diagram (top = most composed, bottom = infra floor):\n")
    for d in range(max_depth, -1, -1):
        names = sorted(layers[d])
        if not names:
            continue
        label = "infra floor" if d == 0 else f"depth {d}"
        print(f"Layer {d} ({label}):")
        for line in _boxed_row(names):
            print(f"  {line}")
        if d > 0:
            print("                    |")
            print("                    v")


def print_fan_in(graph: dict[str, set[str]], threshold: int = 2) -> None:
    """Print packages imported directly by at least `threshold` other packages."""
    in_degree: dict[str, int] = dict.fromkeys(graph, 0)
    importers: dict[str, list[str]] = {pkg: [] for pkg in graph}
    for pkg, targets in graph.items():
        for target in targets:
            in_degree[target] = in_degree.get(target, 0) + 1
            importers.setdefault(target, []).append(pkg)

    heavily_depended_on = sorted(
        ((pkg, count) for pkg, count in in_degree.items() if count >= threshold),
        key=lambda item: item[1],
        reverse=True,
    )
    if not heavily_depended_on:
        return

    print(f"\nHeavily depended-upon packages (imported by >= {threshold} others):")
    for pkg, count in heavily_depended_on:
        total = len(graph)
        who = ", ".join(sorted(importers[pkg]))
        print(f"  {pkg} - imported by {count}/{total - 1} other packages: {who}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg", type=str, default=None, help="Path to render an SVG graph to")
    args = parser.parse_args()

    graph = build_package_graph(SRC_DIR)
    print(f"Packages found: {len(graph)}")

    cycles = find_cycles(graph)
    print(f"Cycles found: {len(cycles)}")
    for cycle in cycles:
        print(f"  {' -> '.join(cycle)}")

    if cycles:
        print("\nSkipping layered diagram: depth is undefined while a cycle exists.")
    else:
        print_layered_diagram(graph)

    print_fan_in(graph)

    print("\nPackage -> imports (raw edge list):")
    for pkg, targets in sorted(graph.items()):
        if targets:
            print(f"  {pkg} -> {', '.join(sorted(targets))}")
        else:
            print(f"  {pkg} -> (no first-party imports)")

    if args.svg:
        render_svg(graph, Path(args.svg))
        print(f"\nSVG written to {args.svg}")

    if cycles:
        sys.exit(1)


if __name__ == "__main__":
    main()
