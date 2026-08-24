"""AST-based code quality analyzer.

Analyzes Python files for common code smells and complexity metrics.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Generator

# Thresholds for code quality analysis
LONG_FUNCTION = 50  # lines
MANY_ARGS = 5
DEEP_NESTING = 4
MANY_BRANCHES = 10
LARGE_CLASS = 200  # lines
MANY_METHODS = 15


@dataclass
class FunctionMetrics:
    """Metrics collected for a single function."""

    name: str
    file: str
    line: int
    lines_of_code: int
    num_arguments: int
    num_returns: int
    nested_depth: int
    num_branches: int  # if/elif/else/for/while/try/except
    num_local_vars: int
    has_docstring: bool


@dataclass
class ClassMetrics:
    """Metrics collected for a single class."""

    name: str
    file: str
    line: int
    num_methods: int
    num_attributes: int
    lines_of_code: int
    has_docstring: bool


@dataclass
class FileMetrics:
    """Metrics collected for a single Python file."""

    path: str
    lines_of_code: int
    num_functions: int
    num_classes: int
    num_imports: int
    functions: list[FunctionMetrics] = field(default_factory=list)
    classes: list[ClassMetrics] = field(default_factory=list)


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor that collects code metrics."""

    def __init__(self, file_path: str, source: str) -> None:
        """Initialize the analyzer for a single file."""
        self.file_path = file_path
        self.source = source
        self.lines = source.splitlines()
        self.functions: list[FunctionMetrics] = []
        self.classes: list[ClassMetrics] = []
        self.num_imports = 0
        self._current_depth = 0

    def _count_lines(self, node: ast.AST) -> int:
        """Count lines of code for a node."""
        end_lineno = getattr(node, "end_lineno", None)
        lineno = getattr(node, "lineno", None)
        if end_lineno is not None and lineno is not None:
            result: int = end_lineno - lineno + 1
            return result
        return 0

    def _has_docstring(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
        """Check if function/class has a docstring."""
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant):
                return isinstance(node.body[0].value.value, str)
        return False

    def _count_branches(self, node: ast.AST) -> int:
        """Count branching statements in a node."""
        count = 0
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.Try,
                    ast.ExceptHandler,
                    ast.With,
                    ast.Match,
                ),
            ):
                count += 1
        return count

    def _count_nested_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Find maximum nesting depth."""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                child_depth = self._count_nested_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._count_nested_depth(child, depth)
                max_depth = max(max_depth, child_depth)
        return max_depth

    def _count_local_vars(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Count local variable assignments."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                count += len(child.targets)
            elif isinstance(child, ast.AnnAssign):
                count += 1
        return count

    def _count_returns(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Count return statements."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                count += 1
        return count

    def visit_Import(self, node: ast.Import) -> None:
        """Count import statements."""
        self.num_imports += len(node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Count import-from statements."""
        self.num_imports += len(node.names)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze a function definition."""
        self._analyze_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Analyze an async function definition."""
        self._analyze_function(node)
        self.generic_visit(node)

    def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Skip methods (they'll be counted in class analysis)
        # We still want to analyze them, just mark them differently
        num_args = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
        if node.args.vararg:
            num_args += 1
        if node.args.kwarg:
            num_args += 1

        metrics = FunctionMetrics(
            name=node.name,
            file=self.file_path,
            line=node.lineno,
            lines_of_code=self._count_lines(node),
            num_arguments=num_args,
            num_returns=self._count_returns(node),
            nested_depth=self._count_nested_depth(node),
            num_branches=self._count_branches(node),
            num_local_vars=self._count_local_vars(node),
            has_docstring=self._has_docstring(node),
        )
        self.functions.append(metrics)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Analyze a class definition."""
        num_methods = sum(
            1 for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )

        # Count class-level attributes
        num_attrs = 0
        for child in node.body:
            if isinstance(child, ast.Assign):
                num_attrs += len(child.targets)
            elif isinstance(child, ast.AnnAssign):
                num_attrs += 1

        metrics = ClassMetrics(
            name=node.name,
            file=self.file_path,
            line=node.lineno,
            num_methods=num_methods,
            num_attributes=num_attrs,
            lines_of_code=self._count_lines(node),
            has_docstring=self._has_docstring(node),
        )
        self.classes.append(metrics)
        self.generic_visit(node)


def analyze_file(file_path: Path) -> FileMetrics | None:
    """Analyze a single Python file."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  Skipping {file_path}: {e}")
        return None

    analyzer = CodeAnalyzer(str(file_path), source)
    analyzer.visit(tree)

    return FileMetrics(
        path=str(file_path),
        lines_of_code=len(source.splitlines()),
        num_functions=len(analyzer.functions),
        num_classes=len(analyzer.classes),
        num_imports=analyzer.num_imports,
        functions=analyzer.functions,
        classes=analyzer.classes,
    )


def find_python_files(directory: Path) -> Generator[Path, None, None]:
    """Find all Python files in directory, excluding venv and hidden dirs."""
    for path in directory.rglob("*.py"):
        parts = path.parts
        if any(
            part.startswith(".") or part in ("venv", ".venv", "__pycache__", "node_modules")
            for part in parts
        ):
            continue
        yield path


def print_report(all_metrics: list[FileMetrics]) -> None:  # noqa: C901
    """Print analysis report."""
    print("\n" + "=" * 70)
    print("CODE QUALITY REPORT")
    print("=" * 70)

    # Summary
    total_loc = sum(m.lines_of_code for m in all_metrics)
    total_functions = sum(m.num_functions for m in all_metrics)
    total_classes = sum(m.num_classes for m in all_metrics)

    all_functions = [f for m in all_metrics for f in m.functions]
    all_classes = [c for m in all_metrics for c in m.classes]

    print("\nSUMMARY")
    print(f"  Files analyzed: {len(all_metrics)}")
    print(f"  Total lines of code: {total_loc:,}")
    print(f"  Total functions: {total_functions}")
    print(f"  Total classes: {total_classes}")

    # Docstring coverage
    funcs_with_docs = sum(1 for f in all_functions if f.has_docstring)
    classes_with_docs = sum(1 for c in all_classes if c.has_docstring)

    func_doc_pct = (funcs_with_docs / total_functions * 100) if total_functions else 0
    class_doc_pct = (classes_with_docs / total_classes * 100) if total_classes else 0

    print("\nDOCSTRING COVERAGE")
    print(f"  Functions with docstrings: {funcs_with_docs}/{total_functions} ({func_doc_pct:.1f}%)")
    print(f"  Classes with docstrings: {classes_with_docs}/{total_classes} ({class_doc_pct:.1f}%)")

    # Problem areas
    print(f"\n{'=' * 70}")
    print("POTENTIAL ISSUES")
    print("=" * 70)

    # Long functions
    long_funcs = [f for f in all_functions if f.lines_of_code > LONG_FUNCTION]
    if long_funcs:
        print(f"\n[!!] LONG FUNCTIONS (>{LONG_FUNCTION} lines): {len(long_funcs)}")
        for f in sorted(long_funcs, key=lambda x: x.lines_of_code, reverse=True)[:10]:
            print(f"  {f.file}:{f.line} - {f.name}() - {f.lines_of_code} lines")

    # Too many arguments
    many_args = [f for f in all_functions if f.num_arguments > MANY_ARGS]
    if many_args:
        print(f"\n[!] MANY ARGUMENTS (>{MANY_ARGS}): {len(many_args)}")
        for f in sorted(many_args, key=lambda x: x.num_arguments, reverse=True)[:10]:
            print(f"  {f.file}:{f.line} - {f.name}() - {f.num_arguments} args")

    # Deep nesting
    deep_nest = [f for f in all_functions if f.nested_depth > DEEP_NESTING]
    if deep_nest:
        print(f"\n[!] DEEP NESTING (>{DEEP_NESTING} levels): {len(deep_nest)}")
        for f in sorted(deep_nest, key=lambda x: x.nested_depth, reverse=True)[:10]:
            print(f"  {f.file}:{f.line} - {f.name}() - depth {f.nested_depth}")

    # Many branches (high cyclomatic complexity proxy)
    branchy = [f for f in all_functions if f.num_branches > MANY_BRANCHES]
    if branchy:
        print(f"\n[!] HIGH COMPLEXITY (>{MANY_BRANCHES} branches): {len(branchy)}")
        for f in sorted(branchy, key=lambda x: x.num_branches, reverse=True)[:10]:
            print(f"  {f.file}:{f.line} - {f.name}() - {f.num_branches} branches")

    # Large classes
    large_classes = [c for c in all_classes if c.lines_of_code > LARGE_CLASS]
    if large_classes:
        print(f"\n[!!] LARGE CLASSES (>{LARGE_CLASS} lines): {len(large_classes)}")
        for c in sorted(large_classes, key=lambda x: x.lines_of_code, reverse=True):
            print(f"  {c.file}:{c.line} - {c.name} - {c.lines_of_code} lines")

    # Classes with many methods
    method_heavy = [c for c in all_classes if c.num_methods > MANY_METHODS]
    if method_heavy:
        print(f"\n[!] MANY METHODS (>{MANY_METHODS}): {len(method_heavy)}")
        for c in sorted(method_heavy, key=lambda x: x.num_methods, reverse=True):
            print(f"  {c.file}:{c.line} - {c.name} - {c.num_methods} methods")

    # No issues found
    if not any([long_funcs, many_args, deep_nest, branchy, large_classes, method_heavy]):
        print("\n[OK] No major issues found!")

    # Top 10 largest functions
    print(f"\n{'=' * 70}")
    print("TOP 10 LARGEST FUNCTIONS")
    print("=" * 70)
    for f in sorted(all_functions, key=lambda x: x.lines_of_code, reverse=True)[:10]:
        doc_marker = "[D]" if f.has_docstring else "  "
        print(f"  {doc_marker} {f.lines_of_code:3d} lines | {f.name}() @ {f.file}:{f.line}")

    print()


def main() -> None:
    """Run code quality analysis on the specified path."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Python code quality using AST")
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to analyze")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        files = [target]
    else:
        files = list(find_python_files(target))

    print(f"Analyzing {len(files)} Python files...")

    all_metrics = []
    for file_path in files:
        metrics = analyze_file(file_path)
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        print_report(all_metrics)
    else:
        print("No files analyzed.")


if __name__ == "__main__":
    main()
