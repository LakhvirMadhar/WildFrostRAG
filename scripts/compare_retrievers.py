#!/usr/bin/env python3
"""Generate a multi-section markdown comparison report from retrieval experiment metrics.

Usage:
    # List all available experiments
    python -m scripts.compare_retrievers --list

    # Generate full sectioned report
    python -m scripts.compare_retrievers --run-num 1 \
        --sections \
            "BM25 Ablation:bm25/003,bm25/002" \
            "Fulltext Ablation:fulltext/002,fulltext/006" \
            "Embeddings:vector_hf/004,vector_gemma/004" \
            "Hybrid:bm25_vector_gemma/003,fulltext_vector_hf/001,fulltext_vector_gemma/001" \
        --include-all

    # Custom output path
    python -m scripts.compare_retrievers --run-num 1 --sections "..." --output outputs/run_1/my_report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from nltk.corpus import stopwords as nltk_stopwords

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_settings


def load_experiment(experiment_dir: Path) -> dict[str, Any] | None:
    """Load config and metrics from an experiment directory."""
    config_path = experiment_dir / "config.json"
    metrics_path = experiment_dir / "metrics.json"

    if not config_path.exists() or not metrics_path.exists():
        return None

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    return {"config": config, "metrics": metrics, "path": str(experiment_dir)}


def discover_experiments(run_num: int) -> dict[str, list[dict[str, Any]]]:
    """Discover all experiments in a run, grouped by retriever type."""
    base = get_settings().paths.outputs_dir / f"run_{run_num}" / "retrievals"
    experiments: dict[str, list[dict[str, Any]]] = {}

    if not base.exists():
        return experiments

    for retriever_dir in sorted(base.iterdir()):
        if not retriever_dir.is_dir():
            continue
        retriever_name = retriever_dir.name
        experiments[retriever_name] = []

        for exp_dir in sorted(retriever_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp = load_experiment(exp_dir)
            if exp:
                exp["experiment_id"] = exp_dir.name
                exp["retriever_dir"] = retriever_name
                experiments[retriever_name].append(exp)

    return experiments


def resolve_experiments(exp_refs: list[str], run_num: int) -> list[dict[str, Any]]:
    """Resolve experiment references like 'bm25/002' to loaded experiment dicts."""
    base = get_settings().paths.outputs_dir / f"run_{run_num}" / "retrievals"
    results = []

    for ref in exp_refs:
        exp_dir = base / ref
        if not exp_dir.exists():
            print(f"  WARNING: Experiment not found: {ref}, skipping")
            continue
        exp = load_experiment(exp_dir)
        if not exp:
            print(f"  WARNING: Missing config/metrics in: {ref}, skipping")
            continue
        parts = ref.split("/")
        exp["retriever_dir"] = parts[0]
        exp["experiment_id"] = parts[1]
        results.append(exp)

    return results


STOPWORD_RELEVANT_RETRIEVERS = {
    "bm25",
    "fulltext",
    "fulltext_vector_hf",
    "fulltext_vector_gemma",
}


def get_display_name(exp: dict[str, Any]) -> str:
    """Build a human-readable column name for an experiment."""
    config = exp["config"]
    retriever_dir = exp["retriever_dir"]
    exp_id = exp["experiment_id"]
    meta = config.get("additional_metadata", {})

    name = retriever_dir

    if retriever_dir not in STOPWORD_RELEVANT_RETRIEVERS:
        if "text2cypher" in retriever_dir:
            prompt_version = config.get("text2cypher_prompt_version", "")
            if prompt_version:
                name += f" {prompt_version}"
        return f"{name} ({exp_id})"

    # New-style: separate sw_query / sw_docs flags
    sw_query = meta.get("sw_query")
    sw_docs = meta.get("sw_docs")
    if sw_query is not None or sw_docs is not None:
        q = "Q" if sw_query else ""
        d = "D" if sw_docs else ""
        tag = q + d
        if tag:
            name += f" +sw({tag})"
        else:
            name += " -sw"
        return f"{name} ({exp_id})"

    # Legacy: single stopwords flag
    stopwords = meta.get("stopwords")
    if stopwords is not None:
        name += " +sw" if stopwords else " -sw"

    return f"{name} ({exp_id})"


METRIC_KEYS = [
    ("avg_hit@1", "Hit@1"),
    ("avg_hit@3", "Hit@3"),
    ("avg_hit@5", "Hit@5"),
    ("avg_hit@10", "Hit@10"),
    ("avg_mrr", "MRR"),
    ("avg_precision@1", "Precision@1"),
    ("avg_recall@10", "Recall@10"),
]


def generate_table(experiments: list[dict[str, Any]]) -> list[str]:
    """Generate a single markdown comparison table (no heading)."""
    col_names = [get_display_name(exp) for exp in experiments]

    # Extract metric values
    rows = []
    for metric_key, metric_label in METRIC_KEYS:
        values = []
        for exp in experiments:
            agg = exp["metrics"]["aggregate_metrics"]
            values.append(agg.get(metric_key, 0.0))
        rows.append((metric_label, values))

    # Relevant doc found row
    found_values = []
    for exp in experiments:
        per_query = exp["metrics"].get("per_query_metrics", [])
        total = len(per_query) if per_query else exp["metrics"]["total_queries"]
        found = sum(1 for q in per_query if q["metrics"].get("hit@10", 0) > 0)
        found_values.append(f"{found}/{total}")

    lines = []

    # Table header
    header = "| Metric | " + " | ".join(col_names) + " |"
    separator = "|---|" + "|".join(["---"] * len(col_names)) + "|"
    lines.append(header)
    lines.append(separator)

    # Metric rows with bolding
    for metric_label, values in rows:
        max_val = max(values)
        cells = []
        for v in values:
            formatted = f"{v:.3f}"
            if v == max_val and values.count(max_val) < len(values):
                formatted = f"**{formatted}**"
            cells.append(formatted)
        lines.append(f"| {metric_label} | " + " | ".join(cells) + " |")

    # Relevant doc found row
    lines.append("| Relevant doc found | " + " | ".join(found_values) + " |")

    return lines


def generate_details(experiments: list[dict[str, Any]]) -> list[str]:
    """Generate experiment details section."""
    lines = []
    for exp in experiments:
        config = exp["config"]
        name = get_display_name(exp)
        desc = config.get("description", "--")
        meta = config.get("additional_metadata", {})
        timestamp = config.get("timestamp", "")[:16]

        lines.append(f"- **{name}**: {desc}")
        sw_query = meta.get("sw_query")
        sw_docs = meta.get("sw_docs")
        if sw_query is not None or sw_docs is not None:
            parts = []
            if sw_query is not None:
                parts.append(f"SW query: {'yes' if sw_query else 'no'}")
            if sw_docs is not None:
                parts.append(f"SW docs: {'yes' if sw_docs else 'no'}")
            lines.append(f"  - {', '.join(parts)}")
        else:
            stopwords = meta.get("stopwords")
            if stopwords is not None:
                lines.append(f"  - Stop words: {'removed' if stopwords else 'kept'}")
        lines.append(f"  - Timestamp: {timestamp}")

    return lines


def generate_missed_queries(experiments: list[dict[str, Any]]) -> list[str]:
    """Generate a missed queries analysis section.

    For each query missed by at least one experiment, shows a row with
    hit/miss per retriever and highlights the stop words in the query.
    """
    stop_words = set(nltk_stopwords.words("english"))

    # Collect per-query hit@10 across all experiments
    # {query_id: {"query": str, exp_key: hit@10_value, ...}}
    query_map: dict[int, dict[str, Any]] = {}
    exp_names = []
    for exp in experiments:
        name = get_display_name(exp)
        exp_names.append(name)
        for q in exp["metrics"].get("per_query_metrics", []):
            qid = q["query_id"]
            if qid not in query_map:
                query_map[qid] = {"query": q["query"]}
            query_map[qid][name] = q["metrics"].get("hit@10", 0)

    # Filter to queries missed by at least one experiment
    missed_qids = sorted(
        qid for qid, data in query_map.items() if any(data.get(name, 0) == 0 for name in exp_names)
    )

    if not missed_qids:
        return ["*All queries found by all retrievers.*", ""]

    lines = []

    # Only show stopword column if any experiment uses stopword removal
    any_sw = any(exp["retriever_dir"] in STOPWORD_RELEVANT_RETRIEVERS for exp in experiments)

    # Build table header
    if any_sw:
        header = "| QID | Query | After +sw(Q) | " + " | ".join(exp_names) + " |"
        sep = "|---|---|---|" + "|".join(["---"] * len(exp_names)) + "|"
    else:
        header = "| QID | Query | " + " | ".join(exp_names) + " |"
        sep = "|---|---|" + "|".join(["---"] * len(exp_names)) + "|"
    lines.append(header)
    lines.append(sep)

    for qid in missed_qids:
        data = query_map[qid]
        query = data["query"]

        # Hit/miss cells
        cells = []
        for name in exp_names:
            hit = data.get(name, 0)
            cells.append("hit" if hit > 0 else "**MISS**")

        if any_sw:
            # Show what the query becomes after stopword removal
            tokens = query.lower().split()
            filtered = [
                t.strip("?.,!")
                for t in tokens
                if t.strip("?.,!") and t.strip("?.,!") not in stop_words
            ]
            sw_removed = " ".join(filtered) if filtered else "—"
            lines.append(f"| {qid} | {query} | {sw_removed} | " + " | ".join(cells) + " |")
        else:
            lines.append(f"| {qid} | {query} | " + " | ".join(cells) + " |")

    lines.append("")

    # Summary stats
    total_queries = len(query_map)
    lines.append(
        f"*{len(missed_qids)} of {total_queries} queries missed by at least one retriever.*"
    )
    lines.append("")

    return lines


def generate_report(sections: list[tuple[str, list[dict[str, Any]]]], run_num: int) -> str:
    """Generate a full multi-section markdown report."""
    lines = []
    lines.append(f"# Retrieval Comparison -- Run {run_num}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(
        "All experiments use 50 simple reference-based queries, auto-annotated via URL matching."
    )
    lines.append("Unannotated queries count as failures (metrics = 0).")
    lines.append("")
    lines.append("**Notation:**")
    lines.append("- `+sw(Q)` = stop words removed from queries")
    lines.append("- `+sw(D)` = stop words removed from documents")
    lines.append("- `+sw(QD)` = stop words removed from both")
    lines.append("- `-sw` = no stop word removal")
    lines.append("")

    # All experiment details collected across sections (deduplicated)
    all_experiments = {}

    for section_name, experiments in sections:
        if not experiments:
            continue

        lines.append(f"## {section_name}")
        lines.append("")
        lines.extend(generate_table(experiments))
        lines.append("")

        for exp in experiments:
            key = f"{exp['retriever_dir']}/{exp['experiment_id']}"
            all_experiments[key] = exp

    # Missed queries analysis (across all experiments)
    all_exps_list = list(all_experiments.values())
    missed_lines = generate_missed_queries(all_exps_list)
    if missed_lines:
        lines.append("## Missed Queries")
        lines.append("")
        lines.extend(missed_lines)

    # Experiment details at the end
    lines.append("## Experiment Details")
    lines.append("")
    lines.extend(generate_details(all_exps_list))
    lines.append("")

    return "\n".join(lines)


def parse_sections(section_args: list[str]) -> list[tuple[str, list[str]]]:
    """Parse section arguments like 'Section Name:exp1,exp2,exp3'."""
    sections = []
    for arg in section_args:
        if ":" not in arg:
            print(f"Invalid section format: '{arg}'. Expected 'Name:exp1,exp2,...'")
            sys.exit(1)
        name, refs_str = arg.split(":", 1)
        refs = [r.strip() for r in refs_str.split(",") if r.strip()]
        sections.append((name.strip(), refs))
    return sections


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for retrieval comparison."""
    parser = argparse.ArgumentParser(description="Compare retrieval experiment metrics")
    parser.add_argument("--run-num", type=int, default=1, help="Run number (default: 1)")
    parser.add_argument(
        "--sections",
        nargs="+",
        help="Sections as 'Name:exp1,exp2,...' (e.g., 'BM25:bm25/002,bm25/003')",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Add a final table combining all experiments across sections",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path (default: outputs/run_{N}/retrieval_comparison.md)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available experiments and exit"
    )
    return parser.parse_args()


def main() -> None:
    """Compare retrieval experiments and generate a markdown report."""
    args = parse_args()
    all_discovered = discover_experiments(args.run_num)

    if not all_discovered:
        print(f"No experiments found for run {args.run_num}")
        sys.exit(1)

    if args.list:
        print(f"Experiments in run {args.run_num}:\n")
        for retriever, exps in all_discovered.items():
            for exp in exps:
                config = exp["config"]
                desc = config.get("description", "--")
                sw = config.get("additional_metadata", {}).get("stopwords")
                sw_str = f" [sw={'yes' if sw else 'no'}]" if sw is not None else ""
                agg = exp["metrics"]["aggregate_metrics"]
                mrr = agg.get("avg_mrr", 0)
                print(f"  {retriever}/{exp['experiment_id']}: MRR={mrr:.3f}{sw_str} -- {desc}")
        return

    if not args.sections:
        print("No --sections provided. Use --list to see available experiments.")
        sys.exit(1)

    # Parse and resolve sections
    parsed_sections = parse_sections(args.sections)
    resolved_sections = []
    all_experiments_combined = []

    for name, refs in parsed_sections:
        experiments = resolve_experiments(refs, args.run_num)
        resolved_sections.append((name, experiments))
        all_experiments_combined.extend(experiments)

    # Deduplicate for the "all" table
    if args.include_all:
        seen = set()
        deduped = []
        for exp in all_experiments_combined:
            key = f"{exp['retriever_dir']}/{exp['experiment_id']}"
            if key not in seen:
                seen.add(key)
                deduped.append(exp)
        resolved_sections.append(("All Retrievers", deduped))

    # Generate report
    md = generate_report(resolved_sections, args.run_num)

    # Output
    output_path = args.output or f"outputs/run_{args.run_num}/retrieval_comparison.md"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Saved to {output_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
