#!/usr/bin/env python3
"""Report fan-in/fan-out per module - see services/architecture/coupling_analysis.py.

Usage:
    python -m wildfrost_rag.cli.coupling_report
    python -m wildfrost_rag.cli.coupling_report --top 20
    python -m wildfrost_rag.cli.coupling_report --package wildfrost_rag.services
"""

import argparse

from wildfrost_rag.core.logger import logger
from wildfrost_rag.services.architecture.coupling_analysis import (
    compute_fan_in_fan_out,
    format_top_n,
)


def main() -> int:
    """CLI entry point - compute and print the fan-in/fan-out report."""
    parser = argparse.ArgumentParser(
        description="Report fan-in/fan-out per module using import-linter's own graph library"
    )
    parser.add_argument(
        "--package",
        default="wildfrost_rag",
        help="Dotted package to analyze (default: wildfrost_rag)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many modules to show per list (default: 15)",
    )
    args = parser.parse_args()

    fan_in, fan_out = compute_fan_in_fan_out(args.package)

    logger.info(f"Analyzed {len(fan_in)} modules under '{args.package}'")
    print(f"\n=== Top {args.top} by fan-in (most depended-upon) ===")
    print(format_top_n(fan_in, args.top))
    print(f"\n=== Top {args.top} by fan-out (imports the most other modules) ===")
    print(format_top_n(fan_out, args.top))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
