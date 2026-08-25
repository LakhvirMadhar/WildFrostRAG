#!/usr/bin/env python3
"""Batch Retrieval Runner for WildFrostRAG.

This script orchestrates running multiple retrieval experiments from a YAML config file,
enabling "go get coffee" workflow for testing multiple retrievers and embedders.

Usage:
    poetry run python -m scripts.run_batch_retrievals --config experiments_config.yaml
"""

import asyncio
import argparse
import yaml
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.evaluate_retrievers import run as run_retrieval
from utils.config import get_settings
from utils.logger import logger


async def main() -> None:  # noqa: C901
    """Run batch retrieval experiments from a YAML config."""
    parser = argparse.ArgumentParser(description="Run batch retrieval experiments")
    parser.add_argument("--config", required=True, help="Path to experiments config YAML")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_num = config["run_num"]
    chunking = config.get("chunking", False)
    retrievers = config["retrievers"]

    print(f"Starting batch retrieval for Run {run_num}")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Chunking: {chunking}")
    print(f"Retrievers to run: {len(retrievers)}")
    print("=" * 60)

    # Validate embeddings exist in config
    settings = get_settings()
    for retriever_config in retrievers:
        if "embedder" in retriever_config:
            embedder = retriever_config["embedder"]
            if embedder not in settings.embedding_configs:
                logger.error(f"Unknown embedder: {embedder}")
                logger.error(f"Available embedders: {list(settings.embedding_configs.keys())}")
                logger.error("Please add to config.py or run add_embeddings.py")
                exit(1)

    # Run each retriever
    results = []
    for i, retriever_config in enumerate(retrievers):
        print(
            f"\n[{i + 1}/{len(retrievers)}] Running {retriever_config['type']}...",
            end="",
        )
        if "embedder" in retriever_config:
            print(f" (embedder: {retriever_config['embedder']})", end="")
        print()

        # Check if already exists
        if experiment_exists(run_num, retriever_config):
            if is_llm_based(retriever_config["type"]):
                print("  Creating new experiment (LLM-based retriever)")
            else:
                print("  [OK] Already exists, skipping")
                results.append({"retriever": retriever_config, "status": "skipped"})
                continue

        # Run retriever
        try:
            await run_retriever_direct(run_num, chunking, retriever_config)
            print("  [OK] Completed")
            results.append({"retriever": retriever_config, "status": "success"})
        except Exception as e:
            print(f"  [FAILED] {e}")
            results.append({"retriever": retriever_config, "status": "failed", "error": str(e)})

            # Stop if non-LLM retriever fails
            if retriever_config["type"] != "text2cypher":
                print("\n  Stopping batch run (non-LLM retriever failed)")
                break

    # Print summary
    print("\n" + "=" * 60)
    success_count = len([r for r in results if r["status"] == "success"])
    skipped_count = len([r for r in results if r["status"] == "skipped"])
    failed_count = len([r for r in results if r["status"] == "failed"])

    print("Batch complete!")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print("=" * 60)

    # Print next steps
    print("\nNext steps:")
    print("  1. Open annotation_gui.ipynb")
    print("  2. Evaluate retrieval results")
    print(f"  3. Run: poetry run python -m scripts.calculate_retrieval_metrics --run-num {run_num}")


def experiment_exists(run_num: int, retriever_config: dict[str, str]) -> bool:
    """Check if experiment already exists."""
    retriever_type = retriever_config["type"]

    # Add embedder suffix for vector retrievers
    if "embedder" in retriever_config:
        retriever_type = f"{retriever_type}_{retriever_config['embedder']}"

    path = get_settings().outputs_dir / f"run_{run_num}" / "retrievals" / retriever_type
    return path.exists() and any(path.iterdir())  # Check if any experiments exist


def is_llm_based(retriever_type: str) -> bool:
    """Check if retriever uses LLM (should re-run even if exists)."""
    return retriever_type == "text2cypher"


async def run_retriever_direct(
    run_num: int, chunking: bool, retriever_config: dict[str, str]
) -> None:
    """Run evaluate_retrievers directly."""
    retrieval_args = argparse.Namespace(
        run_num=run_num,
        retriever=retriever_config["type"],
        chunking="yes" if chunking else "no",
        description=retriever_config.get("description", ""),
        text2cypher_prompt=retriever_config.get("prompt", "TEXT2CYPHER_PROMPT_V1"),
        query_ids=None,
        exclude_query_ids=None,
        k=10,
        file="queries/simple_reference_based_queries.csv",
        embedder=retriever_config.get("embedder", "hf"),
        queries_json=None,
        sw_query="yes",
        sw_docs="yes",
    )
    await run_retrieval(retrieval_args)


if __name__ == "__main__":
    asyncio.run(main())
