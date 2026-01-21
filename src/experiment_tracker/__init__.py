"""
Experiment tracker - lightweight MLflow-like experiment management.

This module provides:
- Experiment registry (YAML-based)
- Convenience shortcuts (latest/bm25, current run)
- Query interface for finding experiments
"""

from .registry import ExperimentRegistry

__all__ = ['ExperimentRegistry']
