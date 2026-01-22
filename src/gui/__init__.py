"""
WildFrostRAG GUI module.

This module provides annotation GUIs for experiments:
- ExperimentBrowserGUI: Discover and select experiments
- UnifiedAnnotationGUI: Single GUI for both retrieval and generation

Usage:
    from src.gui import browse_experiments
    browser = browse_experiments()
"""

from src.gui.experiment_adapters import (
    ExperimentDataAdapter,
    ExperimentRegistry,
    QueryResult,
    ExperimentMetadata,
    get_adapter,
)
from src.gui.unified_annotation_gui import UnifiedAnnotationGUI, create_unified_gui
from src.gui.experiment_browser_gui import ExperimentBrowserGUI, browse_experiments

__all__ = [
    # Adapters
    'ExperimentDataAdapter',
    'ExperimentRegistry',
    'QueryResult',
    'ExperimentMetadata',
    'get_adapter',
    # GUIs
    'UnifiedAnnotationGUI',
    'ExperimentBrowserGUI',
    # Convenience functions
    'create_unified_gui',
    'browse_experiments',
]
