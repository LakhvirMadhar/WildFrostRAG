"""Experiment Browser GUI for WildFrostRAG.

This module provides a browser interface for discovering and selecting
experiments to annotate. It auto-discovers experiments from the registry
and launches the appropriate annotation GUI.

Usage:
    from gui.experiment_browser_gui import ExperimentBrowserGUI

    browser = ExperimentBrowserGUI()
    browser.display()

    # After selecting an experiment, get the annotation GUI:
    gui = browser.get_annotation_gui()
    gui.display()
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import display, clear_output

from gui.experiment_adapters import ExperimentRegistry, get_adapter
from gui.unified_annotation_gui import UnifiedAnnotationGUI
from utils.logger import logger


class ExperimentBrowserGUI:
    """Browser GUI for discovering and selecting experiments.

    Features:
    - Auto-discovers experiments from experiments.yaml
    - Filter by run number, experiment type, retriever type
    - Preview experiment metadata before selection
    - Launch appropriate annotation GUI for selected experiment
    """

    def __init__(self, outputs_dir: Path | None = None) -> None:
        """Initialize experiment browser.

        Args:
            outputs_dir: Path to outputs directory. Defaults to settings.outputs_dir.
        """
        self.registry = ExperimentRegistry(outputs_dir)
        self.selected_experiment: dict[str, Any] | None = None
        self._annotation_gui: UnifiedAnnotationGUI | None = None

        # Create widgets
        self._create_widgets()
        self._setup_callbacks()

        # Initial load
        self._refresh_experiments()

    def _create_widgets(self) -> None:
        """Create all browser widgets."""
        # Title
        self.title = widgets.HTML(value='<h2 style="margin: 0 0 20px 0;">Experiment Browser</h2>')

        # Filters section
        self.run_dropdown = widgets.Dropdown(
            options=[("All Runs", None)],
            value=None,
            description="Run:",
            layout=widgets.Layout(width="150px"),
        )

        self.type_dropdown = widgets.Dropdown(
            options=[
                ("All Types", None),
                ("Retrieval", "retrieval"),
                ("Generation", "generation"),
            ],
            value=None,
            description="Type:",
            layout=widgets.Layout(width="180px"),
        )

        self.retriever_dropdown = widgets.Dropdown(
            options=[("All Retrievers", None)],
            value=None,
            description="Retriever:",
            layout=widgets.Layout(width="200px"),
        )

        self.refresh_button = widgets.Button(
            description="Refresh",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )

        # Experiment list
        self.experiment_list = widgets.Select(
            options=[],
            description="",
            layout=widgets.Layout(width="100%", height="300px"),
        )

        # Experiment count label
        self.count_label = widgets.HTML(value="")

        # Preview panel
        self.preview_panel = widgets.HTML(
            value='<div style="padding: 20px; text-align: center; color: #666;"><i>Select an experiment to see details</i></div>'
        )

        # Action buttons
        self.annotate_button = widgets.Button(
            description="Open Annotation GUI",
            button_style="success",
            disabled=True,
            layout=widgets.Layout(width="200px", height="40px"),
        )

        # Output area for displaying annotation GUI
        self.output_area = widgets.Output()

    def _setup_callbacks(self) -> None:
        """Setup event handlers."""
        self.run_dropdown.observe(self._on_filter_change, names="value")
        self.type_dropdown.observe(self._on_filter_change, names="value")
        self.retriever_dropdown.observe(self._on_filter_change, names="value")
        self.refresh_button.on_click(self._on_refresh)
        self.experiment_list.observe(self._on_experiment_select, names="value")
        self.annotate_button.on_click(self._on_annotate)

    def _refresh_experiments(self) -> None:
        """Refresh experiment list from registry."""
        # Reload registry
        self.registry._registry = None

        # Update run dropdown
        runs = self.registry.get_all_runs()
        run_options = [("All Runs", None)] + [(f"Run {r}", r) for r in runs]
        self.run_dropdown.options = run_options

        # Update retriever dropdown
        retrievers = self.registry.get_retrievers()
        retriever_options = [("All Retrievers", None)] + [(r, r) for r in retrievers]
        self.retriever_dropdown.options = retriever_options

        # Update experiment list
        self._update_experiment_list()

    def _update_experiment_list(self) -> None:
        """Update experiment list based on current filters."""
        experiments = self.registry.get_experiments(
            run_num=self.run_dropdown.value,
            experiment_type=self.type_dropdown.value,
            retriever_type=self.retriever_dropdown.value,
        )

        # Sort by timestamp descending (newest first)
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Build list options
        options = []
        for exp in experiments:
            # Create display label
            exp_type_icon = "📊" if exp["type"] == "retrieval" else "💬"
            label = f"{exp_type_icon} Run {exp['run_number']} | {exp['id']}"
            if exp.get("description"):
                label += f" - {exp['description'][:30]}..."
            options.append((label, exp))

        self.experiment_list.options = options
        self.count_label.value = (
            f'<span style="color: #666;">Found {len(experiments)} experiments</span>'
        )

        # Reset selection
        self.selected_experiment = None
        self.annotate_button.disabled = True
        self.preview_panel.value = '<div style="padding: 20px; text-align: center; color: #666;"><i>Select an experiment to see details</i></div>'

    def _on_filter_change(self, change: dict[str, Any]) -> None:
        """Handle filter dropdown changes."""
        self._update_experiment_list()

    def _on_refresh(self, _: widgets.Button) -> None:
        """Handle refresh button click."""
        self._refresh_experiments()

    def _on_experiment_select(self, change: dict[str, Any]) -> None:
        """Handle experiment selection."""
        if change["new"] is None:
            self.selected_experiment = None
            self.annotate_button.disabled = True
            self.preview_panel.value = '<div style="padding: 20px; text-align: center; color: #666;"><i>Select an experiment to see details</i></div>'
            return

        self.selected_experiment = change["new"]
        self.annotate_button.disabled = False
        self._update_preview()

    def _update_preview(self) -> None:
        """Update preview panel with selected experiment details."""
        if not self.selected_experiment:
            return

        exp = self.selected_experiment

        # Build detail rows
        rows = []

        # Basic info
        rows.append(f"<tr><td><b>ID:</b></td><td>{exp.get('id', 'unknown')}</td></tr>")
        rows.append(f"<tr><td><b>Type:</b></td><td>{exp.get('type', 'unknown')}</td></tr>")
        rows.append(f"<tr><td><b>Run:</b></td><td>{exp.get('run_number', '?')}</td></tr>")

        # Timestamp
        ts = exp.get("timestamp", "")
        if ts:
            # Format timestamp nicely
            try:
                dt = datetime.fromisoformat(ts)
                ts_formatted = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                ts_formatted = ts[:16]
            rows.append(f"<tr><td><b>Timestamp:</b></td><td>{ts_formatted}</td></tr>")

        # Type-specific info
        if exp.get("type") == "retrieval":
            rows.append(
                f"<tr><td><b>Retriever:</b></td><td>{exp.get('retriever_type', 'unknown')}</td></tr>"
            )
            rows.append(
                f"<tr><td><b>Chunking:</b></td><td>{'Yes' if exp.get('chunking') else 'No'}</td></tr>"
            )
        elif exp.get("type") == "generation":
            rows.append(
                f"<tr><td><b>Retrieval Source:</b></td><td>{exp.get('retrieval_reference', 'N/A')}</td></tr>"
            )
            rows.append(
                f"<tr><td><b>System Prompt:</b></td><td>{exp.get('system_prompt_version', 'unknown')}</td></tr>"
            )

        # Query stats
        total = exp.get("total_queries", 0)
        successful = exp.get("successful_queries", 0)
        rows.append(f"<tr><td><b>Queries:</b></td><td>{successful}/{total} successful</td></tr>")

        # Description
        if exp.get("description"):
            rows.append(f"<tr><td><b>Description:</b></td><td>{exp['description']}</td></tr>")

        # Path
        path_str = str(exp.get("path", ""))
        rows.append(
            f'<tr><td><b>Path:</b></td><td style="font-size: 13px; word-break: break-all;">{path_str}</td></tr>'
        )

        table_html = f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 15px; line-height: 1.8;">
            {"".join(rows)}
        </table>
        """

        # Determine border color based on type
        border_color = "#4caf50" if exp.get("type") == "retrieval" else "#2196f3"

        self.preview_panel.value = f"""
        <div style="padding: 15px; border-left: 4px solid {border_color}; background-color: #fafafa; border-radius: 0 5px 5px 0;">
            <h4 style="margin: 0 0 15px 0;">Experiment Details</h4>
            {table_html}
        </div>
        """

    def _on_annotate(self, _: widgets.Button) -> None:
        """Handle annotate button click."""
        if not self.selected_experiment:
            return

        exp = self.selected_experiment
        path = exp.get("path")

        if not path or not Path(path).exists():
            self.preview_panel.value = f"""
            <div style="padding: 15px; background-color: #ffebee; border-radius: 5px; color: #c62828;">
                <b>Error:</b> Experiment path not found: {path}
            </div>
            """
            return

        # Create unified GUI
        try:
            adapter = get_adapter(path)
            self._annotation_gui = UnifiedAnnotationGUI(adapter)

            # Display in output area
            with self.output_area:
                clear_output()  # type: ignore[no-untyped-call]
                self._annotation_gui.display()

        except Exception as e:
            logger.error(f"Error creating annotation GUI: {e}")
            self.preview_panel.value = f"""
            <div style="padding: 15px; background-color: #ffebee; border-radius: 5px; color: #c62828;">
                <b>Error:</b> {str(e)}
            </div>
            """

    def get_annotation_gui(self) -> UnifiedAnnotationGUI | None:
        """Get the current annotation GUI instance.

        Returns:
            UnifiedAnnotationGUI instance, or None if not launched.
        """
        return self._annotation_gui

    def display(self) -> None:
        """Display the experiment browser."""
        # Filters row
        filters_row = widgets.HBox(
            [
                self.run_dropdown,
                self.type_dropdown,
                self.retriever_dropdown,
                self.refresh_button,
            ],
            layout=widgets.Layout(gap="10px", margin="10px 0"),
        )

        # Main content: list on left, preview on right
        list_section = widgets.VBox(
            [
                widgets.HTML(value='<h4 style="margin: 0 0 10px 0;">Experiments</h4>'),
                self.experiment_list,
                self.count_label,
            ],
            layout=widgets.Layout(width="50%"),
        )

        preview_section = widgets.VBox(
            [
                widgets.HTML(value='<h4 style="margin: 0 0 10px 0;">Preview</h4>'),
                self.preview_panel,
                widgets.HTML(value='<div style="height: 15px;"></div>'),
                self.annotate_button,
            ],
            layout=widgets.Layout(width="50%", padding="0 0 0 20px"),
        )

        content_row = widgets.HBox([list_section, preview_section])

        # Main container (browser section)
        browser_container = widgets.VBox(
            [
                self.title,
                filters_row,
                widgets.HTML(value='<hr style="margin: 15px 0;">'),
                content_row,
            ],
            layout=widgets.Layout(padding="20px", border="1px solid #ddd", border_radius="5px"),
        )

        # Full container with output area for annotation GUI
        full_container = widgets.VBox(
            [
                browser_container,
                widgets.HTML(value='<div style="height: 20px;"></div>'),
                self.output_area,
            ]
        )

        display(full_container)  # type: ignore[no-untyped-call]


def browse_experiments(outputs_dir: Path | None = None) -> ExperimentBrowserGUI:
    """Convenience function to create and display experiment browser.

    Args:
        outputs_dir: Path to outputs directory (optional)

    Returns:
        ExperimentBrowserGUI instance
    """
    browser = ExperimentBrowserGUI(outputs_dir)
    browser.display()
    return browser
