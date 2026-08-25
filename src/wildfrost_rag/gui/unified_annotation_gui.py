"""Unified Annotation GUI for WildFrostRAG.

A single GUI for annotating both retrieval and generation experiments.
Features:
- Chunks tab first (main annotation work)
- Shows ground truth and doc_references from queries JSON
- Auto-populates relevance based on URL matching
- Writes new relevant URLs back to queries JSON
- Shows LLM response when available (generation experiments)
"""

from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import display

from wildfrost_rag.gui.experiment_adapters import ExperimentDataAdapter, QueryResult, get_adapter
from wildfrost_rag.gui.chunk_widgets import create_chunk_widget, create_chunks_summary
from wildfrost_rag.utils.logger import logger

# Import query processing functions
from wildfrost_rag.rag.evaluation.query_data import load_queries_json, add_doc_reference


class UnifiedAnnotationGUI:
    """Unified GUI for annotating retrieval and generation experiments."""

    def __init__(
        self, adapter: ExperimentDataAdapter, queries_json_path: Path | None = None
    ) -> None:
        """Initialize the unified annotation GUI.

        Args:
            adapter: Data adapter for the experiment
            queries_json_path: Path to queries JSON file (for ground truth)
        """
        self.adapter = adapter
        self.queries = adapter.get_queries()
        self.metadata = adapter.get_metadata()
        self.current_index = 0

        # Queries JSON for ground truth
        self.queries_json_path = queries_json_path
        if queries_json_path is None:
            # Default path
            default_path = Path("queries/simple_reference_based_queries.json")
            if default_path.exists():
                self.queries_json_path = default_path

        # Load ground truth data
        self._ground_truth_cache: dict[int, dict[str, Any]] = {}
        self._load_ground_truth()

        # Cache for annotations
        self._annotations_cache: dict[int, dict[str, Any]] = {}
        self._load_annotations_cache()

        # Track expanded/collapsed state for chunks
        self._chunk_expanded: dict[str, bool] = {}

        # Config
        self.chunk_text_size = 14
        self._tabs: widgets.Tab | None = None

        # Create widgets
        self._create_widgets()
        self._setup_callbacks()

        # Initial display
        self._update_display()

    def _load_ground_truth(self) -> None:
        """Load ground truth data from queries JSON."""
        if self.queries_json_path and self.queries_json_path.exists():
            try:
                data = load_queries_json(self.queries_json_path)
                for query in data.get("queries", []):
                    self._ground_truth_cache[query["query_id"]] = {
                        "ground_truth": query.get("ground_truth", ""),
                        "doc_references": query.get("doc_references", []),
                    }
                logger.info(f"Loaded ground truth for {len(self._ground_truth_cache)} queries")
            except Exception as e:
                logger.error(f"Failed to load ground truth: {e}")

    def _load_annotations_cache(self) -> None:
        """Load existing annotations into cache."""
        annotations = self.adapter.load_annotations()
        for query_id_str, ann in annotations.items():
            self._annotations_cache[int(query_id_str)] = ann

    def _create_widgets(self) -> None:
        """Create all GUI widgets."""
        # Header with experiment info
        self.header_display = widgets.HTML()

        # Query navigation
        self.query_dropdown = widgets.Dropdown(
            options=[(str(q.query_id), i) for i, q in enumerate(self.queries)],
            value=0,
            description="Query:",
            layout=widgets.Layout(width="150px"),
            style={"description_width": "50px"},
        )

        self.prev_button = widgets.Button(
            description="< Prev",
            button_style="info",
            layout=widgets.Layout(width="80px"),
        )
        self.next_button = widgets.Button(
            description="Next >",
            button_style="info",
            layout=widgets.Layout(width="80px"),
        )
        self.progress_label = widgets.HTML()

        self.jump_unvalidated_button = widgets.Button(
            description="Jump to Unvalidated",
            button_style="info",
            layout=widgets.Layout(width="150px"),
        )

        # Query display (shows query, ground truth, doc references)
        self.query_display = widgets.HTML()

        # LLM Response display (for generation experiments)
        self.response_display = widgets.HTML()

        # Validation buttons
        self.pass_button = widgets.Button(
            description="Pass",
            button_style="success",
            layout=widgets.Layout(width="80px"),
        )
        self.fail_button = widgets.Button(
            description="Fail",
            button_style="danger",
            layout=widgets.Layout(width="80px"),
        )
        self.clear_button = widgets.Button(description="Clear", layout=widgets.Layout(width="80px"))
        self.validation_status = widgets.HTML()
        self.save_status = widgets.HTML()

        # Qualitative coding inputs
        self.open_coding_input = widgets.Textarea(
            placeholder="Open coding notes...",
            layout=widgets.Layout(width="100%", height="60px"),
        )
        self.axial_coding_input = widgets.Textarea(
            placeholder="Axial coding category...",
            layout=widgets.Layout(width="100%", height="40px"),
        )

        # Font size slider (controls whole GUI)
        self.font_size_slider = widgets.IntSlider(
            value=self.chunk_text_size,
            min=10,
            max=24,
            step=1,
            description="Font:",
            layout=widgets.Layout(width="200px"),
            style={"description_width": "40px"},
        )

        # Collapse/Expand all button (persistent state)
        self.collapse_all_button = widgets.Button(
            description="Collapse All",
            button_style="",
            layout=widgets.Layout(width="120px"),
        )
        self._all_collapsed = False  # Track collapse state at class level

        # Summary container (fixed, not scrollable)
        self.summary_container = widgets.VBox([], layout=widgets.Layout(width="100%"))

        # Chunks container (scrollable)
        self.chunks_container = widgets.VBox([], layout=widgets.Layout(width="100%"))

    def _setup_callbacks(self) -> None:
        """Setup event handlers."""
        self.query_dropdown.observe(self._on_dropdown_change, names="value")
        self.prev_button.on_click(self._on_prev)
        self.next_button.on_click(self._on_next)
        self.jump_unvalidated_button.on_click(self._on_jump_unvalidated)
        self.pass_button.on_click(self._on_pass)
        self.fail_button.on_click(self._on_fail)
        self.clear_button.on_click(self._on_clear)
        self.open_coding_input.observe(self._on_open_coding_change, names="value")
        self.axial_coding_input.observe(self._on_axial_coding_change, names="value")
        self.font_size_slider.observe(self._on_font_size_change, names="value")
        self.collapse_all_button.on_click(self._on_collapse_all)

    def _on_dropdown_change(self, change: dict[str, Any]) -> None:
        # Clear chunks immediately to avoid visual glitch of old content
        self.chunks_container.children = []
        self.summary_container.children = []

        self.current_index = change["new"]
        self._update_display()

    def _on_prev(self, btn: widgets.Button) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.query_dropdown.value = self.current_index
            # Note: dropdown observer already calls _update_display()

    def _on_next(self, btn: widgets.Button) -> None:
        if self.current_index < len(self.queries) - 1:
            self.current_index += 1
            self.query_dropdown.value = self.current_index
            # Note: dropdown observer already calls _update_display()

    def _on_jump_unvalidated(self, btn: widgets.Button) -> None:
        for i, query in enumerate(self.queries):
            ann = self._annotations_cache.get(query.query_id, {})
            if "validation" not in ann:
                self.current_index = i
                self.query_dropdown.value = i
                # Note: dropdown observer already calls _update_display()
                return
        self.save_status.value = '<span style="color: green;">All queries validated!</span>'

    def _on_pass(self, btn: widgets.Button) -> None:
        self._save_annotation("validation", "pass")
        self._update_validation_status()

    def _on_fail(self, btn: widgets.Button) -> None:
        self._save_annotation("validation", "fail")
        self._update_validation_status()

    def _on_clear(self, btn: widgets.Button) -> None:
        self._save_annotation("validation", None)
        self._update_validation_status()

    def _on_open_coding_change(self, change: dict[str, Any]) -> None:
        self._save_annotation("open_coding", change["new"])

    def _on_axial_coding_change(self, change: dict[str, Any]) -> None:
        self._save_annotation("axial_coding", change["new"])

    def _on_font_size_change(self, change: dict[str, Any]) -> None:
        self.chunk_text_size = change["new"]
        # Update the whole GUI display with new font size
        self._update_display()

    def _on_collapse_all(self, btn: widgets.Button) -> None:
        """Toggle collapse/expand all chunks."""
        query = self.queries[self.current_index]
        self._all_collapsed = not self._all_collapsed

        # Update button text
        if self._all_collapsed:
            btn.description = "Expand All"
        else:
            btn.description = "Collapse All"

        # Update all chunk expanded states for current query
        for i in range(len(query.retrieved_chunks)):
            chunk_key = f"{query.query_id}_{i}"
            self._chunk_expanded[chunk_key] = not self._all_collapsed

        # Refresh only the chunks display (not whole display to preserve button state)
        self._update_chunks_display(query)

    def _save_annotation(self, key: str, value: str | None) -> None:
        """Save an annotation for the current query."""
        query = self.queries[self.current_index]
        query_id = query.query_id

        if query_id not in self._annotations_cache:
            self._annotations_cache[query_id] = {}

        if value is None:
            self._annotations_cache[query_id].pop(key, None)
        else:
            self._annotations_cache[query_id][key] = value

        self.adapter.save_annotation(query_id, self._annotations_cache[query_id])
        self._show_save_status()

    def _show_save_status(self) -> None:
        self.save_status.value = '<span style="color: green; font-size: 12px;">Saved</span>'

    def _update_display(self) -> None:
        """Update all display elements for current query."""
        query = self.queries[self.current_index]

        self._update_header()
        self._update_query_display(query)
        self._update_response_display(query)
        self._update_validation_status()
        self._update_coding_inputs(query)
        self._update_chunks_display(query)
        self._update_progress()
        self._update_tab_title(query)

    def _update_header(self) -> None:
        """Update header with experiment info."""
        meta = self.metadata
        exp_type = meta.experiment_type.capitalize()

        self.header_display.value = f"""
        <div style="padding: 10px; background-color: #e8f5e9; border-radius: 5px; border-left: 4px solid #4caf50; margin-bottom: 15px;">
            <h3 style="margin: 0 0 10px 0; color: #2e7d32;">{exp_type} Experiment: {meta.experiment_id}</h3>
            <div style="display: flex; gap: 30px; font-size: 14px; flex-wrap: wrap;">
                <div><b>Run:</b> {meta.run_number}</div>
                <div><b>Queries:</b> {meta.total_queries}</div>
                {f"<div><b>Retriever:</b> {meta.retriever_type}</div>" if meta.retriever_type else ""}
                {f"<div><b>LLM:</b> {meta.llm_model}</div>" if meta.llm_model else ""}
            </div>
        </div>
        """

    def _update_query_display(self, query: QueryResult) -> None:
        """Update query display with ground truth info."""
        gt_data = self._ground_truth_cache.get(query.query_id, {})
        ground_truth = gt_data.get("ground_truth", "")
        doc_refs = gt_data.get("doc_references", [])

        # Format doc references as clickable links - SEPARATE BUBBLE
        refs_html = ""
        if doc_refs:
            refs_list = "".join(
                [
                    f'<li><a href="{url}" target="_blank" style="color: #1565c0;">{url}</a></li>'
                    for url in doc_refs
                ]
            )
            refs_html = f"""
            <div style="margin-top: 10px; padding: 10px; background-color: #e8f5e9; border-radius: 5px; border-left: 3px solid #4caf50;">
                <b style="color: #2e7d32;">Doc References:</b>
                <ul style="margin: 5px 0; padding-left: 20px;">{refs_list}</ul>
            </div>
            """

        # Ground truth section - SEPARATE BUBBLE
        gt_html = ""
        if ground_truth:
            gt_html = f"""
            <div style="margin-top: 10px; padding: 10px; background-color: #fff8e1; border-radius: 5px; border-left: 3px solid #ffc107;">
                <b style="color: #f57c00;">Ground Truth:</b>
                <div style="margin-top: 5px; white-space: pre-wrap;">{ground_truth}</div>
            </div>
            """

        self.query_display.value = f"""
        <div style="padding: 15px; background-color: #e3f2fd; border-radius: 5px; margin-bottom: 15px;">
            <h4 style="margin: 0 0 10px 0; color: #1565c0;">Query #{query.query_id}</h4>
            <div style="font-size: 16px; font-weight: 500;">{query.query}</div>
            {gt_html}
            {refs_html}
        </div>
        """

    def _update_response_display(self, query: QueryResult) -> None:
        """Update LLM response display (for generation experiments)."""
        if query.response:
            self.response_display.value = f"""
            <div style="padding: 15px; background-color: #f3e5f5; border-radius: 5px; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: #7b1fa2;">LLM Response</h4>
                <div style="white-space: pre-wrap;">{query.response}</div>
            </div>
            """
        else:
            self.response_display.value = """
            <div style="padding: 15px; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 15px; color: #999;">
                <i>No LLM response (retrieval-only experiment)</i>
            </div>
            """

    def _update_validation_status(self) -> None:
        """Update validation status display."""
        query = self.queries[self.current_index]
        ann = self._annotations_cache.get(query.query_id, {})
        validation = ann.get("validation")

        if validation == "pass":
            self.validation_status.value = (
                '<span style="color: green; font-weight: bold; font-size: 16px;">PASS</span>'
            )
        elif validation == "fail":
            self.validation_status.value = (
                '<span style="color: red; font-weight: bold; font-size: 16px;">FAIL</span>'
            )
        else:
            self.validation_status.value = (
                '<span style="color: gray; font-size: 14px;">Not validated</span>'
            )

    def _update_coding_inputs(self, query: QueryResult) -> None:
        """Update coding input fields."""
        ann = self._annotations_cache.get(query.query_id, {})
        self.open_coding_input.value = ann.get("open_coding", "")
        self.axial_coding_input.value = ann.get("axial_coding", "")

    def _update_chunks_display(self, query: QueryResult) -> None:
        """Update chunks display with auto-relevance."""
        if not query.retrieved_chunks:
            self.summary_container.children = []
            self.chunks_container.children = [
                widgets.HTML(
                    value='<div style="padding: 20px; text-align: center; color: #666;"><i>No chunks retrieved</i></div>'
                )
            ]
            return

        # Get doc references for auto-relevance
        gt_data = self._ground_truth_cache.get(query.query_id, {})
        doc_refs = gt_data.get("doc_references", [])

        # Get relevance annotations from cache
        ann = self._annotations_cache.get(query.query_id, {})
        relevance_annotations = ann.get("relevance_annotations", [])

        # Create summary
        total_chunks = len(query.retrieved_chunks)
        relevant_count = sum(1 for ra in relevance_annotations if ra.get("is_relevant", False))

        # Store reference to chunk widgets for collapse/expand all
        self._current_chunk_widgets = []

        summary = create_chunks_summary(
            total_chunks,
            relevant_count,
            relevance_annotations=relevance_annotations,
            doc_references=doc_refs,
            query_text=query.query,
        )

        # Put summary in fixed container (not scrollable)
        self.summary_container.children = [summary]

        # Create chunk widgets (scrollable)
        chunk_widgets = []
        for i, chunk in enumerate(query.retrieved_chunks):
            chunk_widget = self._create_chunk_widget(query.query_id, i, chunk, doc_refs)
            chunk_widgets.append(chunk_widget)
            self._current_chunk_widgets.append(chunk_widget)

        self.chunks_container.children = chunk_widgets

    def _get_chunk_relevance(self, query_id: int, chunk_idx: int) -> tuple[bool, bool]:
        """Check existing relevance annotation for a chunk.

        Returns:
            Tuple of (is_relevant, has_annotation).
        """
        annotation = self._annotations_cache.get(query_id, {})
        relevance_list = annotation.get("relevance_annotations", [])

        for relevance_annotation in relevance_list:
            if relevance_annotation.get("chunk_index") == chunk_idx:
                return relevance_annotation.get("is_relevant", False), True
        return False, False

    def _auto_populate_relevance(
        self, query_id: int, chunk_idx: int, chunk: dict[str, Any], doc_refs: list[str]
    ) -> bool:
        """Auto-populate relevance based on URL matching.

        Returns:
            True if chunk was auto-marked as relevant.
        """
        source_url = chunk.get("source_url", "")
        if not source_url or not any(source_url in ref or ref in source_url for ref in doc_refs):
            return False

        self.adapter.save_chunk_relevance(query_id, chunk_idx, True)
        if query_id not in self._annotations_cache:
            self._annotations_cache[query_id] = {}
        if "relevance_annotations" not in self._annotations_cache[query_id]:
            self._annotations_cache[query_id]["relevance_annotations"] = []
        self._annotations_cache[query_id]["relevance_annotations"].append(
            {"chunk_index": chunk_idx, "is_relevant": True, "auto_populated": True}
        )
        return True

    def _update_relevance_cache(self, query_id: int, chunk_idx: int, new_value: bool) -> None:
        """Update the relevance annotation cache for a chunk."""
        if query_id not in self._annotations_cache:
            self._annotations_cache[query_id] = {}
        ann = self._annotations_cache[query_id]
        if "relevance_annotations" not in ann:
            ann["relevance_annotations"] = []

        for relevance_annotation in ann["relevance_annotations"]:
            if relevance_annotation.get("chunk_index") == chunk_idx:
                relevance_annotation["is_relevant"] = new_value
                return
        ann["relevance_annotations"].append({"chunk_index": chunk_idx, "is_relevant": new_value})

    def _create_chunk_widget(
        self, query_id: int, chunk_idx: int, chunk: dict[str, Any], doc_refs: list[str]
    ) -> widgets.Widget:
        """Create a chunk widget with auto-relevance."""
        chunk_key = f"{query_id}_{chunk_idx}"

        is_relevant, has_annotation = self._get_chunk_relevance(query_id, chunk_idx)

        if not has_annotation:
            if self._auto_populate_relevance(query_id, chunk_idx, chunk, doc_refs):
                is_relevant = True

        def on_relevance_change(new_value: bool) -> None:
            self.adapter.save_chunk_relevance(query_id, chunk_idx, new_value)
            self._update_relevance_cache(query_id, chunk_idx, new_value)

            if new_value and self.queries_json_path:
                source_url = chunk.get("source_url", "")
                if source_url:
                    added = add_doc_reference(self.queries_json_path, query_id, source_url)
                    if added and query_id in self._ground_truth_cache:
                        self._ground_truth_cache[query_id]["doc_references"].append(source_url)
                        self.save_status.value = f'<span style="color: green; font-size: 12px;">Added {source_url} to doc_references</span>'

            self._show_save_status()
            query = self.queries[self.current_index]
            self._update_chunks_display(query)

        def on_toggle(new_expanded: bool) -> None:
            self._chunk_expanded[chunk_key] = new_expanded

        kwargs: dict[str, bool] = {}
        if chunk_key in self._chunk_expanded:
            kwargs["is_expanded"] = self._chunk_expanded[chunk_key]

        return create_chunk_widget(
            chunk=chunk,
            chunk_idx=chunk_idx,
            is_relevant=is_relevant,
            font_size=self.chunk_text_size,
            on_relevance_change=on_relevance_change,
            on_toggle=on_toggle,
            show_detailed_metadata=True,
            **kwargs,
        )

    def _update_progress(self) -> None:
        """Update progress display."""
        total = len(self.queries)
        self.progress_label.value = (
            f'<span style="font-size: 14px;">{self.current_index + 1} / {total}</span>'
        )

    def _update_tab_title(self, query: QueryResult) -> None:
        """Update the chunks tab title with chunk count."""
        if self._tabs is not None:
            self._tabs.set_title(0, f"Chunks ({len(query.retrieved_chunks)})")

    def get_validation_summary(self) -> dict[str, int]:
        """Get summary of validation status."""
        summary = {"pass": 0, "fail": 0, "unvalidated": 0}
        for query in self.queries:
            ann = self._annotations_cache.get(query.query_id, {})
            validation = ann.get("validation")
            if validation == "pass":
                summary["pass"] += 1
            elif validation == "fail":
                summary["fail"] += 1
            else:
                summary["unvalidated"] += 1
        return summary

    def display(self) -> None:
        """Display the unified annotation GUI."""
        # Navigation row with query dropdown
        nav_row = widgets.HBox(
            [self.query_dropdown, self.jump_unvalidated_button],
            layout=widgets.Layout(margin="10px 0", align_items="center"),
        )

        # Font size control row (below query dropdown, controls whole GUI)
        font_row = widgets.HBox(
            [
                widgets.HTML(value='<span style="margin-right: 10px;">Display:</span>'),
                self.font_size_slider,
            ],
            layout=widgets.Layout(margin="5px 0", align_items="center"),
        )

        # Navigation buttons
        nav_buttons = widgets.HBox(
            [self.prev_button, self.progress_label, self.next_button],
            layout=widgets.Layout(justify_content="center", margin="10px 0"),
        )

        # Validation row
        validation_row = widgets.HBox(
            [
                self.pass_button,
                self.fail_button,
                self.clear_button,
                widgets.HTML(value='<div style="width: 20px;"></div>'),
                self.validation_status,
                widgets.HTML(value='<div style="width: 20px;"></div>'),
                self.save_status,
            ],
            layout=widgets.Layout(justify_content="center", margin="15px 0"),
        )

        # Coding section - ensure no scrollbar with proper layout
        coding_section = widgets.VBox(
            [
                widgets.HTML(value='<h4 style="margin: 15px 0 10px 0;">Qualitative Coding</h4>'),
                self.open_coding_input,
                widgets.HTML(value='<div style="height: 10px;"></div>'),
                self.axial_coding_input,
            ],
            layout=widgets.Layout(overflow="visible", width="100%"),
        )

        # Chunks section (FIRST TAB)
        # Header with title and collapse button
        chunks_header = widgets.HBox(
            [
                widgets.HTML(
                    value='<h3 style="margin: 0; color: #1976d2; font-size: 18px; font-weight: bold;">Retrieved Chunks</h3>'
                ),
                self.collapse_all_button,
            ],
            layout=widgets.Layout(
                justify_content="space-between",
                align_items="center",
                margin="0 0 10px 0",
            ),
        )

        # Make only the actual chunks scrollable, not the header/summary
        scrollable_chunks = widgets.VBox(
            [self.chunks_container],
            layout=widgets.Layout(width="100%", max_height="450px", overflow_y="auto"),
        )

        chunks_section = widgets.VBox(
            [
                chunks_header,
                self.summary_container,  # Fixed summary (metrics, doc refs, query)
                scrollable_chunks,  # Scrollable chunk list
            ],
            layout=widgets.Layout(width="100%"),
        )

        # Query & Response section (SECOND TAB)
        query_response_section = widgets.VBox(
            [self.query_display, self.response_display, validation_row, coding_section]
        )

        # Create tabs - CHUNKS FIRST
        self._tabs = widgets.Tab()
        self._tabs.children = [chunks_section, query_response_section]
        self._tabs.set_title(
            0, f"Chunks ({len(self.queries[self.current_index].retrieved_chunks)})"
        )
        self._tabs.set_title(1, "Query & Validation")

        # Main container
        container = widgets.VBox(
            [
                self.header_display,
                nav_row,
                font_row,
                nav_buttons,
                widgets.HTML(value='<hr style="margin: 10px 0;">'),
                self._tabs,
            ],
            layout=widgets.Layout(padding="20px"),
        )

        display(container)  # type: ignore[no-untyped-call]


def create_unified_gui(
    experiment_path: Path, queries_json_path: Path | None = None
) -> UnifiedAnnotationGUI:
    """Create a unified annotation GUI for any experiment type.

    Args:
        experiment_path: Path to experiment directory
        queries_json_path: Path to queries JSON file (optional)

    Returns:
        UnifiedAnnotationGUI instance
    """
    adapter = get_adapter(experiment_path)
    return UnifiedAnnotationGUI(adapter, queries_json_path)
