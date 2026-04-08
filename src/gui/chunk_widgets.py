"""Shared chunk widget components for WildFrostRAG annotation GUIs.

This module provides reusable widgets for displaying retrieved chunks
across both retrieval and generation annotation GUIs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

import ipywidgets as widgets


def _build_metadata_html(chunk: dict[str, Any], show_detailed: bool) -> str:
    """Build HTML string for chunk metadata (source, score, search type)."""
    parts = []

    source_url = chunk.get("source_url", "")
    source_file = chunk.get("source_file", "") or chunk.get("source", "")
    if source_file:
        filename = Path(source_file).name
        if source_url:
            parts.append(
                f'<b>Source:</b> <a href="{source_url}" target="_blank">{filename}</a>'
            )
        else:
            parts.append(f"<b>Source:</b> {filename}")

    score = chunk.get("score")
    if score is not None:
        parts.append(f"<b>Score:</b> {score:.4f}")

    search_type = chunk.get("search_type")
    if search_type:
        parts.append(f"<b>Type:</b> {search_type}")

    if show_detailed:
        rrf_score = chunk.get("rrf_score")
        if rrf_score is not None and rrf_score != score:
            parts.append(f"<b>RRF:</b> {rrf_score:.4f}")

        source_retriever = chunk.get("source_retriever")
        if source_retriever:
            parts.append(f"<b>Retriever:</b> {source_retriever}")

        retriever_scores = chunk.get("retriever_scores", {})
        if retriever_scores:
            scores_str = ", ".join(
                [f"{k}: {v:.4f}" for k, v in retriever_scores.items()]
            )
            parts.append(f"<b>Scores:</b> {scores_str}")

    return " | ".join(parts)


def _build_rank_badge(chunk_idx: int) -> str:
    """Build HTML for the rank badge with color based on position."""
    rank_color = (
        "#4caf50" if chunk_idx < 3 else "#2196f3" if chunk_idx < 5 else "#9e9e9e"
    )
    return f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="background-color: {rank_color}; color: white; padding: 2px 8px; border-radius: 10px; font-weight: bold;">
            #{chunk_idx + 1}
        </span>
        <span style="font-weight: bold;">Chunk</span>
    </div>
    """


def create_chunk_widget(
    chunk: dict[str, Any],
    chunk_idx: int,
    is_relevant: bool = False,
    is_expanded: bool = False,
    font_size: int = 14,
    on_relevance_change: Callable[[bool], None] | None = None,
    on_toggle: Callable[[bool], None] | None = None,
    show_detailed_metadata: bool = False,
) -> widgets.Widget:
    """Create a widget for displaying a single chunk.

    Args:
        chunk: Dict containing chunk data (text, source_file, source_url, score, etc.)
        chunk_idx: Index of the chunk (0-based)
        is_relevant: Whether the chunk is marked as relevant
        is_expanded: Whether the chunk text is expanded
        font_size: Font size for chunk text
        on_relevance_change: Callback when relevance checkbox changes
        on_toggle: Callback when expand/collapse is toggled
        show_detailed_metadata: If True, show detailed metadata (for retrieval GUI)

    Returns:
        ipywidgets.Widget containing the chunk display
    """
    checkbox = widgets.Checkbox(
        value=is_relevant, description="Relevant", layout=widgets.Layout(width="auto")
    )

    if on_relevance_change:

        def _on_change(change: dict[str, Any]) -> None:
            on_relevance_change(change["new"])

        checkbox.observe(_on_change, names="value")

    text = chunk.get("retrieved_text", "No text available")
    metadata_html = _build_metadata_html(chunk, show_detailed_metadata)

    toggle_button = widgets.Button(
        description="Collapse" if is_expanded else "Expand",
        button_style="",
        layout=widgets.Layout(width="100px", height="28px"),
    )

    text_display = widgets.HTML(
        value=f'<div style="font-size: {font_size}px; white-space: pre-wrap; padding: 10px; background-color: #fafafa; border-radius: 3px;">{text}</div>',
        layout=widgets.Layout(
            display="block" if is_expanded else "none",
            width="100%",
            height="auto",
            overflow="visible",
        ),
    )

    def toggle_text(btn: widgets.Button) -> None:
        current_expanded = text_display.layout.display == "block"
        new_expanded = not current_expanded
        text_display.layout.display = "block" if new_expanded else "none"
        btn.description = "Collapse" if new_expanded else "Expand"
        if on_toggle:
            on_toggle(new_expanded)

    toggle_button.on_click(toggle_text)

    header = widgets.HBox(
        [widgets.HTML(value=_build_rank_badge(chunk_idx)), toggle_button, checkbox],
        layout=widgets.Layout(
            justify_content="flex-start", align_items="center", gap="10px"
        ),
    )

    metadata_widget = widgets.HTML(
        value=f'<div style="font-size: 12px; color: #666; padding: 5px 0;">{metadata_html}</div>'
    )

    border_color = "#4caf50" if is_relevant else "#2196F3"
    container = widgets.VBox(
        [header, metadata_widget, text_display],
        layout=widgets.Layout(
            width="100%",
            height="auto",
            padding="12px",
            margin="8px 0",
            border=f"2px solid {border_color}",
            border_radius="8px",
            background_color="white",
            overflow="visible",
        ),
    )

    return container


@dataclass
class _HitMetrics:
    """Computed retrieval hit metrics."""

    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    hit_at_10: int = 0
    mrr: float = 0.0


def _calculate_hit_metrics(
    relevance_annotations: list[dict[str, Any]] | None,
    total_chunks: int,
) -> _HitMetrics:
    """Calculate Hit@k and MRR from relevance annotations."""
    metrics = _HitMetrics()
    if not relevance_annotations:
        return metrics

    relevance_map: dict[int, bool] = {}
    for annotation in relevance_annotations:
        idx = annotation.get("chunk_index")
        if idx is not None:
            relevance_map[idx] = annotation.get("is_relevant", False)

    hit_thresholds = {1: "hit_at_1", 3: "hit_at_3", 5: "hit_at_5", 10: "hit_at_10"}
    for rank in range(total_chunks):
        if relevance_map.get(rank, False):
            if metrics.mrr == 0.0:
                metrics.mrr = 1.0 / (rank + 1)
            for threshold, attr in hit_thresholds.items():
                if rank < threshold:
                    setattr(metrics, attr, 1)

    return metrics


def create_chunks_summary(
    total_chunks: int,
    relevant_count: int,
    relevance_annotations: list[dict[str, Any]] | None = None,
    doc_references: list[str] | None = None,
    query_text: str | None = None,
) -> widgets.Widget:
    """Create a summary widget showing chunk statistics and retrieval metrics.

    Args:
        total_chunks: Total number of chunks
        relevant_count: Number of chunks marked as relevant
        relevance_annotations: List of relevance annotations with chunk_index and is_relevant
        doc_references: List of doc reference URLs
        query_text: The query text to display

    Returns:
        Summary widget (HTML)
    """
    precision = relevant_count / total_chunks if total_chunks > 0 else 0
    total_relevant_expected = len(doc_references) if doc_references else 0
    recall = (
        relevant_count / total_relevant_expected if total_relevant_expected > 0 else 0
    )
    recall_display = f"{recall:.2%}" if total_relevant_expected > 0 else "N/A"

    hits = _calculate_hit_metrics(relevance_annotations, total_chunks)

    metrics_html = f"""
    <div style="padding: 15px; background-color: #e3f2fd; border-radius: 5px; margin-bottom: 15px;">
        <h4 style="margin: 0 0 10px 0; color: #1976d2;">Retrieval Summary</h4>
        <div style="display: flex; gap: 20px; font-size: 14px; flex-wrap: wrap;">
            <div><b>Total Chunks:</b> {total_chunks}</div>
            <div><b>Marked Relevant:</b> {relevant_count}</div>
            <div><b>Precision@k:</b> {precision:.2%}</div>
            <div><b>Recall@k:</b> {recall_display}</div>
        </div>
        <div style="display: flex; gap: 20px; font-size: 14px; margin-top: 8px; flex-wrap: wrap;">
            <div><b>Hit@1:</b> {"✓" if hits.hit_at_1 else "✗"}</div>
            <div><b>Hit@3:</b> {"✓" if hits.hit_at_3 else "✗"}</div>
            <div><b>Hit@5:</b> {"✓" if hits.hit_at_5 else "✗"}</div>
            <div><b>Hit@10:</b> {"✓" if hits.hit_at_10 else "✗"}</div>
            <div><b>MRR:</b> {hits.mrr:.3f}</div>
        </div>
    """

    if query_text:
        metrics_html += f"""
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #90caf9;">
            <b>Query:</b> {query_text}
        </div>
        """

    if doc_references:
        refs_list = "".join(
            [
                f'<a href="{url}" target="_blank" style="color: #1565c0; margin-right: 10px;">{url}</a>'
                for url in doc_references
            ]
        )
        metrics_html += f"""
        <div style="margin-top: 8px;">
            <b>Expected Sources:</b> {refs_list}
        </div>
        """

    metrics_html += "</div>"

    return widgets.HTML(value=metrics_html)
