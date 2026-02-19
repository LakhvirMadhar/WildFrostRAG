"""
Shared chunk widget components for WildFrostRAG annotation GUIs.

This module provides reusable widgets for displaying retrieved chunks
across both retrieval and generation annotation GUIs.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import ipywidgets as widgets


def create_chunk_widget(
    chunk: Dict[str, Any],
    chunk_idx: int,
    is_relevant: bool = False,
    is_expanded: bool = False,
    font_size: int = 14,
    on_relevance_change: Optional[Callable[[bool], None]] = None,
    on_toggle: Optional[Callable[[bool], None]] = None,
    show_detailed_metadata: bool = False
) -> widgets.Widget:
    """
    Create a widget for displaying a single chunk.

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
    # Create relevance checkbox
    checkbox = widgets.Checkbox(
        value=is_relevant,
        description='Relevant',
        layout=widgets.Layout(width='auto')
    )

    if on_relevance_change:
        def _on_change(change):
            on_relevance_change(change['new'])
        checkbox.observe(_on_change, names='value')

    # Build metadata display
    metadata_parts = []

    # Source with clickable URL
    source_url = chunk.get('source_url', '')
    source_file = chunk.get('source_file', '') or chunk.get('source', '')
    if source_file:
        filename = Path(source_file).name
        if source_url:
            metadata_parts.append(f'<b>Source:</b> <a href="{source_url}" target="_blank">{filename}</a>')
        else:
            metadata_parts.append(f'<b>Source:</b> {filename}')

    # Score
    score = chunk.get('score')
    if score is not None:
        metadata_parts.append(f'<b>Score:</b> {score:.4f}')

    # Search type
    search_type = chunk.get('search_type')
    if search_type:
        metadata_parts.append(f'<b>Type:</b> {search_type}')

    # Detailed metadata for retrieval GUI
    if show_detailed_metadata:
        # RRF score
        rrf_score = chunk.get('rrf_score')
        if rrf_score is not None and rrf_score != score:
            metadata_parts.append(f'<b>RRF:</b> {rrf_score:.4f}')

        # Source retriever for hybrid
        source_retriever = chunk.get('source_retriever')
        if source_retriever:
            metadata_parts.append(f'<b>Retriever:</b> {source_retriever}')

        # Component scores for hybrid
        retriever_scores = chunk.get('retriever_scores', {})
        if retriever_scores:
            scores_str = ', '.join([f'{k}: {v:.4f}' for k, v in retriever_scores.items()])
            metadata_parts.append(f'<b>Scores:</b> {scores_str}')

    metadata_html = ' | '.join(metadata_parts)

    # Get text content
    text = chunk.get('text', 'No text available')

    # Create toggle button
    toggle_button = widgets.Button(
        description='Collapse' if is_expanded else 'Expand',
        button_style='',
        layout=widgets.Layout(width='100px', height='28px')
    )

    # Create text display
    text_display = widgets.HTML(
        value=f'<div style="font-size: {font_size}px; white-space: pre-wrap; padding: 10px; background-color: #fafafa; border-radius: 3px;">{text}</div>',
        layout=widgets.Layout(
            display='block' if is_expanded else 'none',
            width='100%',
            height='auto',
            overflow='visible'
        )
    )

    def toggle_text(btn):
        current_expanded = text_display.layout.display == 'block'
        new_expanded = not current_expanded
        text_display.layout.display = 'block' if new_expanded else 'none'
        btn.description = 'Collapse' if new_expanded else 'Expand'
        if on_toggle:
            on_toggle(new_expanded)

    toggle_button.on_click(toggle_text)

    # Chunk number with rank badge
    rank_color = '#4caf50' if chunk_idx < 3 else '#2196f3' if chunk_idx < 5 else '#9e9e9e'
    header_html = f'''
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="background-color: {rank_color}; color: white; padding: 2px 8px; border-radius: 10px; font-weight: bold;">
            #{chunk_idx + 1}
        </span>
        <span style="font-weight: bold;">Chunk</span>
    </div>
    '''

    header = widgets.HBox([
        widgets.HTML(value=header_html),
        toggle_button,
        checkbox
    ], layout=widgets.Layout(justify_content='flex-start', align_items='center', gap='10px'))

    # Metadata widget
    metadata_widget = widgets.HTML(
        value=f'<div style="font-size: 12px; color: #666; padding: 5px 0;">{metadata_html}</div>'
    )

    # Container
    border_color = '#4caf50' if is_relevant else '#2196F3'
    container = widgets.VBox([
        header,
        metadata_widget,
        text_display
    ], layout=widgets.Layout(
        width='100%',
        height='auto',
        padding='12px',
        margin='8px 0',
        border=f'2px solid {border_color}',
        border_radius='8px',
        background_color='white',
        overflow='visible'
    ))

    return container


def create_chunks_summary(
    total_chunks: int,
    relevant_count: int,
    relevance_annotations: Optional[list] = None,
    doc_references: Optional[list] = None,
    query_text: Optional[str] = None
) -> widgets.Widget:
    """
    Create a summary widget showing chunk statistics and retrieval metrics.

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

    # Calculate Recall@k: relevant_found / total_relevant_expected
    # total_relevant_expected comes from doc_references (ground truth)
    total_relevant_expected = len(doc_references) if doc_references else 0
    recall = relevant_count / total_relevant_expected if total_relevant_expected > 0 else 0

    # Calculate retrieval metrics from relevance_annotations
    hit_at_1 = hit_at_3 = hit_at_5 = hit_at_10 = 0
    mrr = 0.0

    if relevance_annotations:
        # Build a map of chunk_index -> is_relevant
        relevance_map = {}
        for ra in relevance_annotations:
            idx = ra.get('chunk_index')
            if idx is not None:
                relevance_map[idx] = ra.get('is_relevant', False)

        # Calculate Hit@k and MRR
        first_relevant_rank = None
        for rank in range(total_chunks):
            if relevance_map.get(rank, False):
                if first_relevant_rank is None:
                    first_relevant_rank = rank + 1  # 1-indexed rank
                    mrr = 1.0 / first_relevant_rank
                # Hit@k: did we find a relevant doc in top k?
                if rank < 1:
                    hit_at_1 = 1
                if rank < 3:
                    hit_at_3 = 1
                if rank < 5:
                    hit_at_5 = 1
                if rank < 10:
                    hit_at_10 = 1

    # Build recall display
    recall_display = f'{recall:.2%}' if total_relevant_expected > 0 else 'N/A'

    # Build metrics HTML
    metrics_html = f'''
    <div style="padding: 15px; background-color: #e3f2fd; border-radius: 5px; margin-bottom: 15px;">
        <h4 style="margin: 0 0 10px 0; color: #1976d2;">Retrieval Summary</h4>
        <div style="display: flex; gap: 20px; font-size: 14px; flex-wrap: wrap;">
            <div><b>Total Chunks:</b> {total_chunks}</div>
            <div><b>Marked Relevant:</b> {relevant_count}</div>
            <div><b>Precision@k:</b> {precision:.2%}</div>
            <div><b>Recall@k:</b> {recall_display}</div>
        </div>
        <div style="display: flex; gap: 20px; font-size: 14px; margin-top: 8px; flex-wrap: wrap;">
            <div><b>Hit@1:</b> {"✓" if hit_at_1 else "✗"}</div>
            <div><b>Hit@3:</b> {"✓" if hit_at_3 else "✗"}</div>
            <div><b>Hit@5:</b> {"✓" if hit_at_5 else "✗"}</div>
            <div><b>Hit@10:</b> {"✓" if hit_at_10 else "✗"}</div>
            <div><b>MRR:</b> {mrr:.3f}</div>
        </div>
    '''

    # Add query text if provided
    if query_text:
        metrics_html += f'''
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #90caf9;">
            <b>Query:</b> {query_text}
        </div>
        '''

    # Add doc_references if provided
    if doc_references:
        refs_list = ''.join([f'<a href="{url}" target="_blank" style="color: #1565c0; margin-right: 10px;">{url}</a>' for url in doc_references])
        metrics_html += f'''
        <div style="margin-top: 8px;">
            <b>Expected Sources:</b> {refs_list}
        </div>
        '''

    metrics_html += '</div>'

    return widgets.HTML(value=metrics_html)
