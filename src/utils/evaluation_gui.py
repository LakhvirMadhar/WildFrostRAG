"""
Query Annotation GUI for WildFrostRAG.

This module provides an interactive graphical user interface for annotating and evaluating
RAG responses. It allows users to validate responses, perform qualitative coding (open and axial),
and assess retrieval results through a Jupyter notebook interface.
"""

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML
from dataclasses import dataclass
import time

@dataclass
class DisplayConfig:
    """Configuration for display font sizes"""
    query_label_size: int = 16
    query_text_size: int = 14
    ground_truth_label_size: int = 16
    ground_truth_text_size: int = 14
    doc_reference_label_size: int = 16
    doc_reference_text_size: int = 14
    openai_response_label_size: int = 16
    openai_response_text_size: int = 14
    chunk_text_size: int = 14

class QueryAnnotationGUI:
    # Define available versions and their column mappings
    VERSION_CONFIGS = {
        'zero_shot': {
            'response_column': 'openAI_zero_shot',
            'validation_column': 'openAI_zero_shot_validation',
            'open_coding_column': 'openAI_zero_shot Open Coding',
            'axial_coding_column': 'openAI_zero_shot Axial Coding',
            'display_name': 'Zero-Shot',
            'chunks_column': None  # No chunks for zero-shot
        },
        'rag': {
            'response_column': 'openAI_RAG_response',
            'validation_column': 'openAI_RAG_validation',
            'open_coding_column': 'openAI_RAG Open Coding',
            'axial_coding_column': 'openAI_RAG Axial Coding',
            'display_name': 'RAG',
            'chunks_column': 'retrieved_chunks'
        }
    }

    def __init__(self, filepath='queries/simple_reference_based_queries.csv', config: DisplayConfig = None):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.config = config if config else DisplayConfig()

        # Initialize columns for all versions
        for version_config in self.VERSION_CONFIGS.values():
            if version_config['validation_column'] not in self.df.columns:
                self.df[version_config['validation_column']] = ''
            if version_config['open_coding_column'] not in self.df.columns:
                self.df[version_config['open_coding_column']] = ''
            if version_config['axial_coding_column'] not in self.df.columns:
                self.df[version_config['axial_coding_column']] = ''

        # Get list of query IDs that have non-empty queries
        self.valid_query_ids = self.df[
            self.df['query'].notna() & (self.df['query'] != '')
        ]['query_id'].tolist()

        self.current_index = 0
        self.current_version = 'zero_shot'  # Default version

        # Create widgets
        self.create_widgets()
        self.setup_callbacks()
        self.update_display()

    def get_current_version_config(self):
        """Get the configuration for the currently selected version"""
        return self.VERSION_CONFIGS[self.current_version]

    def create_widgets(self):
        """Create all GUI widgets"""

        # Version selector dropdown
        version_options = {
            config['display_name']: key
            for key, config in self.VERSION_CONFIGS.items()
        }
        self.version_dropdown = widgets.Dropdown(
            options=list(version_options.keys()),
            value=list(version_options.keys())[0],
            description='Version:',
            layout=widgets.Layout(width='200px')
        )

        # Query ID dropdown
        self.dropdown = widgets.Dropdown(
            options=self.valid_query_ids,
            value=self.valid_query_ids[0] if self.valid_query_ids else None,
            description='Query ID:',
            layout=widgets.Layout(width='300px')
        )

        # Jump to next unvalidated button
        self.jump_unvalidated_button = widgets.Button(
            description='⇨ Next Unvalidated',
            button_style='primary',
            layout=widgets.Layout(width='180px')
        )

        # Navigation buttons
        self.prev_button = widgets.Button(
            description='◀ Previous',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )

        self.next_button = widgets.Button(
            description='Next ▶',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )

        # Progress label between buttons
        self.progress_label = widgets.HTML(
            value=f"<div style='text-align: center;'><b>Query 1 of {len(self.valid_query_ids)}</b></div>",
            layout=widgets.Layout(width='200px')
        )

        # Display widgets
        self.query_display = widgets.HTML(value='')
        self.ground_truth_display = widgets.HTML(value='')
        self.doc_reference_display = widgets.HTML(value='')
        self.openai_response_display = widgets.HTML(value='')

        # Metrics display (for retrieval metrics) - Changed to Output widget
        self.metrics_display = widgets.Output()

        # Pass/Fail/Clear buttons
        self.pass_button = widgets.Button(
            description='✓ Pass',
            button_style='success',
            layout=widgets.Layout(width='150px', height='40px')
        )

        self.fail_button = widgets.Button(
            description='✗ Fail',
            button_style='danger',
            layout=widgets.Layout(width='150px', height='40px')
        )

        self.clear_button = widgets.Button(
            description='Clear',
            button_style='warning',
            layout=widgets.Layout(width='150px', height='40px')
        )

        self.validation_status = widgets.HTML(value='')

        # Text inputs
        self.open_coding_input = widgets.Textarea(
            placeholder='Enter open coding notes...',
            description='Open Coding:',
            layout=widgets.Layout(width='100%', height='100px'),
            style={'description_width': '120px'}
        )

        self.axial_coding_input = widgets.Textarea(
            placeholder='Enter axial coding notes...',
            description='Axial Coding:',
            layout=widgets.Layout(width='100%', height='100px'),
            style={'description_width': '120px'}
        )

        # Save status indicator
        self.save_status = widgets.HTML(value='')

    def setup_callbacks(self):
        """Setup all event handlers"""
        self.version_dropdown.observe(self.on_version_change, names='value')
        self.dropdown.observe(self.on_dropdown_change, names='value')
        self.jump_unvalidated_button.on_click(self.on_jump_unvalidated_click)
        self.prev_button.on_click(self.on_prev_click)
        self.next_button.on_click(self.on_next_click)
        self.pass_button.on_click(self.on_pass_click)
        self.fail_button.on_click(self.on_fail_click)
        self.clear_button.on_click(self.on_clear_click)
        self.open_coding_input.observe(self.on_open_coding_change, names='value')
        self.axial_coding_input.observe(self.on_axial_coding_change, names='value')

    def on_version_change(self, change):
        """Handle version dropdown change"""
        # Map display name back to version key
        version_options = {
            config['display_name']: key
            for key, config in self.VERSION_CONFIGS.items()
        }
        self.current_version = version_options[change['new']]
        self.update_display()

    def on_chunk_relevance_change(self, chunk_idx, is_relevant):
        """Handle chunk relevance checkbox change"""
        row_idx = self.get_current_row_index()
        version_config = self.get_current_version_config()
        relevance_col = f"{version_config['response_column']}_chunk_relevance"

        # Load existing relevance dict
        existing_relevance = self.df.loc[row_idx, relevance_col] if pd.notna(self.df.loc[row_idx, relevance_col]) else '{}'
        try:
            import json
            relevance_dict = json.loads(existing_relevance) if existing_relevance else {}
        except:
            relevance_dict = {}

        # Update relevance for this chunk
        chunk_id = f"chunk_{chunk_idx}"
        relevance_dict[chunk_id] = is_relevant

        # Save back to dataframe
        self.df.at[row_idx, relevance_col] = json.dumps(relevance_dict)
        self.save_to_csv()

    def get_current_row(self):
        """Get the current row based on selected query_id"""
        query_id = self.dropdown.value
        return self.df[self.df['query_id'] == query_id].iloc[0]

    def get_current_row_index(self):
        """Get the DataFrame index for current query_id"""
        query_id = self.dropdown.value
        return self.df[self.df['query_id'] == query_id].index[0]

    def save_to_csv(self):
        """Save DataFrame to CSV"""
        self.df.to_csv(self.filepath, index=False)
        self.show_save_status()

    def show_save_status(self):
        """Show a temporary save status message"""
        self.save_status.value = '<span style="color: green;">✓ Saved</span>'

    def find_next_unvalidated(self):
        """Find the next unvalidated query starting from current position"""
        version_config = self.get_current_version_config()
        validation_col = version_config['validation_column']

        # Get queries that are unvalidated (empty or NA)
        unvalidated_mask = (
            self.df['query'].notna() &
            (self.df['query'] != '') &
            (self.df[validation_col].isna() | (self.df[validation_col] == ''))
        )
        unvalidated_query_ids = self.df[unvalidated_mask]['query_id'].tolist()

        if not unvalidated_query_ids:
            return None

        # Find next unvalidated after current position
        current_query_id = self.dropdown.value
        for query_id in unvalidated_query_ids:
            if query_id > current_query_id:
                return query_id

        # If none found after current, return first unvalidated (wrap around)
        return unvalidated_query_ids[0]

    def format_metrics_display(self, row):
        """Format and display retrieval metrics if available - returns widget container"""
        version_config = self.get_current_version_config()
        chunks_col = version_config.get('chunks_column')

        if not chunks_col or chunks_col not in row.index:
            return widgets.HTML(value='<div style="padding: 10px 0;"><i>No metrics available for this version</i></div>')

        chunks_json = row[chunks_col]
        if pd.isna(chunks_json) or chunks_json == '':
            return widgets.HTML(value='<div style="padding: 10px 0;"><i>No retrieval data available</i></div>')

        try:
            import json
            chunks = json.loads(chunks_json)

            if not chunks:
                return widgets.HTML(value='<div style="padding: 10px 0;"><i>No chunks retrieved</i></div>')

            # Get row index for storing relevance annotations
            row_idx = row.name if hasattr(row, 'name') else self.get_current_row_index()
            relevance_col = f"{version_config['response_column']}_chunk_relevance"

            # Initialize relevance column if needed
            if relevance_col not in self.df.columns:
                self.df[relevance_col] = ''

            # Load existing relevance annotations
            existing_relevance = self.df.loc[row_idx, relevance_col] if pd.notna(self.df.loc[row_idx, relevance_col]) else '{}'
            try:
                relevance_dict = json.loads(existing_relevance) if existing_relevance else {}
            except:
                relevance_dict = {}

            # Create chunk display widgets
            chunk_widgets = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{i}"
                is_relevant = relevance_dict.get(chunk_id, False)

                # Create checkbox
                checkbox = widgets.Checkbox(
                    value=is_relevant,
                    description='Relevant',
                    layout=widgets.Layout(width='auto')
                )

                # Create callback
                def make_callback(idx):
                    def on_change(change):
                        self.on_chunk_relevance_change(idx, change['new'])
                    return on_change

                checkbox.observe(make_callback(i), names='value')

                # Build metadata display with ALL available fields
                metadata_parts = []

                # Add URL if available (from source_file or url field)
                url = chunk.get('url') or chunk.get('source_file') or chunk.get('source')
                if url:
                    metadata_parts.append(f'<b>URL:</b> <a href="{url}" target="_blank">{url}</a>')

                # Add headers
                if chunk.get('header1'):
                    metadata_parts.append(f'<b>Header 1:</b> {chunk["header1"]}')
                if chunk.get('header2'):
                    metadata_parts.append(f'<b>Header 2:</b> {chunk["header2"]}')
                if chunk.get('header3'):
                    metadata_parts.append(f'<b>Header 3:</b> {chunk["header3"]}')

                # Add any other metadata fields (excluding text and score)
                for key, value in chunk.items():
                    if key not in ['text', 'score', 'url', 'source_file', 'source', 'header1', 'header2', 'header3'] and value:
                        metadata_parts.append(f'<b>{key}:</b> {value}')

                metadata_html = '<div style="margin-bottom: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 13px; line-height: 1.6;">' + '<br>'.join(metadata_parts) + '</div>' if metadata_parts else ''

                # Create text display (simple like query/response display)
                text_display = widgets.HTML(
                    value=f"""
                    <div style="padding: 10px 0;">
                        <b>Full Text:</b><br>
                        <span style="font-size: {self.config.chunk_text_size}px; line-height: 1.5;">{chunk.get('text', 'No text available')}</span>
                    </div>
                    """
                )

                # Create toggle button for expand/collapse
                toggle_button = widgets.Button(
                    description='▲ Collapse',
                    button_style='',
                    layout=widgets.Layout(width='120px', height='30px')
                )

                # Create content container (initially visible)
                content_container = widgets.VBox([
                    widgets.HTML(value=metadata_html) if metadata_html else widgets.HTML(value=''),
                    text_display
                ], layout=widgets.Layout(display='block'))

                # Create chunk header with title and checkbox
                chunk_header = widgets.HBox([
                    widgets.HTML(value=f'<b style="font-size: 14px;">Chunk {i+1}</b>'),
                    toggle_button,
                    checkbox
                ], layout=widgets.Layout(justify_content='flex-start', align_items='center', margin='0 0 5px 0'))

                # Score on its own line
                score_display = widgets.HTML(
                    value=f'<div style="padding: 5px 0; margin-bottom: 10px;"><span style="padding: 5px 10px; background-color: #e3f2fd; border-radius: 3px; font-weight: bold; color: #1976d2;">Score: {chunk.get("score", 0):.4f}</span></div>'
                )

                # Create toggle callback
                def make_toggle_callback(btn, container):
                    def on_toggle(b):
                        if container.layout.display == 'none':
                            container.layout.display = 'block'
                            btn.description = '▲ Collapse'
                        else:
                            container.layout.display = 'none'
                            btn.description = '▼ Expand'
                    return on_toggle

                toggle_button.on_click(make_toggle_callback(toggle_button, content_container))

                # Create chunk container
                chunk_container = widgets.VBox([
                    chunk_header,
                    score_display,
                    content_container
                ], layout=widgets.Layout(
                    margin='10px 0',
                    padding='15px',
                    border='2px solid #2196F3',
                    border_radius='8px',
                    background_color='#fafafa'
                ))

                chunk_widgets.append(chunk_container)

            # Create summary header
            summary_html = widgets.HTML(
                value=f"""
                <div style="padding: 15px; background-color: #e8f5e9; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="margin: 0 0 10px 0; color: #2e7d32;">Retrieval Summary</h3>
                    <p style="margin: 0; font-size: 14px;"><b>Total chunks retrieved:</b> {len(chunks)}</p>
                </div>
                """
            )

            # Coding notes section for metrics view
            coding_notes_html = widgets.HTML(value='<h3 style="margin: 20px 0 10px 0;">Coding Notes</h3>')

            # Create main metrics container
            metrics_container = widgets.VBox([
                summary_html,
                coding_notes_html,
                self.open_coding_input,
                widgets.HTML(value='<div style="height: 10px;"></div>'),
                self.axial_coding_input,
                widgets.HTML(value='<h4 style="margin: 30px 0 10px 0;">Retrieved Chunks:</h4>'),
                widgets.VBox(chunk_widgets)
            ])

            return metrics_container

        except Exception as e:
            from .logger import logger
            logger.error(f"Error parsing metrics: {str(e)}")
            return widgets.HTML(value=f'<div style="padding: 10px; color: red; background-color: #ffebee; border-radius: 5px;">Error parsing metrics. Please check the logs for details.</div>')

    def update_display(self):
        """Update all display widgets based on current selection"""
        row = self.get_current_row()
        query_id = self.dropdown.value
        version_config = self.get_current_version_config()

        # Update navigation
        self.current_index = self.valid_query_ids.index(query_id)
        self.progress_label.value = f"<div style='text-align: center;'><b>Query {self.current_index + 1} of {len(self.valid_query_ids)}</b></div>"

        # Update prev/next button states
        self.prev_button.disabled = (self.current_index == 0)
        self.next_button.disabled = (self.current_index == len(self.valid_query_ids) - 1)

        # Check if there are any unvalidated queries
        next_unvalidated = self.find_next_unvalidated()
        self.jump_unvalidated_button.disabled = (next_unvalidated is None)

        # Display query information
        query = row['query'] if pd.notna(row['query']) else '<i>No query</i>'
        self.query_display.value = (
            f'<div style="padding: 10px 0;">'
            f'<b style="font-size: {self.config.query_label_size}px;">Query:</b><br>'
            f'<span style="font-size: {self.config.query_text_size}px;">{query}</span>'
            f'</div>'
        )

        ground_truth = row['ground_truth'] if pd.notna(row['ground_truth']) else '<i>No ground truth</i>'
        self.ground_truth_display.value = (
            f'<div style="padding: 10px 0;">'
            f'<b style="font-size: {self.config.ground_truth_label_size}px;">Ground Truth:</b><br>'
            f'<span style="font-size: {self.config.ground_truth_text_size}px;">{ground_truth}</span>'
            f'</div>'
        )

        doc_ref = row['doc_reference'] if pd.notna(row['doc_reference']) else ''
        if doc_ref:
            self.doc_reference_display.value = (
                f'<div style="padding: 10px 0;">'
                f'<b style="font-size: {self.config.doc_reference_label_size}px;">Document Reference:</b><br>'
                f'<a href="{doc_ref}" target="_blank" style="font-size: {self.config.doc_reference_text_size}px;">{doc_ref}</a>'
                f'</div>'
            )
        else:
            self.doc_reference_display.value = (
                f'<div style="padding: 10px 0;">'
                f'<b style="font-size: {self.config.doc_reference_label_size}px;">Document Reference:</b><br>'
                f'<span style="font-size: {self.config.doc_reference_text_size}px;"><i>No reference</i></span>'
                f'</div>'
            )

        # Display response based on selected version
        response_col = version_config['response_column']
        response = row[response_col] if pd.notna(row[response_col]) and response_col in row.index else '<i>No response yet</i>'
        display_name = version_config['display_name']
        self.openai_response_display.value = (
            f'<div style="padding: 10px; background-color: #e8f4f8; border-radius: 5px; border-left: 4px solid #2196F3;">'
            f'<b style="font-size: {self.config.openai_response_label_size}px;">OpenAI {display_name} Response:</b><br>'
            f'<span style="font-size: {self.config.openai_response_text_size}px;">{response}</span>'
            f'</div>'
        )

        # Display metrics if available - Updated to use Output widget
        self.metrics_display.clear_output()
        with self.metrics_display:
            metrics_widget = self.format_metrics_display(row)
            display(metrics_widget)

        # Update validation status
        validation_col = version_config['validation_column']
        validation = row[validation_col] if validation_col in row.index else ''
        if pd.notna(validation) and validation != '':
            if validation.lower() == 'pass':
                self.validation_status.value = '<span style="color: green; font-size: 16px; font-weight: bold;">✓ PASS</span>'
            else:
                self.validation_status.value = '<span style="color: red; font-size: 16px; font-weight: bold;">✗ FAIL</span>'
        else:
            self.validation_status.value = '<span style="color: gray; font-style: italic;">Not validated yet</span>'

        # Update text inputs (without triggering save)
        self.open_coding_input.unobserve(self.on_open_coding_change, names='value')
        self.axial_coding_input.unobserve(self.on_axial_coding_change, names='value')

        open_coding_col = version_config['open_coding_column']
        axial_coding_col = version_config['axial_coding_column']
        self.open_coding_input.value = row[open_coding_col] if open_coding_col in row.index and pd.notna(row[open_coding_col]) else ''
        self.axial_coding_input.value = row[axial_coding_col] if axial_coding_col in row.index and pd.notna(row[axial_coding_col]) else ''

        self.open_coding_input.observe(self.on_open_coding_change, names='value')
        self.axial_coding_input.observe(self.on_axial_coding_change, names='value')

        # Clear save status when switching queries
        self.save_status.value = ''

    def on_dropdown_change(self, change):
        """Handle dropdown selection change"""
        self.update_display()

    def on_jump_unvalidated_click(self, button):
        """Handle jump to next unvalidated button click"""
        next_unvalidated = self.find_next_unvalidated()
        if next_unvalidated is not None:
            self.dropdown.value = next_unvalidated

    def on_prev_click(self, button):
        """Handle previous button click"""
        if self.current_index > 0:
            self.dropdown.value = self.valid_query_ids[self.current_index - 1]

    def on_next_click(self, button):
        """Handle next button click"""
        if self.current_index < len(self.valid_query_ids) - 1:
            self.dropdown.value = self.valid_query_ids[self.current_index + 1]

    def on_pass_click(self, button):
        """Handle pass button click"""
        idx = self.get_current_row_index()
        version_config = self.get_current_version_config()
        self.df.at[idx, version_config['validation_column']] = 'Pass'
        self.save_to_csv()
        self.update_display()

    def on_fail_click(self, button):
        """Handle fail button click"""
        idx = self.get_current_row_index()
        version_config = self.get_current_version_config()
        self.df.at[idx, version_config['validation_column']] = 'Fail'
        self.save_to_csv()
        self.update_display()

    def on_clear_click(self, button):
        """Handle clear button click"""
        idx = self.get_current_row_index()
        version_config = self.get_current_version_config()
        self.df.at[idx, version_config['validation_column']] = ''
        self.save_to_csv()
        self.update_display()

    def on_open_coding_change(self, change):
        """Handle open coding text change - auto-save"""
        idx = self.get_current_row_index()
        version_config = self.get_current_version_config()
        self.df.at[idx, version_config['open_coding_column']] = change['new']
        self.save_to_csv()

    def on_axial_coding_change(self, change):
        """Handle axial coding text change - auto-save"""
        idx = self.get_current_row_index()
        version_config = self.get_current_version_config()
        self.df.at[idx, version_config['axial_coding_column']] = change['new']
        self.save_to_csv()

    def display(self):
        """Display the complete GUI with tabs"""

        # Version selector and query selector row
        version_query_box = widgets.HBox([
            self.version_dropdown,
            self.dropdown,
            self.jump_unvalidated_button
        ], layout=widgets.Layout(justify_content='flex-start', margin='10px 0'))

        # Navigation row (centered with label in middle)
        nav_box = widgets.HBox([
            self.prev_button,
            self.progress_label,
            self.next_button
        ], layout=widgets.Layout(justify_content='center', margin='10px 0'))

        # Query information section (for main tab)
        info_box = widgets.VBox([
            self.query_display,
            widgets.HTML(value='<div style="height: 5px;"></div>'),
            self.ground_truth_display,
            widgets.HTML(value='<div style="height: 5px;"></div>'),
            self.doc_reference_display,
            widgets.HTML(value='<div style="height: 10px;"></div>'),
            self.openai_response_display
        ])

        # Validation section (centered)
        validation_buttons = widgets.HBox([
            self.pass_button,
            self.fail_button,
            self.clear_button,
            self.validation_status,
            self.save_status
        ], layout=widgets.Layout(justify_content='center', margin='20px 0'))

        # Coding inputs section
        coding_box = widgets.VBox([
            widgets.HTML(value='<h3>Coding Notes</h3>'),
            self.open_coding_input,
            widgets.HTML(value='<div style="height: 10px;"></div>'),
            self.axial_coding_input
        ])

        # Main tab content
        main_tab_content = widgets.VBox([
            info_box,
            widgets.HTML(value='<hr>'),
            validation_buttons,
            coding_box
        ])

        # Retrieval metrics tab content (coding notes now right after summary)
        metrics_tab_content = widgets.VBox([
            self.metrics_display
        ], layout=widgets.Layout(padding='10px'))

        # Create tabs
        tab = widgets.Tab()
        tab.children = [main_tab_content, metrics_tab_content]
        tab.set_title(0, 'Query & Response')
        tab.set_title(1, 'Retrieval Metrics')

        # Main container
        main_container = widgets.VBox([
            widgets.HTML(value='<h2>Query Annotation Tool</h2>'),
            version_query_box,
            nav_box,
            widgets.HTML(value='<hr>'),
            tab
        ], layout=widgets.Layout(padding='20px'))

        display(main_container)