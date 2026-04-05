import re

from langchain_text_splitters.html import HTMLHeaderTextSplitter
from bs4 import BeautifulSoup, Tag
from typing import List
from langchain_core.documents import Document
from tqdm import tqdm

from data_processing.text_utils import clean_element_text

# Elements to remove from HTML before processing
UNWANTED_SELECTORS = [
    'navbox',          # Custom navboxes
    'printfooter',     # "Retrieved from..."
    'catlinks',        # "Categories: ..."
    'mw-navigation',   # "Navigation menu", "Personal tools", etc.
    'mw-head',         # Top header/search
    'mw-panel',        # Side panel
    'footer',          # "Privacy policy", "About", etc.
    'mw-editsection',  # "Edit" links
    'toc',             # Table of Contents
    'script',          # JavaScript blocks
    'siteSub',         # "From Wildfrost Wiki"
    'jump-to-nav',     # Empty jump-to-nav container
    'mw-jump-link',    # "Jump to navigation" / "Jump to search"
    'mw-collapsible',  # Collapsible card art containers (image-only)
    'head',            # <head> tag (title, meta, scripts — no body text)
]


_get_cell_text = clean_element_text


def _convert_infobox_to_text(table: Tag, line_sep: str = '\n') -> str:
    """
    Convert a card infobox (vertical key-value table) into readable text.

    Handles two patterns:
    - Single-cell rows: th label followed by td value -> "Label: Value"
    - Multi-cell stat rows: th|th|th followed by td|td|td -> "Health: 2 | Attack: 1 | Counter: 4"
    """
    rows = table.find_all('tr', recursive=False)
    if not rows:
        # Try inside tbody
        tbody = table.find('tbody')
        rows = tbody.find_all('tr', recursive=False) if tbody else []

    lines = []
    i = 0
    while i < len(rows):
        row = rows[i]
        headers = row.find_all('th', recursive=False)
        cells = row.find_all('td', recursive=False)

        # Multi-column header row (e.g., Health | Attack | Counter)
        # Look ahead to pair with the next row's values
        if len(headers) > 1 and not cells:
            header_texts = [_get_cell_text(h) for h in headers]
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                value_cells = next_row.find_all('td', recursive=False)
                value_texts = [_get_cell_text(v) for v in value_cells]
                pairs = []
                for h, v in zip(header_texts, value_texts):
                    if h and v:
                        pairs.append(f"{h}: {v}")
                if pairs:
                    lines.append(' | '.join(pairs))
                i += 2
                continue

        # Single header cell - pair with next row's value
        if len(headers) == 1 and not cells:
            label = _get_cell_text(headers[0])
            if label and i + 1 < len(rows):
                next_row = rows[i + 1]
                value_cells = next_row.find_all('td', recursive=False)
                if value_cells:
                    value = _get_cell_text(value_cells[0])
                    if label and value:
                        lines.append(f"{label}: {value}")
                    elif label:
                        lines.append(label)
                    i += 2
                    continue

        # Fallback: just output whatever text is in the row
        text = _get_cell_text(row)
        if text:
            lines.append(text)
        i += 1

    return line_sep.join(lines)


def _convert_data_table_to_text(table: Tag, line_sep: str = '\n') -> str:
    """
    Convert a multi-column data table (like Charm list) into readable text.

    Header row separated from data rows by a blank line, cells joined by " | ".
    """
    rows = table.find_all('tr')
    if not rows:
        return ''

    lines = []
    for i, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        cell_texts = [_get_cell_text(c) for c in cells]
        cell_texts = [t for t in cell_texts if t]

        if cell_texts:
            lines.append(' | '.join(cell_texts))

        # Add blank line after header row
        if i == 0 and row.find('th'):
            lines.append('')

    return line_sep.join(lines)


def _is_infobox(table: Tag) -> bool:
    """Detect if a table is a card infobox (vertical key-value layout)."""
    return table.get('id') == 'infobox'


def _convert_table_to_text(table: Tag, line_sep: str = '\n') -> str:
    """
    Convert an HTML table into readable structured text.

    Detects card infoboxes vs data tables and formats accordingly.
    """
    if _is_infobox(table):
        return _convert_infobox_to_text(table, line_sep)
    return _convert_data_table_to_text(table, line_sep)


def _extract_clean_text(soup: BeautifulSoup) -> str:
    """
    Extract clean text from parsed HTML, preserving structure for tables,
    lists, paragraphs, and headings.
    """
    # Block marker: survives the newline-to-space replacement later.
    # All intentional line breaks use this; raw \n is a pretty-print artifact.
    BM = '\u2029'
    BM2 = BM + BM  # blank line separator

    # Move all infoboxes right before the first <h2> so intro text comes first.
    # Wiki pages float infoboxes right; in plain text we want:
    #   # Card Name
    #   Card is a Clunker that can be obtained...  (intro paragraphs)
    #   <all infobox fields>                       (stats for each phase)
    #   ## Strategy
    infoboxes = soup.find_all('table', id='infobox')
    if infoboxes:
        first_h2 = soup.find('h2')
        if first_h2:
            for ib in reversed(infoboxes):
                first_h2.insert_before(ib)

    # Convert tables to structured text (using block marker for line breaks)
    # Add blank line after each table so it doesn't run into the next paragraph.
    for table in soup.find_all('table'):
        table_text = _convert_table_to_text(table, line_sep=BM)
        table.replace_with(BM + table_text + BM2)

    # Remove images (icon PNGs aren't useful as text)
    for img in soup.find_all('img'):
        img.decompose()

    # Flatten inline elements: replace each tag with its stripped text
    # so pretty-printed newlines inside <a>\n  Clunker\n</a> collapse
    # to just "Clunker" flowing inline with the paragraph.
    from bs4 import NavigableString
    for tag in soup.find_all(['a', 'span', 'b', 'i', 'strong', 'em', 'small', 'code']):
        text_content = tag.get_text(separator=' ', strip=True)
        tag.replace_with(NavigableString(f' {text_content} '))

    # Convert headers to markdown and mark block boundaries
    header_md = {'h1': '# ', 'h2': '## ', 'h3': '### ', 'h4': '#### '}
    for tag_name, prefix in header_md.items():
        for tag in soup.find_all(tag_name):
            heading_text = tag.get_text(separator=' ', strip=True)
            tag.replace_with(NavigableString(f'{BM2}{prefix}{heading_text}{BM2}'))

    for tag_name in ['p', 'li', 'br']:
        for tag in soup.find_all(tag_name):
            tag.append(BM)

    text = soup.get_text()

    # Replace all raw newlines (pretty-print artifacts) with spaces
    text = text.replace('\n', ' ')
    # Restore intentional line breaks from block markers
    text = text.replace(BM, '\n')
    # Collapse whitespace within lines
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Remove spaces before punctuation (artifact of inline flattening)
    text = re.sub(r' ([.,;:!?])', r'\1', text)
    # Strip each line, collapse multiple blank lines into one
    lines = [line.strip() for line in text.splitlines()]
    result_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                result_lines.append('')
            prev_empty = True
        else:
            result_lines.append(line)
            prev_empty = False
    text = '\n'.join(result_lines).strip()

    return text


def process_html_files(filepaths: List[str], split_text: bool = True) -> List[Document]:
    """
    Parses and optionally chunks HTML content from a list of files.

    Args:
        filepaths: A list of string paths to the HTML files to be processed.
        split_text: If True, splits text by headers. If False, returns full content.

    Returns:
        A list of LangChain Document objects.
    """
    all_chunks = []

    # Define the headers to split on.
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]

    # Create the HTMLHeaderTextSplitter instance once.
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Use tqdm for progress bar
    for filepath in tqdm(filepaths, desc="Processing HTML files", unit="file"):
        try:
            # Load the raw HTML content from the file.
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Use BeautifulSoup for preprocessing to remove noise.
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # --- Preprocessing Step: Remove Unwanted Elements ---
            
            # Remove by class, id, or tag name
            for selector in UNWANTED_SELECTORS:
                # Remove by class
                for element in soup.find_all(class_=selector):
                    element.decompose()
                # Remove by ID
                for element in soup.find_all(id=selector):
                    element.decompose()
                # Remove by tag name (e.g. footer)
                for element in soup.find_all(selector):
                    element.decompose()

            # Convert the modified BeautifulSoup object back to a string for the splitter.
            cleaned_html = str(soup)

            if split_text:
                # Split the cleaned document based on the headers and add to the list.
                text_chunks = html_splitter.split_text(cleaned_html)
            else:
                # Create a single document with readable structure preserved
                full_text = _extract_clean_text(soup)
                text_chunks = [Document(page_content=full_text)]
            
            # Add source filepath to metadata for each chunk
            for chunk in text_chunks:
                chunk.metadata['source'] = filepath
            
            all_chunks.extend(text_chunks)
            

        except FileNotFoundError:
            tqdm.write(f"Error: The file '{filepath}' was not found.")
        except Exception as e:
            tqdm.write(f"An error occurred while processing '{filepath}': {e}")

    return all_chunks


if __name__ == '__main__':
    # Example usage of the function with a list of file paths.
    # Note: The file 'data/structured_outputs/items/Azul Battle Axe.html' must exist for this to run.
    sample_filepaths = ['data/structured_outputs/items/Azul Battle Axe.html', 'data/structured_outputs/items/Azul Candle.html']
    
    # Process the files and get all the chunks.
    all_document_chunks = process_html_files(sample_filepaths)

    # Print a summary of all chunks from all files.
    if all_document_chunks:
        print("\n--- All Chunks from All Files ---")
        for i, chunk in enumerate(all_document_chunks):
            print(f"Chunk {i+1}: '{chunk.page_content}...'")
            print(f"Metadata: {chunk.metadata}")
            print("-" * 20)
        print(f"\nTotal chunks returned: {len(all_document_chunks)}")
    else:
        print("\nNo chunks were created.")