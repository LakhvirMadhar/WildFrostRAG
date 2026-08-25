import re

from langchain_text_splitters.html import HTMLHeaderTextSplitter
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from langchain_core.documents import Document
from tqdm import tqdm

from wildfrost_rag.data_processing.text_utils import clean_element_text
from wildfrost_rag.utils.logger import logger

# Block markers for text extraction — survive newline-to-space replacement.
# All intentional line breaks use these; raw \n is a pretty-print artifact.
BLOCK_MARKER = "\u2029"
BLOCK_MARKER_DOUBLE = BLOCK_MARKER + BLOCK_MARKER

# Elements to remove from HTML before processing
UNWANTED_SELECTORS = [
    "navbox",  # Custom navboxes
    "printfooter",  # "Retrieved from..."
    "catlinks",  # "Categories: ..."
    "mw-navigation",  # "Navigation menu", "Personal tools", etc.
    "mw-head",  # Top header/search
    "mw-panel",  # Side panel
    "footer",  # "Privacy policy", "About", etc.
    "mw-editsection",  # "Edit" links
    "toc",  # Table of Contents
    "script",  # JavaScript blocks
    "siteSub",  # "From Wildfrost Wiki"
    "jump-to-nav",  # Empty jump-to-nav container
    "mw-jump-link",  # "Jump to navigation" / "Jump to search"
    "mw-collapsible",  # Collapsible card art containers (image-only)
    "head",  # <head> tag (title, meta, scripts — no body text)
]


def _get_infobox_rows(table: Tag) -> list[Tag]:
    """Get table rows from an infobox, checking tbody if needed."""
    rows: list[Tag] = list(table.find_all("tr", recursive=False))
    if not rows:
        tbody = table.find("tbody")
        rows = list(tbody.find_all("tr", recursive=False)) if tbody else []
    return rows


def _parse_multi_column_row(headers: list[Tag], next_row: Tag) -> str | None:
    """Parse a multi-column header+value pair (e.g., Health: 2 | Attack: 1)."""
    header_texts = [clean_element_text(h) for h in headers]
    value_cells = next_row.find_all("td", recursive=False)
    value_texts = [clean_element_text(v) for v in value_cells]
    pairs = [f"{h}: {v}" for h, v in zip(header_texts, value_texts, strict=False) if h and v]
    return " | ".join(pairs) if pairs else None


def _parse_single_header_row(header: Tag, next_row: Tag) -> str | None:
    """Parse a single header followed by a value row (e.g., 'Tribe: Snowdwellers')."""
    label = clean_element_text(header)
    if not label:
        return None
    value_cells = next_row.find_all("td", recursive=False)
    if not value_cells:
        return None
    value = clean_element_text(value_cells[0])
    if label and value:
        return f"{label}: {value}"
    return label if label else None


def _convert_infobox_to_text(table: Tag, line_sep: str = "\n") -> str:
    """Convert a card infobox (vertical key-value table) into readable text.

    Handles two patterns:
    - Single-cell rows: th label followed by td value -> "Label: Value"
    - Multi-cell stat rows: th|th|th followed by td|td|td -> "Health: 2 | Attack: 1 | Counter: 4"
    """
    rows = _get_infobox_rows(table)
    lines = []
    i = 0
    while i < len(rows):
        row = rows[i]
        headers = row.find_all("th", recursive=False)
        cells = row.find_all("td", recursive=False)

        # Multi-column header row — look ahead to pair with next row's values
        if len(headers) > 1 and not cells and i + 1 < len(rows):
            result = _parse_multi_column_row(headers, rows[i + 1])
            if result:
                lines.append(result)
            i += 2
            continue

        # Single header cell — pair with next row's value
        if len(headers) == 1 and not cells and i + 1 < len(rows):
            result = _parse_single_header_row(headers[0], rows[i + 1])
            if result:
                lines.append(result)
                i += 2
                continue

        # Fallback: output whatever text is in the row
        text = clean_element_text(row)
        if text:
            lines.append(text)
        i += 1

    return line_sep.join(lines)


def _convert_data_table_to_text(table: Tag, line_sep: str = "\n") -> str:
    """Convert a multi-column data table (like Charm list) into readable text.

    Header row separated from data rows by a blank line, cells joined by " | ".
    """
    rows = table.find_all("tr")
    if not rows:
        return ""

    lines = []
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        cell_texts = [clean_element_text(c) for c in cells]
        cell_texts = [t for t in cell_texts if t]

        if cell_texts:
            lines.append(" | ".join(cell_texts))

        # Add blank line after header row
        if i == 0 and row.find("th"):
            lines.append("")

    return line_sep.join(lines)


def _is_infobox(table: Tag) -> bool:
    """Detect if a table is a card infobox (vertical key-value layout)."""
    return table.get("id") == "infobox"


def _convert_table_to_text(table: Tag, line_sep: str = "\n") -> str:
    """Convert an HTML table into readable structured text.

    Detects card infoboxes vs data tables and formats accordingly.
    """
    if _is_infobox(table):
        return _convert_infobox_to_text(table, line_sep)
    return _convert_data_table_to_text(table, line_sep)


def _reorder_infoboxes(soup: BeautifulSoup) -> None:
    """Move infoboxes before the first h2 so intro text comes first."""
    infoboxes = soup.find_all("table", id="infobox")
    if infoboxes:
        first_h2 = soup.find("h2")
        if first_h2:
            for ib in reversed(infoboxes):
                first_h2.insert_before(ib)


def _transform_soup_elements(soup: BeautifulSoup) -> None:
    """Transform HTML elements into text-friendly representations in place."""
    # Convert tables to structured text
    for table in soup.find_all("table"):
        table_text = _convert_table_to_text(table, line_sep=BLOCK_MARKER)
        table.replace_with(BLOCK_MARKER + table_text + BLOCK_MARKER_DOUBLE)

    # Remove images
    for img in soup.find_all("img"):
        img.decompose()

    # Flatten inline elements to plain text
    for tag in soup.find_all(["a", "span", "b", "i", "strong", "em", "small", "code"]):
        text_content = tag.get_text(separator=" ", strip=True)
        tag.replace_with(NavigableString(f" {text_content} "))

    # Convert headers to markdown
    header_md = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}
    for tag_name, prefix in header_md.items():
        for tag in soup.find_all(tag_name):
            heading_text = tag.get_text(separator=" ", strip=True)
            tag.replace_with(
                NavigableString(f"{BLOCK_MARKER_DOUBLE}{prefix}{heading_text}{BLOCK_MARKER_DOUBLE}")
            )

    # Add block markers after paragraphs, list items, line breaks
    for tag_name in ["p", "li", "br"]:
        for tag in soup.find_all(tag_name):
            tag.append(BLOCK_MARKER)


def _clean_raw_text(text: str) -> str:
    """Clean raw extracted text — normalize whitespace and collapse blank lines."""
    # Replace pretty-print newlines with spaces, restore intentional block markers
    text = text.replace("\n", " ")
    text = text.replace(BLOCK_MARKER, "\n")
    # Collapse whitespace within lines
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove spaces before punctuation
    text = re.sub(r" ([.,;:!?])", r"\1", text)
    # Strip each line, collapse multiple blank lines into one
    lines = [line.strip() for line in text.splitlines()]
    result_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                result_lines.append("")
            prev_empty = True
        else:
            result_lines.append(line)
            prev_empty = False
    return "\n".join(result_lines).strip()


def _extract_clean_text(soup: BeautifulSoup) -> str:
    """Extract clean text from parsed HTML.

    Preserves structure for tables, lists, paragraphs, and headings.
    """
    _reorder_infoboxes(soup)
    _transform_soup_elements(soup)
    return _clean_raw_text(soup.get_text())


def process_html_files(filepaths: list[str], split_text: bool = True) -> list[Document]:
    """Parses and optionally chunks HTML content from a list of files.

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
            with open(filepath, encoding="utf-8") as f:
                html_content = f.read()

            # Use BeautifulSoup for preprocessing to remove noise.
            soup = BeautifulSoup(html_content, "html.parser")

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
                chunk.metadata["source"] = filepath

            all_chunks.extend(text_chunks)

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
        except Exception as e:
            logger.error(f"Error processing '{filepath}': {e}")

    return all_chunks
