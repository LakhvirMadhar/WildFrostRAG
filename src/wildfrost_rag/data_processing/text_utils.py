"""Shared text extraction utilities for WildFrostRAG data processing."""

import re

from bs4 import Tag


def clean_element_text(element: Tag) -> str:
    """Extract clean text from a BeautifulSoup element.

    Uses space separator to preserve gaps between inline elements (links, spans),
    then collapses multiple whitespace characters into a single space.

    Args:
        element: A BeautifulSoup Tag (table cell, paragraph, etc.)

    Returns:
        Cleaned text string with normalized whitespace.
    """
    text = element.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()
