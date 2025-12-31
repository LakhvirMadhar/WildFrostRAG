from langchain_text_splitters.html import HTMLHeaderTextSplitter
from bs4 import BeautifulSoup
from typing import List
from langchain_core.documents import Document

from tqdm import tqdm

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
]


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
                # Create a single document for the whole file, but extract text first
                # using get_text with a separator to preserve some structure
                full_text = soup.get_text(separator='\n\n', strip=True)
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