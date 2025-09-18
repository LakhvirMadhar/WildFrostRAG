from langchain.text_splitter import HTMLHeaderTextSplitter
from bs4 import BeautifulSoup
from typing import List
from langchain.docstore.document import Document


def process_html_files(filepaths: List[str]) -> List[Document]:
    """
    Parses and chunks HTML content from a list of files.

    This function preprocesses HTML to remove a specific navbox element
    before using LangChain's HTMLHeaderTextSplitter to create structured text chunks.

    Args:
        filepaths: A list of string paths to the HTML files to be processed.

    Returns:
        A list of LangChain Document objects, where each document is a chunk
        of the original HTML content.
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

    for filepath in filepaths:
        try:
            # Load the raw HTML content from the file.
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Use BeautifulSoup for preprocessing to remove noise.
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Define a list of classes for unwanted tables or divs.
            unwanted_classes = ['']
            
            # --- Preprocessing Step: Remove Unwanted Elements ---
            
            # First, find and remove the element with the specific ID.
            navbox_table = soup.find(id='navbox')
            if navbox_table:
                navbox_table.decompose()
                
            # Then, iterate over the list of unwanted classes and remove those elements.
            for unwanted_class in unwanted_classes:
                for element in soup.find_all(class_=unwanted_class):
                    element.decompose()

            # Convert the modified BeautifulSoup object back to a string for the splitter.
            cleaned_html = str(soup)

            # Split the cleaned document based on the headers and add to the list.
            text_chunks = html_splitter.split_text(cleaned_html)
            all_chunks.extend(text_chunks)
            
            print(f"Processed '{filepath}': {len(text_chunks)} chunks created.")

        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
        except Exception as e:
            print(f"An error occurred while processing '{filepath}': {e}")

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