import os
import json
from typing import Any
from .logger import logger


def save_to_json(filename_path: str, data: dict[str, Any]) -> None:
    """Saves a dictionary to a JSON file, creating the directory if needed.

    Args:
        filename_path (str): The full path to the file to save (e.g., 'data/output.json').
        data (Dict[str, Any]): The dictionary to be saved.
    """
    # Sanitize the filename path to prevent directory traversal
    safe_path = os.path.normpath(filename_path)

    # Ensure the filename is within the expected directory structure
    # Prevent paths that go outside the current directory
    if os.path.isabs(filename_path):
        # If it's an absolute path, reject it
        logger.error(f"Absolute path not allowed: {filename_path}")
        return

    # Check if the resolved path goes outside the intended directory
    full_path = os.path.abspath(safe_path)
    expected_prefix = os.path.abspath(".")
    if not full_path.startswith(expected_prefix):
        logger.error(f"Invalid path detected: {filename_path}")
        return

    # Create directory if it doesn't exist
    directory = os.path.dirname(safe_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            logger.info(f"Saved data to: {safe_path}")
    except OSError as e:
        logger.error(f"Error: Unable to save data to {safe_path}. Error: {e}")
