import os
import json
from typing import Dict, Any
from .logger import logger


def save_to_json(filename_path: str, data: Dict[str, Any]):
    """
    Saves a dictionary to a JSON file, creating the directory if needed.

    Args:
        filename_path (str): The full path to the file to save (e.g., 'data/output.json').
        data (Dict[str, Any]): The dictionary to be saved.
    """

    # os.makedirs(filename_path, exist_ok=True)
    # logger.info(f"Directory created or already exists: {filename_path}")

    try:
        with open(filename_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            logger.info(f"Saved data to: {filename_path}")
    except IOError as e:
        logger.error(f"Error: Unable to save data to {filename_path}. Error: {e}")


def generate_directories():
    """
    Generate the directories needed to run the program
    """
    dirs = ['data/html_output']

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        logger.info(f"Created or directory already exists: {dir}")