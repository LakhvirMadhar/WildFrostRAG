import os
import json
from typing import Dict, Any, Tuple
from .logger import logger


def save_to_json(filename_path: str, data: Dict[str, Any]):
    """
    Saves a dictionary to a JSON file, creating the directory if needed.

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
    expected_prefix = os.path.abspath('.')
    if not full_path.startswith(expected_prefix):
        logger.error(f"Invalid path detected: {filename_path}")
        return

    # Create directory if it doesn't exist
    directory = os.path.dirname(safe_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(safe_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            logger.info(f"Saved data to: {safe_path}")
    except IOError as e:
        logger.error(f"Error: Unable to save data to {safe_path}. Error: {e}")


def format_prompt_tuple(prompt_tuple: Tuple[str, ...], **kwargs) -> str:
    """
    Format a prompt tuple with provided keyword arguments.

    Prompt tuples follow the pattern: (template_string, param1, param2, ...)
    where template_string contains {param1}, {param2}, etc. placeholders.

    Args:
        prompt_tuple: Tuple where first element is template string,
                     remaining elements are expected parameter names
        **kwargs: Keyword arguments to fill in the template

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If required parameters are missing

    Example:
        >>> PROMPT = ("Hello {name}, you are {age} years old", "name", "age")
        >>> format_prompt_tuple(PROMPT, name="Alice", age=30)
        "Hello Alice, you are 30 years old"
    """
    template = prompt_tuple[0]
    expected_params = prompt_tuple[1:]

    # Validate that all expected parameters are provided
    missing_params = [param for param in expected_params if param not in kwargs]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")

    # Format the template with provided kwargs
    return template.format(**kwargs)