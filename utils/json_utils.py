"""
Utility functions for consistent JSON handling across the project.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Standardized JSON indent for all project output
JSON_INDENT = 2

def save_json(data, filepath, ensure_dir=True):
    """
    Save data to a JSON file with standardized formatting.
    
    Args:
        data: The data structure to save as JSON
        filepath: Path to the output JSON file
        ensure_dir: Whether to create parent directories if they don't exist
    
    Returns:
        bool: Success status
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=JSON_INDENT, ensure_ascii=False)
            
        logger.info(f"Successfully saved JSON data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON data to {filepath}: {e}")
        return False

def load_json(filepath):
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        The loaded data structure, or None if an error occurred
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return None

def format_json(data):
    """
    Format data as a JSON string with standardized indentation.
    
    Args:
        data: The data structure to format
        
    Returns:
        str: Formatted JSON string
    """
    return json.dumps(data, indent=JSON_INDENT, ensure_ascii=False)
