"""
Utility functions for the download_videos package.
"""
import os
import logging
import sys

# Try importing from utils package
try:
    from utils import colored_logging, init_logging
except ImportError:
    # Fall back to adding the parent directory to sys.path
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from utils import colored_logging, init_logging

# Get logger with colored output
logger = init_logging.get_logger(__name__)

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
