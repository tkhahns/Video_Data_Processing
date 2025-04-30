"""
Utility functions for the download_videos package.
"""
import os
import logging

logger = logging.getLogger(__name__)

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
