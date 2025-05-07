"""
Utility functions for the download_models package.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path for more reliable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

try:
    # Try absolute import 
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging if utils module not found
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.warning("Could not import colored logging, using default logging instead")

# Models directory
MODELS_DIR = Path("./models/downloaded")

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
