"""
Model downloader package.
Provides tools for downloading pre-trained models from various sources.
"""

__version__ = "1.0.0"

# Initialize logging early
import os
import sys

# Add project root to path for more reliable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
