"""
Entry point for running the download_models package as a module.
Enables execution as: python -m src.download_models
"""
import os
import sys
from pathlib import Path

# Add the project root to the path for direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import from the utils package
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
    logger.info("Starting model downloader...")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Could not import colored logging, using default logging instead")
    logger.info("Starting model downloader...")

# Import the main function
from src.download_models.main import main

# Execute main function directly when run as a script
if __name__ == "__main__":
    sys.exit(main())
