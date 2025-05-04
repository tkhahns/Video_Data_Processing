"""
Entry point for running the download_videos package as a module.
Enables execution as: python -m src.download_videos or python src/download_videos
"""
import sys
import os
from pathlib import Path

# Import the colored logging system
from utils import init_logging

# Initialize colored logging
logger = init_logging.get_logger(__name__)
logger.info("Starting video downloader...")

# Import the main function
from src.download_videos.main import main

# Execute main function directly when run as a script
if __name__ == "__main__":
    sys.exit(main())
