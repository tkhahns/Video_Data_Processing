"""
Entry point for running the download_videos package as a module.
Enables execution as: python -m src.download_videos or python src/download_videos
"""
import sys
import os
from pathlib import Path

# Add the project root to the path for direct execution
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

# Import the main function
from src.download_videos.main import main

# Execute main function directly when run as a script
if __name__ == "__main__":
    sys.exit(main())
