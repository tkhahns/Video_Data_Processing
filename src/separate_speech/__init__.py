"""
Speech separation package.
Provides tools for extracting and isolating speech from video files.
"""

__version__ = "1.0.0"

from pathlib import Path

# Default paths
DEFAULT_MODELS_DIR = Path("./models/downloaded")
DEFAULT_OUTPUT_DIR = Path("./output/separated_speech")
DEFAULT_VIDEOS_DIR = Path("./data/videos")
DEFAULT_MODEL = "sepformer"
DEFAULT_CHUNK_SIZE = 10  # Default chunk size in seconds
