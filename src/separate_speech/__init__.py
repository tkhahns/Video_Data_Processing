"""
Speech separation package.
Provides tools for extracting and isolating speech from video files.
"""

__version__ = "1.0.0"

import os
from pathlib import Path

# Get the path to the project root directory
_module_path = Path(__file__).resolve()
_src_dir = _module_path.parent.parent
PROJECT_ROOT = _src_dir.parent

# Default paths (all using absolute paths)
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "downloaded"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "separated_speech"
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
DEFAULT_MODEL = "sepformer"
DEFAULT_CHUNK_SIZE = 10  # Default chunk size in seconds
