"""
Utility functions for the separate_speech package.
"""
import os
import logging
import subprocess
import gc
import torch

logger = logging.getLogger(__name__)

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def check_ffmpeg_dependencies():
    """Check if ffmpeg and ffprobe are installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        logger.info("ffmpeg and ffprobe are available.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("ffmpeg or ffprobe is not installed. Please install them to enable MP3 conversion.")
        logger.error("Install ffmpeg using: apt install ffmpeg (Linux) or brew install ffmpeg (macOS).")
        return False

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        logger.warning("psutil not installed. Cannot monitor memory usage.")
        return 0

def clean_memory():
    """Clean up memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
