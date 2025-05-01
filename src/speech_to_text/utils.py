"""
Utility functions for speech-to-text processing.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

# Configure logging
logger = logging.getLogger(__name__)

def ensure_dir_exists(dir_path: str) -> Path:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        dir_path: Directory path
    
    Returns:
        Path object of the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_audio_files(
    paths: Optional[List[str]],
    recursive: bool = False,
    default_dir: Optional[Path] = None
) -> List[Path]:
    """
    Find audio files from given paths or default directory.
    
    Args:
        paths: List of file paths or directories to search
        recursive: Whether to search recursively in directories
        default_dir: Default directory to search if paths is empty
        
    Returns:
        List of found audio file paths
    """
    from src.speech_to_text import SUPPORTED_AUDIO_FORMATS
    
    audio_files = []
    
    # If no paths provided and default_dir exists, use it
    if not paths and default_dir and default_dir.exists():
        paths = [str(default_dir)]
        logger.info(f"No input paths provided. Looking for separated speech files in {default_dir}")
    
    # If still no paths, return empty list
    if not paths:
        return []
    
    for path in paths:
        path = Path(path)
        
        # If path is a file, add it if it's an audio file
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                audio_files.append(path)
                logger.debug(f"Found audio file: {path}")
        
        # If path is a directory, search for audio files
        elif path.is_dir():
            if recursive:
                for audio_file in path.glob("**/*"):
                    if audio_file.is_file() and audio_file.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                        audio_files.append(audio_file)
                        logger.debug(f"Found audio file (recursive): {audio_file}")
            else:
                for audio_file in path.glob("*"):
                    if audio_file.is_file() and audio_file.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                        audio_files.append(audio_file)
                        logger.debug(f"Found audio file: {audio_file}")
    
    if not audio_files:
        logger.info("No audio files found in specified locations")
    else:
        logger.info(f"Found {len(audio_files)} audio files")
        
    return sorted(audio_files)


def check_dependencies() -> bool:
    """
    Check if all dependencies are installed.
    
    Returns:
        True if all dependencies are installed, False otherwise
    """
    try:
        import torch
        import transformers
        import tqdm
        
        # Check for WhisperX
        try:
            import whisperx
            logger.info("WhisperX is installed")
        except ImportError:
            logger.warning("WhisperX not found. Install with: pip install git+https://github.com/m-bain/whisperX.git")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install torch transformers tqdm")
        return False


def get_output_path(audio_path: Path, output_dir: Path) -> Path:
    """
    Generate an output path for a transcription file.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the transcription
        
    Returns:
        Path object for the output file
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Generate output path based on audio filename
    output_path = output_dir / f"{audio_path.stem}_transcript"
    
    return output_path


def is_separated_speech_file(file_path: Path) -> bool:
    """
    Check if a file appears to be from the speech separation module.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if the file appears to be from speech separation, False otherwise
    """
    # Check if file is in the separated speech output directory
    if "separated_speech" in str(file_path):
        return True
    
    # Check filename patterns that might indicate separated speech
    if "_speech" in file_path.stem or "_separated" in file_path.stem:
        return True
        
    return False
