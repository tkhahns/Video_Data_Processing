"""
Utility functions for the separate_speech package.
"""
import os
import logging
import subprocess
import gc
import torch

from utils import init_logging
logger = init_logging.get_logger(__name__)

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

def check_diarization_dependencies():
    """Check if pyannote.audio or speechbrain is installed."""
    pyannote_ok = False
    speechbrain_ok = False
    try:
        import pyannote.audio
        logger.info("pyannote.audio is available for speaker diarization")
        pyannote_ok = True
    except ImportError:
        logger.warning("pyannote.audio is not installed. Speaker diarization with pyannote will not be available.")
        logger.warning("Install with: pip install pyannote.audio==2.1.1")
    try:
        import speechbrain
        logger.info("speechbrain is available for fallback diarization")
        speechbrain_ok = True
    except ImportError:
        logger.warning("speechbrain is not installed. Fallback diarization will not be available.")
        logger.warning("Install with: pip install speechbrain")
    return pyannote_ok or speechbrain_ok

def merge_adjacent_segments(segments, max_gap_seconds=1.0):
    """
    Merge adjacent segments from the same speaker if they're close enough.
    
    Args:
        segments: List of (start, end, speaker_id) tuples
        max_gap_seconds: Maximum gap between segments to be merged
        
    Returns:
        List of merged segments as (start, end, speaker_id) tuples
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    # Initialize with the first segment
    merged_segments = [list(sorted_segments[0])]
    
    # Process the rest
    for start, end, speaker in sorted_segments[1:]:
        prev_start, prev_end, prev_speaker = merged_segments[-1]
        
        # If same speaker and close enough to previous segment, merge them
        if speaker == prev_speaker and start - prev_end <= max_gap_seconds:
            merged_segments[-1][1] = end  # Update end time of previous segment
        else:
            # Otherwise add as a new segment
            merged_segments.append([start, end, speaker])
    
    return [tuple(segment) for segment in merged_segments]

def get_dialogue_filename(base_filename, dialogue_num, speaker_id=None):
    """Generate filename for a dialogue segment."""
    if speaker_id is not None:
        # Include speaker ID in filename if provided
        return f"{base_filename}_dialogue_{dialogue_num}_speaker_{speaker_id}"
    else:
        # Otherwise just use dialogue number
        return f"{base_filename}_dialogue_{dialogue_num}"
