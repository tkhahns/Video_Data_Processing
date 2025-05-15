"""
Extended interface for interactive selection of videos and options.
"""
import os
import logging
import questionary
from pathlib import Path

# Try to get logger
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

def select_videos_interactively(available_videos, batch_mode=False):
    """
    Show an enhanced interactive interface for selecting videos to process.
    
    Args:
        available_videos: List of available video paths
        batch_mode: If True, select all videos without prompting
        
    Returns:
        Tuple of (selected_videos, file_type)
    """
    if not available_videos:
        logger.error("No video files found. Please provide valid video files or directories.")
        return [], None
    
    # If in batch mode, select all videos and use default file type
    if batch_mode:
        logger.info(f"Batch mode enabled: Selecting all {len(available_videos)} videos")
        return available_videos, "mp3"  # Default output format
    
    # Ask how to select videos
    selection_method = questionary.select(
        "How would you like to select videos?",
        choices=[
            "Select individual videos",
            "Select all videos",
            "Select by pattern",
        ]
    ).ask()
    
    selected_videos = []
    
    if selection_method == "Select all videos":
        selected_videos = available_videos
        logger.info(f"Selected all {len(selected_videos)} videos")
        
    elif selection_method == "Select individual videos":
        # Create a list of video choices with filenames and sizes
        video_choices = []
        for video_path in available_videos:
            try:
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                filename = os.path.basename(video_path)
                video_choices.append({
                    'name': f"{filename} ({size_mb:.1f} MB)",
                    'value': video_path
                })
            except Exception as e:
                logger.error(f"Error getting file info for {video_path}: {e}")
                video_choices.append({
                    'name': os.path.basename(video_path),
                    'value': video_path
                })
        
        # Let user select videos
        if len(video_choices) > 10:
            logger.warning(f"Found {len(video_choices)} videos. Consider using pattern matching for easier selection.")
        
        selected_videos = questionary.checkbox(
            "Select videos to process:",
            choices=video_choices,
        ).ask()
        
        if not selected_videos:
            logger.warning("No videos selected. Exiting.")
            return [], None
            
        logger.info(f"Selected {len(selected_videos)} videos")
        
    elif selection_method == "Select by pattern":
        # Let user enter a pattern
        pattern = questionary.text(
            "Enter a pattern to match video filenames (e.g., '*.mp4' or 'video_*'):"
        ).ask()
        
        if not pattern:
            logger.warning("No pattern entered. Using all videos.")
            selected_videos = available_videos
        else:
            import fnmatch
            selected_videos = []
            for video_path in available_videos:
                if fnmatch.fnmatch(os.path.basename(video_path), pattern):
                    selected_videos.append(video_path)
            
            if not selected_videos:
                logger.warning(f"No videos matched pattern '{pattern}'. Exiting.")
                return [], None
                
            logger.info(f"Selected {len(selected_videos)} videos matching pattern '{pattern}'")
    
    # Ask about file type
    file_type_choice = questionary.select(
        "Select output file format:",
        choices=[
            {"name": "MP3 (smaller files)", "value": "mp3"},
            {"name": "WAV (lossless quality)", "value": "wav"},
            {"name": "Both formats", "value": "both"},
        ]
    ).ask()
    
    return selected_videos, file_type_choice
