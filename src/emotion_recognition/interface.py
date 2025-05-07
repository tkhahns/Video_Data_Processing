"""
Interface components for video selection and display.
"""
import os
import logging
from typing import List, Tuple, Optional

# Try importing utility functions
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

def find_video_files(directory: str, recursive: bool = False) -> List[str]:
    """
    Find all video files in a directory.
    
    Args:
        directory: Directory to search in
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of video file paths
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)) and file.lower().endswith(video_extensions):
                video_files.append(os.path.join(directory, file))
    
    return video_files

def select_videos_interactively(video_files: List[str]) -> Tuple[List[str], Optional[str]]:
    """
    Display a list of available videos and prompt user to select.
    
    Args:
        video_files: List of video file paths
        
    Returns:
        Tuple of (selected_video_paths, file_type)
    """
    if not video_files:
        logger.error("No video files found")
        return [], None
    
    # Display the list of available videos
    print("\n=== Available Video Files ===")
    for i, video_path in enumerate(video_files, 1):
        # Get file size in MB
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"[{i}] {os.path.basename(video_path)} ({size_mb:.1f} MB)")
    
    # Prompt for selection
    while True:
        print("\nOptions:")
        print("- Enter numbers (e.g., '1,3,5') to select specific videos")
        print("- Enter 'all' to process all videos")
        print("- Enter 'q' to quit")
        
        selection = input("\nSelect videos to process: ").strip().lower()
        
        if selection == 'q':
            return [], None
            
        if selection == 'all':
            selected_videos = video_files
            break
        
        try:
            # Parse comma-separated indices
            indices = [int(idx.strip()) for idx in selection.split(',')]
            selected_videos = []
            
            for idx in indices:
                if 1 <= idx <= len(video_files):
                    selected_videos.append(video_files[idx-1])
                else:
                    print(f"Error: {idx} is not a valid video number")
                    break
            else:
                # If no break occurred in the loop
                if selected_videos:
                    break
                print("No valid videos selected. Please try again.")
                
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")
    
    # Return selected videos and None for file_type (not used in emotion recognition)
    return selected_videos, None
