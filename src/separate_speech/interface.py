"""
User interface components for file selection and display.
"""
import os
import logging

logger = logging.getLogger(__name__)

def select_videos_interactively(video_files):
    """Display a list of available videos and prompt user to select."""
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
    
    # Now prompt for file type selection
    print("\n=== Select Output File Format ===")
    print("[1] MP3 format (default)")
    print("[2] WAV format")
    print("[3] Both WAV and MP3")
    
    while True:
        file_type_selection = input("\nSelect file format [1-3]: ").strip()
        
        if not file_type_selection:
            # Default to MP3
            file_type = "mp3"
            break
        elif file_type_selection in ["1", "mp3"]:
            file_type = "mp3"
            break
        elif file_type_selection in ["2", "wav"]:
            file_type = "wav"
            break
        elif file_type_selection in ["3", "both"]:
            file_type = "both"
            break
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")
    
    return selected_videos, file_type
