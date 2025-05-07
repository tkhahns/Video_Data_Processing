"""
Main entry point for emotion recognition processing.

This script provides a command-line interface for detecting emotions in videos,
with options for processing specific files or all files in a directory.

Usage:
    python -m src.emotion_recognition_main --input /path/to/video.mp4
    python -m src.emotion_recognition_main --input /path/to/directory --process-all
    python -m src.emotion_recognition_main --interactive
"""
import os
import sys
import argparse
import logging
import glob
from pathlib import Path

# Add the project root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Try importing required modules
try:
    from src.emotion_recognition.processor import EmotionProcessor
    from src.emotion_recognition.interface import find_video_files, select_videos_interactively
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Default video directory
DEFAULT_VIDEO_DIR = "data/videos"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video emotion recognition tool")
    
    parser.add_argument("--input", "-i", type=str, 
                       help=f"Input video file or directory (default: {DEFAULT_VIDEO_DIR})")
    
    parser.add_argument("--output-dir", "-o", type=str, default="./output/emotions",
                       help="Output directory (default: ./output/emotions)")
    
    parser.add_argument("--process-all", "-a", action="store_true",
                       help="Process all video files in the input directory")
    
    parser.add_argument("--interval", "-n", type=float, default=1.0,
                       help="Frame interval in seconds (default: 1.0)")
    
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run models on (default: cpu)")
    
    parser.add_argument("--interactive", action="store_true",
                       help="Use interactive mode to select videos")
    
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Process videos in subdirectories recursively")
    
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
                        
    # Allow for positional arguments (list of video files)
    parser.add_argument("videos", nargs="*", help="Video files to process")
    
    return parser.parse_args()

def main():
    """Main entry point for emotion recognition."""
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize the emotion processor
    processor = EmotionProcessor(
        frame_interval=args.interval,
        confidence_threshold=args.threshold,
        device=args.device
    )
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Handle interactive mode
        if args.interactive:
            logger.info("Using interactive mode for video selection")
            
            # Use default video directory if no input specified
            search_dir = DEFAULT_VIDEO_DIR if not args.input else args.input
            
            # Ensure the directory exists
            if not os.path.isdir(search_dir):
                logger.error(f"Directory not found: {search_dir}")
                if search_dir == DEFAULT_VIDEO_DIR:
                    logger.error(f"The default video directory '{DEFAULT_VIDEO_DIR}' does not exist. "
                                f"Please create it or specify a different directory with --input.")
                return 1
                
            # Find all video files in the specified directory
            video_files = find_video_files(search_dir, recursive=args.recursive)
            
            if not video_files:
                logger.error(f"No video files found in {search_dir}")
                return 1
                
            # Let user select videos
            selected_videos, _ = select_videos_interactively(video_files)
            
            if not selected_videos:
                logger.info("No videos selected. Exiting.")
                return 0
                
            # Process selected videos
            for video_path in selected_videos:
                output_file = processor.process_video(video_path, args.output_dir)
                if output_file:
                    logger.info(f"Processed {video_path} - Output saved to: {output_file}")
            
            return 0
            
        # Handle processing all files in a directory
        if args.process_all:
            # Use default video directory if no input specified
            input_dir = DEFAULT_VIDEO_DIR if not args.input else args.input
            
            if not os.path.isdir(input_dir):
                logger.error(f"Directory not found: {input_dir}")
                return 1
                
            logger.info(f"Processing all videos in directory: {input_dir}")
            
            # Process the directory
            output_files = processor.process_directory(
                input_dir, 
                args.output_dir,
                recursive=args.recursive
            )
            
            logger.info(f"Processed {len(output_files)} videos")
            return 0
        
        # Handle processing specific files (from --input or positional arguments)
        files_to_process = []
        
        # Add the input file if specified and it exists
        if args.input and os.path.isfile(args.input):
            files_to_process.append(args.input)
            
        # Add any files from positional arguments
        files_to_process.extend([v for v in args.videos if os.path.isfile(v)])
        
        # Check if we have any files to process
        if not files_to_process:
            # If no input specified or input is a directory but --process-all was not specified,
            # use the default directory or specified directory to find videos
            input_dir = DEFAULT_VIDEO_DIR
            if args.input and os.path.isdir(args.input):
                input_dir = args.input
                
            # Check if the directory exists
            if not os.path.isdir(input_dir):
                logger.error(f"Directory not found: {input_dir}")
                if input_dir == DEFAULT_VIDEO_DIR:
                    logger.error(f"The default video directory '{DEFAULT_VIDEO_DIR}' does not exist. "
                                f"Please create it or specify a different directory with --input.")
                return 1
                
            # Find video files
            video_files = find_video_files(input_dir, recursive=args.recursive)
            if video_files:
                logger.info(f"Found {len(video_files)} video files in {input_dir}")
                logger.info("Use --process-all to process all files or --interactive to select specific files")
                # List the first 10 files
                for i, file in enumerate(video_files[:10]):
                    logger.info(f"  {i+1}. {os.path.basename(file)}")
                if len(video_files) > 10:
                    logger.info(f"  ... and {len(video_files) - 10} more")
            else:
                logger.error(f"No video files found in {input_dir}")
            return 1
        
        # Process each file
        for video_path in files_to_process:
            logger.info(f"Processing video: {video_path}")
            output_file = processor.process_video(video_path, args.output_dir)
            
            if output_file:
                logger.info(f"Output saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error processing video(s): {e}")
        return 1
    finally:
        # Clean up resources
        if 'processor' in locals():
            processor.release()

if __name__ == "__main__":
    sys.exit(main())
