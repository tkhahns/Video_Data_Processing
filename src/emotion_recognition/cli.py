"""
Command-line interface for emotion recognition.
"""
import os
import sys
import argparse
import logging
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import from the same package
from .processor import EmotionProcessor
from .interface import select_videos_interactively

# Try importing utility functions
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video emotion recognition tool")
    
    parser.add_argument("--input", "-i", type=str,
                        help="Input video file or directory")
    
    parser.add_argument("--output-dir", "-o", type=str, default="./output/emotions",
                        help="Output directory (default: ./output/emotions)")
    
    parser.add_argument("--interval", "-n", type=float, default=1.0,
                        help="Frame interval in seconds (default: 1.0)")
    
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run models on (default: cpu)")
    
    parser.add_argument("--interactive", action="store_true",
                        help="Use interactive mode to select videos")
    
    parser.add_argument("--recursive", action="store_true",
                        help="Process videos in subdirectories recursively")
    
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
                        
    # Allow for positional arguments (list of video files)
    parser.add_argument("videos", nargs="*", help="Video files to process")
    
    return parser.parse_args()

def main():
    """Main entry point for the command-line interface."""
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
    
    try:
        # Handle interactive mode
        if args.interactive or (not args.input and not args.videos):
            logger.info("Using interactive mode for video selection")
            from .interface import find_video_files, select_videos_interactively
            
            # Find all video files in current directory
            video_files = find_video_files(".", recursive=args.recursive)
            
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
            
        # Handle normal mode with explicit input
        input_path = args.input or (args.videos[0] if args.videos else None)
        
        if not input_path:
            logger.error("No input specified. Use --input or provide video files as arguments.")
            return 1
            
        if os.path.isdir(input_path):
            logger.info(f"Processing directory: {input_path}")
            output_files = processor.process_directory(
                input_path, 
                args.output_dir,
                recursive=args.recursive
            )
            logger.info(f"Processed {len(output_files)} videos")
        elif os.path.isfile(input_path):
            logger.info(f"Processing video: {input_path}")
            output_file = processor.process_video(input_path, args.output_dir)
            if output_file:
                logger.info(f"Output saved to: {output_file}")
        else:
            logger.error(f"Input not found: {input_path}")
            return 1
        
        # Process any additional videos specified as positional arguments
        for video_path in args.videos[1:] if args.videos else []:
            if os.path.isfile(video_path):
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
        processor.release()

if __name__ == "__main__":
    sys.exit(main())
