"""
Command-line interface for emotion recognition from videos.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to sys.path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.emotion_recognition import processor, utils

# Set up logging
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

def main():
    """Main entry point for the emotion recognition CLI."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition from Video. '
                                                 'If no command is specified, runs in interactive mode.')
    
    # Add global options - these should be before the subparsers
    parser.add_argument('--use-feat', action='store_true', help='Use py-feat for emotion recognition instead of DeepFace')
    parser.add_argument('--with-pose', '-p', action='store_true', help='Enable body pose estimation (default)')
    parser.add_argument('--no-pose', action='store_true', help='Disable body pose estimation')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # 1. Single video processing command
    single_parser = subparsers.add_parser('process', help='Process a single video file')
    single_parser.add_argument('input', help='Path to input video file')
    single_parser.add_argument('--output', '-o', help='Path to save annotated video', default=None)
    single_parser.add_argument('--log', '-l', help='Path to save emotions log (CSV)', default=None)
    single_parser.add_argument('--show', '-s', action='store_true', help='Show video preview')
    single_parser.add_argument('--skip', '-k', type=int, default=0, 
                               help='Number of frames to skip between processing (0=process all)')
    single_parser.add_argument('--backend', '-b', default='opencv',
                               help='Face detection backend (opencv, ssd, mtcnn, etc.)')
    single_parser.add_argument('--log-only', action='store_true', 
                               help='Only generate log, skip video output for faster processing')
    single_parser.add_argument('--pose-log', help='Path to save pose data (JSON)', default=None)
    
    # 2. Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process multiple videos in a directory')
    batch_parser.add_argument('input_dir', help='Input directory containing videos')
    batch_parser.add_argument('output_dir', help='Output directory for annotated videos')
    batch_parser.add_argument('--log_dir', '-l', help='Directory to save emotion logs', default=None)
    batch_parser.add_argument('--extension', '-e', default='mp4', help='Video file extension to process')
    batch_parser.add_argument('--interactive', '-i', action='store_true', 
                              help='Enable interactive file selection')
    batch_parser.add_argument('--recursive', '-r', action='store_true',
                              help='Search for video files recursively in subdirectories')
    batch_parser.add_argument('--with-pose', '-p', action='store_true',
                              help='Also perform body pose estimation')
    
    # 3. Check dependencies command
    check_parser = subparsers.add_parser('check', help='Check if all dependencies are installed')
    
    # 4. Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode to select files to process')
    interactive_parser.add_argument('--input_dir', '-i', default='data/videos', 
                                    help='Input directory containing videos (default: data/videos)')
    interactive_parser.add_argument('--recursive', '-r', action='store_true',
                                   help='Search for video files recursively in subdirectories')
    interactive_parser.add_argument('--output_dir', '-o', default='output/emotions',
                                   help='Output directory for processed videos (default: output/emotions)')
    interactive_parser.add_argument('--with-pose', '-p', action='store_true',
                                    help='Also perform body pose estimation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Make interactive mode the default if no command is specified
    if args.command is None:
        logger.info("No command specified. Running in interactive mode.")
        # Create default arguments for interactive mode
        args.command = 'interactive'
        args.input_dir = 'data/videos'
        args.recursive = False
        args.output_dir = 'output/emotions'

    # Set body pose estimation as default, unless --no-pose is specified
    if not hasattr(args, 'with_pose'):
        args.with_pose = True
    if hasattr(args, 'no_pose') and args.no_pose:
        args.with_pose = False

    # Check the command and run the corresponding function
    if args.command == 'process':
        # Check dependencies before processing
        if not utils.check_dependencies():
            logger.error("Dependency check failed. Install required packages and try again.")
            return 1
            
        logger.info(f"Processing video: {args.input}")
        logger.info(f"Body pose estimation: {'enabled' if args.with_pose else 'disabled'}")
        
        # Process the video
        success = processor.process_video(
            args.input,
            args.output,
            args.log,
            show_preview=args.show,
            skip_frames=args.skip,
            backend=args.backend,
            log_only=getattr(args, 'log_only', False),
            with_pose=args.with_pose,
            pose_log_path=getattr(args, 'pose_log', None)
        )
        
        return 0 if success else 1
        
    elif args.command == 'batch':
        # Check dependencies before processing
        if not utils.check_dependencies():
            logger.error("Dependency check failed. Install required packages and try again.")
            return 1
            
        logger.info(f"Batch processing videos from: {args.input_dir}")
        
        # Handle interactive mode for batch processing
        if args.interactive:
            # Find video files
            video_files = utils.find_video_files([args.input_dir], args.recursive)
            
            if not video_files:
                logger.error(f"No video files found in {args.input_dir}")
                return 1
                
            logger.info(f"Found {len(video_files)} video file(s). Select which ones to process:")
            selected_files, log_only = utils.select_files_from_list(video_files)
            
            if not selected_files:
                logger.info("No files selected. Exiting.")
                return 0
                
            # Process selected files
            results = {}
            for video_file in selected_files:
                video_name = os.path.basename(video_file)
                base_name = os.path.splitext(video_name)[0]
                
                output_path = os.path.join(args.output_dir, f"{base_name}_emotions.mp4")
                log_path = os.path.join(args.log_dir, f"{base_name}_emotions.csv") if args.log_dir else None
                pose_log_path = os.path.join(args.log_dir, f"{base_name}_pose.json") if args.with_pose else None
                
                logger.info(f"Processing {video_name}...")
                success = processor.process_video(
                    video_file, output_path, log_path, 
                    log_only=log_only,
                    with_pose=args.with_pose,
                    pose_log_path=pose_log_path
                )
                results[video_name] = success
                
                # Clean memory after each video
                utils.clean_memory()
        else:
            # Regular batch processing
            results = processor.batch_process_videos(
                args.input_dir,
                args.output_dir,
                args.log_dir,
                args.extension,
                with_pose=args.with_pose
            )
        
        # Print summary
        total = len(results)
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Batch processing complete: {successful}/{total} videos processed successfully")
        
        return 0 if successful == total else 1
    
    elif args.command == 'interactive':
        # Check dependencies before processing
        if not utils.check_dependencies():
            logger.error("Dependency check failed. Install required packages and try again.")
            return 1
        
        # Create input and output directories if they don't exist
        os.makedirs(args.input_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find video files
        video_files = utils.find_video_files([args.input_dir], args.recursive)
        
        if not video_files:
            logger.error(f"No video files found in {args.input_dir}")
            return 1
            
        logger.info(f"Found {len(video_files)} video file(s). Select which ones to process:")
        selected_files, log_only = utils.select_files_from_list(video_files)
        
        if not selected_files:
            logger.info("No files selected. Exiting.")
            return 0
            
        # Process selected files
        results = {}
        for video_file in selected_files:
            video_name = os.path.basename(video_file)
            base_name = os.path.splitext(video_name)[0]
            
            output_path = os.path.join(args.output_dir, f"{base_name}_emotions_and_pose.mp4")
            log_path = os.path.join(args.output_dir, f"{base_name}_emotions.csv")
            pose_log_path = os.path.join(args.output_dir, f"{base_name}_pose.json") if args.with_pose else None
            
            logger.info(f"Processing {video_name}...")
            success = processor.process_video(
                video_file, output_path, log_path, 
                log_only=log_only,
                with_pose=args.with_pose,
                pose_log_path=pose_log_path
            )
            results[video_name] = success
            
            # Clean memory after each video
            utils.clean_memory()
            
        # Print summary
        total = len(results)
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Interactive processing complete: {successful}/{total} videos processed successfully")
        
        return 0 if successful == total else 1
        
    elif args.command == 'check':
        # Check dependencies
        if utils.check_dependencies():
            logger.info("All dependencies are installed and ready to use")
            
            # Check available backends
            backends = utils.get_available_backends()
            logger.info(f"Available face detection backends: {', '.join(backends)}")
            
            return 0
        else:
            return 1
            
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
