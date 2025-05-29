"""
Command-line interface for emotion and pose recognition from videos.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import getpass

# Add parent directory to sys.path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.emotion_and_pose_recognition import processor, utils

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

def get_huggingface_token():
    """
    Get the Hugging Face token from environment variable or prompt the user.
    
    Returns:
        str: The Hugging Face token
    """
    # Check if token exists in environment variable
    token = os.environ.get("HUGGINGFACE_TOKEN")
    
    if not token:
        # Check if token exists in token file
        token_file = os.path.join(os.path.expanduser("~"), ".huggingface_token")
        if os.path.exists(token_file):
            try:
                with open(token_file, "r") as f:
                    token = f.read().strip()
                logger.info("Using Hugging Face token from saved file")
            except Exception as e:
                logger.error(f"Error reading token file: {e}")
                
    if not token:
        # Prompt user for token
        print("\n=== Hugging Face Authentication Required ===")
        print("This module requires a Hugging Face token for accessing models.")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        print("The token will be saved locally for future use.")
        token = getpass.getpass("Enter your Hugging Face token: ")
        
        # Save token for future use
        if token:
            try:
                token_file = os.path.join(os.path.expanduser("~"), ".huggingface_token")
                with open(token_file, "w") as f:
                    f.write(token)
                os.chmod(token_file, 0o600)  # Set permissions to owner read/write only
                logger.info("Saved Hugging Face token for future use")
            except Exception as e:
                logger.error(f"Error saving token: {e}")
    
    if not token:
        logger.error("No Hugging Face token provided. Some models may not work correctly.")
    
    return token

def main():
    """Main entry point for the emotion and pose recognition CLI."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Facial Emotion and Body Pose Recognition from Video. '
                                                 'If no command is specified, runs in interactive mode.')
    
    # Add global options - these should be before the subparsers
    parser.add_argument('--with-pose', '-p', action='store_true', help='Enable body pose estimation (default)')
    parser.add_argument('--no-pose', action='store_true', help='Disable body pose estimation')
    parser.add_argument('--input-dir', help='Directory containing input video files')
    parser.add_argument('--output-dir', help='Directory to save output files')
    parser.add_argument('--multi-speaker', '-m', action='store_true', 
                        help='Enable multi-speaker tracking (up to 2 speakers) - default')
    parser.add_argument('--single-speaker', action='store_true',
                       help='Use single-speaker mode (disable multi-speaker tracking)')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all files without manual selection')
    
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
    single_parser.add_argument('--multi-speaker', '-m', action='store_true', 
                              help='Enable multi-speaker tracking (up to 2 speakers) - default')
    single_parser.add_argument('--single-speaker', action='store_true',
                             help='Use single-speaker mode (disable multi-speaker tracking)')
    
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
    batch_parser.add_argument('--batch', action='store_true',
                              help='Process all files without manual selection')
    
    # 3. Check dependencies command
    check_parser = subparsers.add_parser('check', help='Check if all dependencies are installed')
    
    # 4. Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode to select files to process')
    interactive_parser.add_argument('--input_dir', '-i', default='data/videos', 
                                    help='Input directory containing videos (default: data/videos)')
    interactive_parser.add_argument('--recursive', '-r', action='store_true',
                                   help='Search for video files recursively in subdirectories')
    interactive_parser.add_argument('--output_dir', '-o', default='output/emotions_and_pose',
                                   help='Output directory for processed videos (default: output/emotions_and_pose)')
    interactive_parser.add_argument('--with-pose', '-p', action='store_true',
                                    help='Also perform body pose estimation')
    interactive_parser.add_argument('--batch', action='store_true',
                                    help='Process all files without manual selection')
    interactive_parser.add_argument('files', nargs='*', help='Specific video files to process')

    # Parse arguments
    args = parser.parse_args()
    
    # Get Hugging Face token before proceeding
    huggingface_token = get_huggingface_token()
    os.environ["HUGGINGFACE_TOKEN"] = huggingface_token if huggingface_token else ""
    
    # Make interactive mode the default if no command is specified
    if args.command is None:
        logger.info("No command specified. Running in interactive mode.")
        # Create default arguments for interactive mode
        args.command = 'interactive'
        args.input_dir = args.input_dir or 'data/videos'  # Use provided input-dir or default
        args.recursive = False
        args.output_dir = args.output_dir or 'output/emotions'  # Use provided output-dir or default

    # Set body pose estimation as default, unless --no-pose is specified
    if not hasattr(args, 'with_pose'):
        args.with_pose = True
    if hasattr(args, 'no_pose') and args.no_pose:
        args.with_pose = False
        
    # Enable multi-speaker tracking by default, unless --single-speaker is specified
    if not hasattr(args, 'multi_speaker'):
        args.multi_speaker = True  # Default to multi-speaker support
    if hasattr(args, 'single_speaker') and args.single_speaker:
        args.multi_speaker = False

    # Handle batch mode
    if hasattr(args, 'batch') and args.batch:
        logger.info("Batch mode enabled - processing all files without manual selection")
    
    # Check the command and run the corresponding function
    if args.command == 'process':
        # Check dependencies before processing
        if not utils.check_dependencies():
            logger.error("Dependency check failed. Install required packages and try again.")
            return 1
            
        logger.info(f"Processing video: {args.input}")
        logger.info(f"Body pose estimation: {'enabled' if args.with_pose else 'disabled'}")
        logger.info(f"Multi-speaker tracking: {'enabled' if args.multi_speaker else 'disabled'}")
        
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
                
                output_path = os.path.join(args.output_dir, f"{base_name}_emotions_and_pose.mp4")
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
        
        # If specific files were provided, use those directly
        if hasattr(args, 'files') and args.files:
            video_files = [Path(file) for file in args.files if os.path.exists(file)]
            if video_files:
                logger.info(f"Using {len(video_files)} specified video files")
            else:
                logger.error("None of the specified video files exist")
                return 1
        else:
            # Find video files
            video_files = utils.find_video_files([args.input_dir], args.recursive)
            
            if not video_files:
                logger.error(f"No video files found in {args.input_dir}")
                return 1
        
        # Check if batch mode is enabled
        batch_mode = getattr(args, 'batch', False)
        if batch_mode:
            logger.info(f"Batch mode enabled: Processing all {len(video_files)} files without manual selection")
            
        # Unless specific files were provided, use the interactive selection
        if not (hasattr(args, 'files') and args.files):
            # Use the batch_mode flag when calling select_files_from_list 
            logger.info(f"Found {len(video_files)} video file(s). Select which ones to process:")
            selected_files, log_only = utils.select_files_from_list(video_files, batch_mode)
            
            if not selected_files:
                logger.info("No files selected. Exiting.")
                return 0
        else:
            # Use the provided files directly
            selected_files = video_files
            log_only = False
            
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
