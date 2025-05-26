#!/usr/bin/env python3
"""
Command-line interface for emotion and pose recognition.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import getpass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to sys.path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.emotion_and_pose_recognition import processor, utils

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
    """Main CLI function for emotion and pose recognition."""
    parser = argparse.ArgumentParser(description="Extract emotions and poses from videos")
    
    # Input/output arguments - allow both input-dir and direct video input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", "-i", help="Input directory with videos")
    input_group.add_argument("--video", "-v", help="Single video file to process")
    
    parser.add_argument("--output-dir", "-o", help="Output directory for processed videos")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all videos without prompting")
    parser.add_argument("--log-dir", "-l", help="Directory to save log files")
    
    # Processing options
    parser.add_argument("--with-pose", action="store_true", help="Enable pose estimation")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose estimation")
    parser.add_argument("--multi-speaker", action="store_true", help="Enable multi-speaker tracking")
    parser.add_argument("--single-speaker", action="store_true", help="Disable multi-speaker tracking (only track one person)")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode for file selection")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between processing")
    
    # Feature extraction
    parser.add_argument("--extract-features", "-e", action="store_true", help="Extract advanced video features")
    parser.add_argument("--feature-models", nargs="+", 
                        choices=["pare", "vitpose", "psa", "rsn", "au_detector", "dan", "eln", 
                                "mediapipe", "pyfeat", "optical_flow", "all"],
                        default=["mediapipe", "pyfeat"],
                        help="Models to use for feature extraction")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import here to avoid circular imports
    from .processor import process_video, batch_process_videos
    
    # Normalize feature models
    feature_models = args.feature_models
    if "all" in feature_models:
        feature_models = ["pare", "vitpose", "psa", "rsn", "au_detector", 
                         "dan", "eln", "mediapipe", "pyfeat", "optical_flow"]
    
    # Set default output directory if not provided
    if not args.output_dir:
        if args.input_dir:
            args.output_dir = os.path.join(args.input_dir, "emotions_output")
        else:
            video_dir = os.path.dirname(args.video)
            args.output_dir = os.path.join(video_dir, "emotions_output")
        logger.info(f"No output directory specified, using {args.output_dir}")
    
    # Set default log directory if not provided
    if not args.log_dir:
        args.log_dir = args.output_dir
    
    # Create video features directory 
    video_features_dir = os.path.join(args.log_dir, "video_features")
    os.makedirs(video_features_dir, exist_ok=True)
    
    # Process based on input type
    if args.video:
        # Single video processing
        input_path = args.video
        video_name = os.path.basename(input_path)
        base_name = os.path.splitext(video_name)[0]
        
        output_path = os.path.join(args.output_dir, f"{base_name}_emotions_and_pose.mp4")
        log_path = os.path.join(args.log_dir, f"{base_name}_emotions.csv") 
        pose_log_path = os.path.join(args.log_dir, f"{base_name}_pose.json") if args.with_pose and not args.no_pose else None
        
        # Set up video features path
        video_features_path = os.path.join(video_features_dir, f"{base_name}_video_features")
        
        logger.info(f"Processing single video: {video_name}...")
        success = process_video(
            input_path, 
            output_path, 
            log_path,
            backend="opencv",
            with_pose=args.with_pose and not args.no_pose,
            pose_log_path=pose_log_path,
            extract_features=args.extract_features,
            video_features_path=video_features_path,
            video_feature_models=feature_models
        )
        
        if success:
            logger.info(f"Successfully processed {video_name}")
            logger.info(f"Video features saved to: {video_features_path}_aggregate.csv")
            logger.info(f"Full features saved to: {video_features_path}_full.json")
        else:
            logger.error(f"Failed to process {video_name}")
        
        return 0 if success else 1
        
    else:
        # Directory processing
        results = batch_process_videos(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            file_extension="mp4",
            with_pose=args.with_pose and not args.no_pose,
            extract_features=args.extract_features,
            video_features_dir=video_features_dir,
            video_feature_models=feature_models
        )
        
        # Report results
        total = len(results)
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Processed {successful}/{total} videos successfully")
        
        if successful < total:
            logger.warning(f"Failed to process {total - successful} videos")
            for name, success in results.items():
                if not success:
                    logger.warning(f"Failed to process: {name}")
        
        return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())
