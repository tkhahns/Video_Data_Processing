"""
Feature extractor module for extracting video features in parallel.

This module provides a way to extract video features directly from source videos
without running the full emotion recognition pipeline.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from . import video_features
from . import multimodal_features
from .utils import find_video_files, select_files_from_list, clean_memory

# Try to use the project's logger
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

def extract_features_from_video(video_path, output_dir, models=None, audio_path=None):
    """
    Extract features from a single video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save features
        models: List of feature models to use
        audio_path: Path to corresponding audio file if available
    
    Returns:
        bool: Success status
    """
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create output file path
        output_path = os.path.join(output_dir, f"{video_name}_features")
        
        logger.info(f"Extracting features from {video_path}")
        
        # Default to all models if not specified
        if not models:
            models = ["mediapipe", "pyfeat", "optical_flow", "pare", "vitpose", 
                     "psa", "rsn", "au_detector", "dan", "eln",
                     "av_hubert", "meld"]
        
        # Separate visual and multimodal models
        visual_models = []
        multimodal_models = []
        
        for model in models:
            if model in ["av_hubert", "meld"]:
                multimodal_models.append(model)
            else:
                visual_models.append(model)
            
        # Check for multimodal models
        if multimodal_models and audio_path:
            logger.info(f"Found multimodal models to extract: {multimodal_models}")
            logger.info(f"Using audio file for multimodal analysis: {audio_path}")
            
            # Extract multimodal features
            multimodal_features.extract_multimodal_features(
                video_path=video_path,
                audio_path=audio_path,
                output_dir=output_dir,
                models=multimodal_models,
                use_gpu=True
            )
        elif multimodal_models and not audio_path:
            logger.warning("Multimodal models requested but no audio path provided. Skipping multimodal features.")
        
        # Extract visual-only features if any were requested
        if visual_models:
            logger.info(f"Extracting visual features with models: {visual_models}")
            video_features.extract_video_features(
                video_path=video_path,
                output_dir=output_dir,
                models=visual_models,
                use_gpu=True,
                sample_rate=1,  # Process every frame
                video_name=video_name
            )
        
        logger.info(f"Successfully extracted features from {video_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting features from {video_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract features from video files in parallel")
    parser.add_argument("--video", action="append", help="Video file to process (can be specified multiple times)")
    parser.add_argument("--input-dir", help="Directory containing input video files")
    parser.add_argument("--output-dir", default="./output/features", help="Directory to save extracted features")
    parser.add_argument("--audio-dir", help="Directory containing audio files for multimodal analysis")
    parser.add_argument("--audio-path", action="append", help="Audio file to use with corresponding video (can be specified multiple times)")
    parser.add_argument("--batch", action="store_true", help="Process all files without manual selection")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode")
    parser.add_argument("--extract-only", action="store_true", help="Only extract features without processing video")
    parser.add_argument("--feature-models", nargs="+", 
                      help="Specify which feature models to use: mediapipe pyfeat optical_flow av_hubert meld pare vitpose psa rsn au_detector dan eln all")
    
    args = parser.parse_args()
    
    # Setup debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Find video files from arguments or input directory
    input_files = []
    if args.video:
        for video_path in args.video:
            if os.path.exists(video_path):
                input_files.append(Path(video_path))
            else:
                logger.warning(f"Video not found: {video_path}")
    
    if args.input_dir:
        input_files.extend(find_video_files([args.input_dir]))
    
    if not input_files:
        logger.error("No video files found. Specify --video or --input-dir.")
        return 1
    
    logger.info(f"Found {len(input_files)} video files")
    
    # If not batch mode and not in extract-only mode, allow interactive selection
    if not args.batch and (args.interactive or len(input_files) > 1):
        logger.info("Enter interactive mode for file selection")
        input_files, _ = select_files_from_list(input_files)
        
        if not input_files:
            logger.warning("No files selected. Exiting.")
            return 0
    
    # Determine feature models to use
    feature_models = args.feature_models if args.feature_models else ["all"]
    if "all" in feature_models:
        feature_models = ["mediapipe", "pyfeat", "optical_flow", "pare", "vitpose", 
                         "psa", "rsn", "au_detector", "dan", "eln", "av_hubert", "meld"]
        
    # Find matching audio files
    audio_files = {}
    
    # First, try using explicitly provided audio paths
    if args.audio_path:
        for i, video_path in enumerate(input_files):
            if i < len(args.audio_path):
                audio_path = args.audio_path[i]
                if os.path.exists(audio_path):
                    audio_files[str(video_path)] = audio_path
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
    
    # If audio directory is provided, look for matching audio files there
    if args.audio_dir and os.path.exists(args.audio_dir):
        logger.info(f"Searching for matching audio files in {args.audio_dir}")
        for video_path in input_files:
            # Only find audio if we don't already have one for this video
            if str(video_path) not in audio_files:
                video_name = video_path.stem
                potential_audio = Path(args.audio_dir) / f"{video_name}.wav"
                if potential_audio.exists():
                    audio_files[str(video_path)] = str(potential_audio)
                    logger.info(f"Found matching audio for {video_name}: {potential_audio}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each video
    success_count = 0
    
    for video_path in input_files:
        video_str = str(video_path)
        audio_path = audio_files.get(video_str)
        
        if extract_features_from_video(video_str, args.output_dir, feature_models, audio_path):
            success_count += 1
        
        # Clean memory after each video
        clean_memory()
    
    logger.info(f"Feature extraction completed: {success_count}/{len(input_files)} videos processed successfully")
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
