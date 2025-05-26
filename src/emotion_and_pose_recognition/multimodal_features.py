"""
Extract multimodal features from video-audio pairs.

This module provides functionality to extract features that require both
video and audio modalities, such as audiovisual emotion recognition.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Try to use project's logger and utilities
try:
    from utils import init_logging, json_utils
    logger = init_logging.get_logger(__name__)
    save_json = json_utils.save_json
    JSON_INDENT = json_utils.JSON_INDENT
except ImportError:
    # Fall back to standard logging and define JSON utilities locally
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    JSON_INDENT = 2
    
    def save_json(data, filepath):
        """Local implementation of save_json if utils module is not available"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=JSON_INDENT, ensure_ascii=False)

from .utils import find_video_files, select_files_from_list, clean_memory

def extract_multimodal_features(video_path, audio_path, output_dir, models=None, use_gpu=True):
    """
    Extract multimodal features from a video-audio pair.
    
    Args:
        video_path: Path to the video file
        audio_path: Path to the matching audio file
        output_dir: Directory to save the extracted features
        models: List of models to use for feature extraction
        use_gpu: Whether to use GPU acceleration when available
        
    Returns:
        Path to the saved features file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base name of the video without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Determine which models to use
    available_models = ["av_hubert", "meld"]
    if models is None or "all" in models:
        models_to_use = available_models
    else:
        models_to_use = [m for m in models if m in available_models]
    
    logger.info(f"Extracting multimodal features from {video_path} with audio {audio_path}")
    logger.info(f"Using models: {models_to_use}")
    
    # Prepare a dictionary for all extracted features
    all_features = {"frame_id": [], "timestamp": [], "video_name": []}
    
    # Initialize the feature extractors and extract features from each model
    for model_name in models_to_use:
        logger.info(f"Extracting features with {model_name} model...")
        
        if model_name == "av_hubert":
            # Import av_hubert model on demand to reduce initial loading time
            from .models import av_hubert_model
            features = av_hubert_model.extract_features(
                video_path=video_path,
                audio_path=audio_path,
                use_gpu=use_gpu
            )
            # Add av_hubert features to all_features
            for key, values in features.items():
                all_features[f"avh_{key}"] = values
                
        elif model_name == "meld":
            # Import meld model on demand
            from .models import meld_model
            features = meld_model.extract_features(
                video_path=video_path,
                audio_path=audio_path,
                use_gpu=use_gpu
            )
            # Add meld features to all_features
            for key, values in features.items():
                all_features[f"meld_{key}"] = values
    
    # Create a DataFrame from all extracted features
    df = pd.DataFrame(all_features)
    
    # Save the features to a CSV file
    output_file = os.path.join(output_dir, f"{video_name}_multimodal_features.csv")
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved multimodal features to {output_file}")
    
    # Also create an aggregate features file with statistics
    aggregate_features = {}
    aggregate_features["video_name"] = video_name
    
    # Compute aggregate statistics for numerical columns
    for col in df.columns:
        if col not in ["frame_id", "timestamp", "video_name"] and df[col].dtype in [np.float64, np.int64]:
            aggregate_features[f"{col}_mean"] = df[col].mean()
            aggregate_features[f"{col}_min"] = df[col].min()
            aggregate_features[f"{col}_max"] = df[col].max()
            aggregate_features[f"{col}_std"] = df[col].std()
    
    # Save the aggregate features to a CSV file
    agg_df = pd.DataFrame([aggregate_features])
    agg_output_file = os.path.join(output_dir, f"{video_name}_multimodal_features.csv")
    agg_df.to_csv(agg_output_file, index=False)
    
    logger.info(f"Saved aggregate multimodal features to {agg_output_file}")
    
    return output_file

def find_matching_audio_files(video_files, speech_dir):
    """Find matching audio files for video files in the speech directory."""
    if not speech_dir or not os.path.exists(speech_dir):
        return {}
    
    audio_matches = {}
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        potential_audio = os.path.join(speech_dir, f"{video_name}.wav")
        
        if os.path.exists(potential_audio):
            audio_matches[str(video_path)] = potential_audio
            logger.info(f"Found matching audio for {video_name}: {potential_audio}")
    
    return audio_matches

def main():
    """Main entry point for the multimodal feature extraction script."""
    parser = argparse.ArgumentParser(description="Extract multimodal features from video-audio pairs")
    parser.add_argument("--video", action="append", help="Video file to process (can be specified multiple times)")
    parser.add_argument("--audio-path", action="append", help="Audio file to use with corresponding video (can be specified multiple times)")
    parser.add_argument("--speech-dir", help="Directory containing separated speech audio files")
    parser.add_argument("--output-dir", default="./output/multimodal_features", help="Directory to save feature files")
    parser.add_argument("--models", nargs="+", help="Space-separated list of models to use: av_hubert meld all")
    parser.add_argument("--batch", action="store_true", help="Process all files without manual selection")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--interactive", action="store_true", help="Force interactive file selection")
    
    args = parser.parse_args()
    
    # Setup debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Get the list of video files to process
    video_files = []
    if args.video:
        for path in args.video:
            if os.path.exists(path):
                video_files.append(Path(path))
            else:
                logger.warning(f"Video file not found: {path}")
    
    if not video_files:
        logger.error("No valid video files provided")
        return 1
    
    # If not in batch mode, allow interactive selection
    if not args.batch and (args.interactive or len(video_files) > 1):
        logger.info("Entering interactive mode for file selection")
        video_files, _ = select_files_from_list(video_files)
        
        if not video_files:
            logger.warning("No files selected. Exiting.")
            return 0
    
    # Find matching audio files for each video
    audio_paths = {}
    
    # First, try to use explicitly provided audio paths
    if args.audio_path:
        for i, video_path in enumerate(video_files):
            if i < len(args.audio_path) and os.path.exists(args.audio_path[i]):
                audio_paths[str(video_path)] = args.audio_path[i]
    
    # Then, try to find matching audio files in the speech directory
    if args.speech_dir:
        speech_matches = find_matching_audio_files(video_files, args.speech_dir)
        # Only add matches for videos that don't already have an audio path
        for video_path, audio_path in speech_matches.items():
            if video_path not in audio_paths:
                audio_paths[video_path] = audio_path
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each video with its matching audio
    success_count = 0
    for video_path in video_files:
        video_str = str(video_path)
        if video_str in audio_paths:
            try:
                extract_multimodal_features(
                    video_path=video_str,
                    audio_path=audio_paths[video_str],
                    output_dir=args.output_dir,
                    models=args.models
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"No matching audio found for {video_path}. Skipping.")
        
        # Clean memory after each video
        clean_memory()
    
    logger.info(f"Multimodal feature extraction completed: {success_count}/{len(video_files)} videos processed successfully")
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
