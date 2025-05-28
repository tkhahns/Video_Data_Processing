"""
Extract visual features from video files.

This module provides functions to extract various types of visual features
from videos, including pose, facial landmarks, and visual emotions.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Try to use project's logger
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

from .utils import find_video_files, select_files_from_list, clean_memory

class VideoFeatureExtractor:
    """Class for extracting visual features from videos."""
    
    def __init__(self, models=None, use_gpu=True, sample_rate=1):
        """
        Initialize the video feature extractor.
        
        Args:
            models: List of feature models to use
            use_gpu: Whether to use GPU acceleration when available
            sample_rate: Process every N-th frame
        """
        self.models = models if models else ["all"]
        self.use_gpu = use_gpu
        self.sample_rate = sample_rate
        
    def extract(self, video_path, output_dir, video_name=None):
        """
        Extract features from a video file.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted features
            video_name: Optional name to use for the video
            
        Returns:
            Dictionary of extracted features
        """
        return extract_video_features(
            video_path=video_path,
            output_dir=output_dir,
            models=self.models,
            use_gpu=self.use_gpu,
            sample_rate=self.sample_rate,
            video_name=video_name
        )

def extract_video_features(video_path, output_dir, models=None, use_gpu=True, sample_rate=1, video_name=None, **kwargs):
    """
    Extract visual features from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted features
        models: List of feature models to use
        use_gpu: Whether to use GPU acceleration if available
        sample_rate: Process every N-th frame
        video_name: Optional name to use for the video (default: extracted from filename)
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary of extracted features
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video name from file path if not provided
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    logger.info(f"Extracting visual features from {video_path}")
    
    # Define visual-only models (move multimodal models to dedicated module)
    available_models = [
        "mediapipe", "pyfeat", "optical_flow", "pare", "vitpose", 
        "psa", "rsn", "au_detector", "dan", "eln"
    ]
    
    # Determine which models to use
    if not models or "all" in models:
        models_to_use = available_models
    else:
        # Filter out any requested multimodal models as they should be handled separately
        models_to_use = [m for m in models if m in available_models]
        
        # Notify if multimodal models were requested but will be handled separately
        multimodal_models = [m for m in models if m in ["av_hubert", "meld"]]
        if multimodal_models:
            logger.info(f"Multimodal models {multimodal_models} will be processed by the multimodal_features module instead")
    
    logger.info(f"Using video feature models: {models_to_use}")
    
    # Dictionary to store all extracted features
    all_features = {"frame_id": [], "timestamp": [], "video_name": []}
    
    # Extract features from each model
    for model_name in models_to_use:
        logger.info(f"Extracting {model_name} features...")
        
        if model_name == "mediapipe":
            from .models import mediapipe_model
            features = mediapipe_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu,
                sample_rate=sample_rate
            )
            # Add mediapipe features to all_features with GMP prefix
            for key, values in features.items():
                all_features[f"GMP_{key}"] = values
                
        elif model_name == "pyfeat":
            from .models import pyfeat_model
            features = pyfeat_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu,
                sample_rate=sample_rate
            )
            # Add pyfeat features to all_features with PYF prefix
            for key, values in features.items():
                all_features[f"PYF_{key}"] = values
                
        elif model_name == "optical_flow":
            from .models import optical_flow_model
            features = optical_flow_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu
            )
            # Add optical flow features to all_features with OF prefix
            for key, values in features.items():
                all_features[f"of_{key}"] = values
                
        elif model_name == "pare":
            from .models import pare_model
            features = pare_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu
            )
            # Add PARE features
            for key, values in features.items():
                all_features[f"PARE_{key}"] = values
                
        elif model_name == "vitpose":
            from .models import vitpose_model
            features = vitpose_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu
            )
            # Add ViTPose features
            for key, values in features.items():
                all_features[f"vit_{key}"] = values
                
        elif model_name == "psa":
            from .models import psa_model
            features = psa_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu
            )
            # Add PSA features
            for key, values in features.items():
                all_features[f"psa_{key}"] = values
                
        elif model_name == "rsn":
            from .models import rsn_model
            features = rsn_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu
            )
            # Add RSN features
            for key, values in features.items():
                all_features[f"rsn_{key}"] = values
                
        elif model_name == "au_detector":
            from .models import au_detector_model
            features = au_detector_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu
            )
            # Add AU detector features
            for key, values in features.items():
                all_features[f"ann_{key}"] = values
                
        elif model_name == "dan":
            from .models import dan_model
            features = dan_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu
            )
            # Add DAN features
            for key, values in features.items():
                all_features[f"dan_{key}"] = values
                
        elif model_name == "eln":
            from .models import eln_model
            features = eln_model.extract_features(
                video_path=video_path,
                use_gpu=use_gpu
            )
            # Add ELN features
            for key, values in features.items():
                all_features[f"eln_{key}"] = values
    
    # Create a DataFrame from all extracted features
    df = pd.DataFrame(all_features)
    
    # Add video name column if not already present
    if "video_name" not in df.columns:
        df["video_name"] = video_name
    
    # Save the features to a CSV file
    output_file = os.path.join(output_dir, f"{video_name}_video_features.csv")
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved video features to {output_file}")
    
    # Also create a summary file with aggregate statistics (mean, min, max, std)
    aggregate_features = {}
    aggregate_features["video_name"] = video_name
    
    # Compute aggregate statistics for numerical columns
    for col in df.columns:
        if col not in ["frame_id", "timestamp", "video_name"] and pd.api.types.is_numeric_dtype(df[col]):
            try:
                aggregate_features[f"{col}_mean"] = df[col].mean()
                aggregate_features[f"{col}_min"] = df[col].min()
                aggregate_features[f"{col}_max"] = df[col].max()
                aggregate_features[f"{col}_std"] = df[col].std()
            except:
                # Skip aggregation for this column if there's an error
                pass
    
    # Save the aggregate features to a CSV file
    agg_df = pd.DataFrame([aggregate_features])
    agg_output_file = os.path.join(output_dir, f"{video_name}_video_features.csv")
    agg_df.to_csv(agg_output_file, index=False)
    
    logger.info(f"Saved aggregate video features to {agg_output_file}")
    
    return all_features

def main():
    """Command-line interface for video feature extraction."""
    parser = argparse.ArgumentParser(description="Extract features from video files")
    parser.add_argument("--video", action="append", help="Video file to process (can be specified multiple times)")
    parser.add_argument("--input-dir", help="Directory containing input video files")
    parser.add_argument("--output-dir", default="./output/features", help="Directory to save feature files")
    parser.add_argument("--sample-rate", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--batch", action="store_true", help="Process all files without manual selection")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--models", nargs="+", 
                      help="Specify which feature models to use: mediapipe pyfeat optical_flow pare vitpose psa rsn au_detector dan eln all")
    parser.add_argument("--interactive", action="store_true", help="Force interactive file selection")
    
    args = parser.parse_args()
    
    # Setup debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        try:
            logger.setLevel(logging.DEBUG)
        except:
            pass
    
    # Find video files from arguments or input directory
    input_files = []
    if args.video:
        for path in args.video:
            if os.path.exists(path):
                input_files.append(Path(path))
            else:
                print(f"Warning: Video not found: {path}")
    
    if args.input_dir:
        input_files.extend(find_video_files([args.input_dir]))
    
    if not input_files:
        print("Error: No video files found. Specify --video or --input-dir.")
        return 1
    
    print(f"Found {len(input_files)} video files")
    
    # If not in batch mode, allow interactive selection
    if not args.batch and (args.interactive or len(input_files) > 1):
        print("Enter interactive mode for file selection")
        input_files, _ = select_files_from_list(input_files)
        
        if not input_files:
            print("No files selected. Exiting.")
            return 0
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each video
    success_count = 0
    for video_path in input_files:
        video_str = str(video_path)
        video_name = os.path.splitext(os.path.basename(video_str))[0]
        
        try:
            # Extract features
            extract_video_features(
                video_path=video_str,
                output_dir=args.output_dir,
                models=args.models,
                use_gpu=True,
                sample_rate=args.sample_rate,
                video_name=video_name
            )
            
            print(f"Successfully extracted features from {video_path}")
            success_count += 1
            
        except Exception as e:
            print(f"Error extracting features from {video_path}: {e}")
            import traceback
            print(traceback.format_exc())
        
        # Clean memory after each video
        clean_memory()
    
    print(f"Feature extraction completed: {success_count}/{len(input_files)} videos processed successfully")
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
