"""
Pipeline-level utilities for combining and processing data from multiple modules.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import glob

# Configure logging
logger = logging.getLogger(__name__)

def create_pipeline_output(results_dir: str, output_file: str = "pipeline_output.csv") -> bool:
    """
    Create a combined pipeline output CSV by merging speech transcripts, 
    audio features, and video features from the pipeline directories.
    
    Args:
        results_dir (str): Root directory containing pipeline results
        output_file (str): Name of the output file to create
    
    Returns:
        bool: True if successful
    """
    logger.info(f"Creating pipeline output from {results_dir}")
    
    # Define directory paths
    speech_dir = os.path.join(results_dir, "speech")
    transcript_dir = os.path.join(results_dir, "transcripts")
    emotions_dir = os.path.join(results_dir, "emotions_and_pose")
    
    # Get audio feature files
    audio_feature_files = glob.glob(os.path.join(speech_dir, "*_audio_features.csv"))
    logger.info(f"Found {len(audio_feature_files)} audio feature files")
    
    # Get video feature files (the aggregate ones, not full JSON files)
    video_feature_paths = [
        # Standard path in video_features directory
        os.path.join(emotions_dir, "video_features"),
        # Direct in emotions directory
        emotions_dir
    ]
    
    video_feature_files = []
    for path in video_feature_paths:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "*_aggregate.csv"))
            if files:
                video_feature_files.extend(files)
                logger.info(f"Found {len(files)} video feature files in {path}")
    
    # Get transcript files that have diarization
    transcript_files = glob.glob(os.path.join(transcript_dir, "*_diarized.csv"))
    if not transcript_files:
        # Try with non-diarized transcripts as fallback
        transcript_files = glob.glob(os.path.join(transcript_dir, "*.csv"))
    
    logger.info(f"Found {len(transcript_files)} transcript files")
    
    if not audio_feature_files and not video_feature_files:
        logger.error("No feature files found. Cannot create pipeline output.")
        return False
    
    # Create a dictionary to hold dataframes by base filename
    all_features = {}
    
    # Process audio feature files
    for file_path in audio_feature_files:
        try:
            # Extract the base name from the file path
            base_name = os.path.basename(file_path)
            video_name = base_name.replace("_audio_features.csv", "")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add video_name column if not present
            if "video_name" not in df.columns:
                df["video_name"] = video_name
            
            # Store in dictionary
            all_features[video_name] = {"audio_features": df}
        except Exception as e:
            logger.error(f"Error processing audio feature file {file_path}: {e}")
    
    # Process video feature files
    for file_path in video_feature_files:
        try:
            # Extract base name - handles either format
            base_name = os.path.basename(file_path)
            if "_video_features_aggregate.csv" in base_name:
                video_name = base_name.replace("_video_features_aggregate.csv", "")
            else:
                # Handle other naming patterns
                video_name = base_name.split("_aggregate")[0]
                video_name = video_name.replace("_emotions_and_pose_video_features", "")
                video_name = video_name.replace("_video_features", "")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add video_name column if not present
            if "video_name" not in df.columns:
                df["video_name"] = video_name
            
            # Store in dictionary
            if video_name in all_features:
                all_features[video_name]["video_features"] = df
            else:
                all_features[video_name] = {"video_features": df}
        except Exception as e:
            logger.error(f"Error processing video feature file {file_path}: {e}")
    
    # Process transcript files (if we want to include transcripts in the output)
    for file_path in transcript_files:
        try:
            base_name = os.path.basename(file_path)
            video_name = base_name.replace("_diarized.csv", "").replace(".csv", "")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add video_name column if not present
            if "video_name" not in df.columns:
                df["video_name"] = video_name
            
            # Store in dictionary
            if video_name in all_features:
                all_features[video_name]["transcript"] = df
            else:
                all_features[video_name] = {"transcript": df}
        except Exception as e:
            logger.error(f"Error processing transcript file {file_path}: {e}")
    
    # Create combined dataframes
    combined_dfs = []
    for video_name, feature_dict in all_features.items():
        try:
            # Start with a minimal dataframe containing just the video name
            combined_df = pd.DataFrame({"video_name": [video_name]})
            
            # Merge audio features if available
            if "audio_features" in feature_dict:
                audio_df = feature_dict["audio_features"]
                # Drop duplicate video_name to avoid conflicts
                if "video_name" in audio_df.columns:
                    audio_df = audio_df.drop(columns=["video_name"])
                
                # If audio_df has multiple rows, take the first row or aggregate
                if len(audio_df) > 1:
                    audio_df = audio_df.iloc[0:1].reset_index(drop=True)
                
                # Use a prefix for audio features
                audio_df = audio_df.add_prefix('audio_')
                
                # Join with combined dataframe
                combined_df = pd.concat([combined_df, audio_df], axis=1)
            
            # Merge video features if available
            if "video_features" in feature_dict:
                video_df = feature_dict["video_features"]
                # Drop duplicate video_name to avoid conflicts
                if "video_name" in video_df.columns and video_df.shape[0] == 1:
                    video_df = video_df.drop(columns=["video_name"])
                
                # If video_df has multiple rows, take the first row
                if len(video_df) > 1:
                    video_df = video_df.iloc[0:1].reset_index(drop=True)
                
                # Use a prefix for video features (except for MELD features which are already prefixed)
                prefixed_cols = {}
                for col in video_df.columns:
                    if not col.startswith("MELD_") and col != "video_name":
                        prefixed_cols[col] = f"video_{col}"
                
                if prefixed_cols:
                    video_df = video_df.rename(columns=prefixed_cols)
                
                # Join with combined dataframe
                combined_df = pd.concat([combined_df, video_df], axis=1)
            
            # Add summary data from transcript if available
            if "transcript" in feature_dict:
                transcript_df = feature_dict["transcript"]
                
                # Extract summary metrics (word count, speaker count, etc.)
                word_count = len(" ".join(transcript_df["text"].astype(str)).split())
                speaker_count = transcript_df["speaker"].nunique() if "speaker" in transcript_df.columns else 1
                duration = transcript_df["end"].max() if "end" in transcript_df.columns else 0
                
                # Create summary dataframe
                summary_df = pd.DataFrame({
                    "transcript_word_count": [word_count],
                    "transcript_speaker_count": [speaker_count],
                    "transcript_duration_seconds": [duration]
                })
                
                # Join with combined dataframe
                combined_df = pd.concat([combined_df, summary_df], axis=1)
            
            # Add to list of combined dataframes
            combined_dfs.append(combined_df)
        except Exception as e:
            logger.error(f"Error combining features for {video_name}: {e}")
    
    if not combined_dfs:
        logger.error("No features were successfully combined.")
        return False
    
    # Concatenate all combined dataframes
    final_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(results_dir, output_file)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Pipeline output saved to {output_path}")
    
    # Also create a simple summary text file
    create_summary_report(final_df, results_dir)
    
    return True

def create_summary_report(df: pd.DataFrame, results_dir: str) -> None:
    """
    Create a simple text summary of the pipeline results.
    
    Args:
        df (DataFrame): Combined features dataframe
        results_dir (str): Directory to save the summary
    """
    try:
        summary_path = os.path.join(results_dir, "pipeline_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=== Video Data Processing Pipeline Summary ===\n\n")
            f.write(f"Total videos processed: {len(df)}\n\n")
            
            # Write video names
            f.write("Processed videos:\n")
            for idx, video_name in enumerate(df['video_name'], 1):
                f.write(f"{idx}. {video_name}\n")
            
            f.write("\n=== Feature Statistics ===\n")
            
            # Count audio and video features
            audio_cols = [col for col in df.columns if col.startswith('audio_')]
            video_cols = [col for col in df.columns if col.startswith('video_') or col.startswith('MELD_')]
            
            f.write(f"\nAudio features extracted: {len(audio_cols)}")
            f.write(f"\nVideo features extracted: {len(video_cols)}")
            
            # Add some sample features if available
            if 'audio_pitch_mean' in df.columns:
                f.write("\n\nSample audio metrics (averages across videos):")
                f.write(f"\n- Average pitch: {df['audio_pitch_mean'].mean():.2f}")
            
            if 'MELD_unique_words' in df.columns:
                f.write("\n\nSample MELD metrics (averages across videos):")
                f.write(f"\n- Average unique words: {df['MELD_unique_words'].mean():.2f}")
                f.write(f"\n- Average utterance duration: {df['MELD_avg_utterance_duration'].mean():.2f} seconds")
            
            f.write("\n\n=== End of Summary ===\n")
        
        logger.info(f"Summary report saved to {summary_path}")
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")

def create_merged_features_json(results_dir: str, output_file: str = "merged_features.json") -> bool:
    """
    Create a comprehensive JSON file containing merged data from all pipeline outputs.
    This preserves more detailed information than the CSV format.
    
    Args:
        results_dir: Directory containing all the pipeline results
        output_file: Name of the output JSON file
    
    Returns:
        True if JSON was created successfully, False otherwise
    """
    # This method would be similar to create_pipeline_output but preserves more
    # complex data structures by using JSON instead of flattening to CSV
    # Implementation would be added as needed
    return False
