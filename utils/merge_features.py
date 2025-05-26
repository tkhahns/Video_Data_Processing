"""
Functions to merge features from different sources into a single CSV file.
"""
import os
import glob
import pandas as pd
import logging
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def create_summary_report(output_data, videos, all_csv_files, summary_file):
    """
    Create a summary report of the merged features.
    
    Args:
        output_data: DataFrame containing the merged features
        videos: Dictionary mapping video names to their CSV files
        all_csv_files: List of all CSV files that were processed
        summary_file: Path to the output summary file
        
    Returns:
        Path to the created summary file
    """
    try:
        # Create summary file
        with open(summary_file, "w") as f:
            f.write(f"Pipeline Output Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Total videos processed: {len(videos)}\n")
            f.write(f"Total feature files merged: {len(all_csv_files)}\n")
            
            if output_data is not None:
                f.write(f"Total features extracted: {len(output_data.columns)}\n\n")
            else:
                f.write(f"No features were extracted successfully\n\n")
                
            f.write(f"Videos included:\n")
            for video_name in videos.keys():
                f.write(f"- {video_name}\n")
                
            if output_data is not None and len(output_data.columns) > 0:
                f.write(f"\nFeature categories:\n")
                # Identify feature categories by prefix
                prefixes = {}
                for col in output_data.columns:
                    parts = col.split('_')
                    if len(parts) > 1:
                        prefix = parts[0]
                        if prefix not in prefixes:
                            prefixes[prefix] = 0
                        prefixes[prefix] += 1
                
                for prefix, count in prefixes.items():
                    if prefix != "video" and count > 1:  # Skip video_name
                        f.write(f"- {prefix}_*: {count} features\n")
        
        logger.info(f"Created summary report: {summary_file}")
        return str(summary_file)
    
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_pipeline_output(results_dir, speech_features_dir=None, video_features_dir=None, multimodal_features_dir=None):
    """
    Create a pipeline output CSV that merges all extracted features.
    
    Args:
        results_dir: Directory containing all pipeline results
        speech_features_dir: Directory containing speech feature CSVs
        video_features_dir: Directory containing video feature CSVs
        multimodal_features_dir: Directory containing multimodal feature CSVs
        
    Returns:
        Path to the created CSV file
    """
    try:
        results_dir = Path(results_dir)
        output_csv = results_dir / "pipeline_output.csv"
        summary_file = results_dir / "pipeline_summary.txt"
        history_csv = Path(results_dir.parent) / "pipeline_history.csv"
        
        # Get current timestamp for this run
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = int(current_time)  # Convert to integer for incrementing
        
        # Read history file if it exists to determine run counter
        run_counter = 1  # Default first run
        if history_csv.exists():
            try:
                history_df = pd.read_csv(history_csv)
                if not history_df.empty:
                    run_counter = len(history_df) + 1
            except Exception as e:
                logger.warning(f"Failed to read history file: {e}")
        
        # List of feature directories to look for CSV files
        feature_dirs = []
        
        # Add provided feature directories if specified
        if speech_features_dir and os.path.exists(speech_features_dir):
            feature_dirs.append(Path(speech_features_dir))
        else:
            # Try to find speech features directory in results directory
            potential_speech_dir = results_dir / "audio_features"
            if potential_speech_dir.exists():
                feature_dirs.append(potential_speech_dir)
        
        if video_features_dir and os.path.exists(video_features_dir):
            feature_dirs.append(Path(video_features_dir))
        else:
            # Try to find video features directory in results directory
            potential_video_dir = results_dir / "video_features"
            if potential_video_dir.exists():
                feature_dirs.append(potential_video_dir)
                
        if multimodal_features_dir and os.path.exists(multimodal_features_dir):
            feature_dirs.append(Path(multimodal_features_dir))
        else:
            # Try to find multimodal features directory in results directory
            potential_multimodal_dir = results_dir / "multimodal_features"
            if potential_multimodal_dir.exists():
                feature_dirs.append(potential_multimodal_dir)
        
        # Also check standard output locations in case features were saved there
        transcript_dir = results_dir / "transcripts"
        if transcript_dir.exists():
            feature_dirs.append(transcript_dir)
            
        emotions_dir = results_dir / "emotions_and_pose"
        if emotions_dir.exists():
            feature_dirs.append(emotions_dir)
            
        speech_dir = results_dir / "speech"
        if speech_dir.exists():
            feature_dirs.append(speech_dir)
        
        # Also search directly in results_dir for any CSVs
        feature_dirs.append(results_dir)
        
        # Find all CSV files
        all_csv_files = []
        for directory in feature_dirs:
            csv_files = list(directory.glob("**/*_features.csv"))
            all_csv_files.extend(csv_files)
            
        logger.info(f"Found {len(all_csv_files)} feature CSV files")
        
        if not all_csv_files:
            logger.warning("No feature CSV files found. Nothing to merge.")
            # Still create an empty summary
            create_summary_report(None, {}, [], summary_file)
            return None
        
        # Group CSV files by base name (video name)
        videos = {}
        for csv_file in all_csv_files:
            # Get base name without suffix (e.g., MVI_0575 from MVI_0575_speech_features.csv or MVI_0575_video_features.csv)
            base_name = csv_file.stem.split('_')[0]
            if base_name not in videos:
                videos[base_name] = []
            videos[base_name].append(csv_file)
        
        # Merge features separately for each video
        merged_dfs = []
        for video_name, csv_files in videos.items():
            speech_features_df = None
            video_features_df = None
            
            # Read and process each CSV file
            for csv_file in csv_files:
                try:
                    csv_str = str(csv_file)
                    df = pd.read_csv(csv_file)
                    
                    # Ensure there's a video_name column
                    if "video_name" not in df.columns and "file_name" not in df.columns:
                        df["video_name"] = video_name
                    elif "file_name" in df.columns and "video_name" not in df.columns:
                        # If there's a file_name but no video_name, create video_name from file_name
                        df["video_name"] = df["file_name"].apply(lambda x: x.split('_')[0] if isinstance(x, str) else video_name)
                    
                    # Categorize features based on file path
                    if "speech_features" in csv_str:
                        speech_features_df = df
                    else:
                        video_features_df = df
                        
                except Exception as e:
                    logger.error(f"Error reading {csv_file}: {e}")
            
            # Merge video and speech features side by side
            if speech_features_df is not None and video_features_df is not None:
                # Ensure they have compatible shapes
                merged_video_df = pd.concat([video_features_df, speech_features_df], axis=1)
                
                # Remove duplicate columns (like video_name)
                merged_video_df = merged_video_df.loc[:, ~merged_video_df.columns.duplicated()]
            elif speech_features_df is not None:
                merged_video_df = speech_features_df
            elif video_features_df is not None:
                merged_video_df = video_features_df
            else:
                logger.warning(f"No valid features found for {video_name}")
                continue
            
            # Add timestamp and run counter
            merged_video_df["timestamp"] = current_time
            merged_video_df["run_id"] = run_counter
            
            merged_dfs.append(merged_video_df)
        
        # Concatenate all video dataframes
        if merged_dfs:
            final_df = pd.concat(merged_dfs, ignore_index=True)
            
            # Ensure video_name is the first column
            cols = list(final_df.columns)
            if "video_name" in cols:
                cols.remove("video_name")
                cols = ["video_name"] + cols
                final_df = final_df[cols]
            
            # Save to CSV with timestamp info
            final_df.to_csv(output_csv, index=False)
            
            # Add to history file
            history_entry = pd.DataFrame({
                "timestamp": [current_time],
                "run_id": [run_counter],
                "results_dir": [str(results_dir)],
                "video_count": [len(videos)],
                "feature_count": [len(final_df.columns)]
            })
            
            # Append or create history file
            try:
                if history_csv.exists():
                    history_df = pd.read_csv(history_csv)
                    history_df = pd.concat([history_df, history_entry], ignore_index=True)
                    history_df.to_csv(history_csv, index=False)
                else:
                    history_entry.to_csv(history_csv, index=False)
                logger.info(f"Updated pipeline history in {history_csv}")
            except Exception as e:
                logger.error(f"Failed to update history file: {e}")
            
            logger.info(f"Created pipeline output CSV with {len(final_df)} rows and {len(final_df.columns)} columns")
            
            # Create combined features CSV in output directory
            combined_output_csv = Path(results_dir).parent / "combined_features.csv"
            
            try:
                # Combine with previous runs if file exists
                if combined_output_csv.exists():
                    prev_df = pd.read_csv(combined_output_csv)
                    combined_df = pd.concat([prev_df, final_df], ignore_index=True)
                    combined_df.to_csv(combined_output_csv, index=False)
                else:
                    final_df.to_csv(combined_output_csv, index=False)
                    
                logger.info(f"Updated combined features file: {combined_output_csv}")
            except Exception as e:
                logger.error(f"Error updating combined features file: {e}")
            
            # Create summary file
            create_summary_report(final_df, videos, all_csv_files, summary_file)
                
            return str(output_csv)
        else:
            logger.warning("No dataframes to merge")
            # Still create an empty summary
            create_summary_report(None, videos, all_csv_files, summary_file)
            return None
            
    except Exception as e:
        logger.error(f"Error creating pipeline output: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
        
if __name__ == "__main__":
    import argparse
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Merge feature CSV files into a single pipeline output.")
    parser.add_argument("results_dir", help="Directory containing pipeline results")
    parser.add_argument("--speech-features", help="Directory containing speech feature CSVs")
    parser.add_argument("--video-features", help="Directory containing video feature CSVs")
    
    args = parser.parse_args()
    
    result = create_pipeline_output(args.results_dir, args.speech_features, args.video_features)
    
    if result:
        print(f"Successfully created pipeline output: {result}")
        sys.exit(0)
    else:
        print("Failed to create pipeline output")
        sys.exit(1)
