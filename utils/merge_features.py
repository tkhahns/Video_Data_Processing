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
        import os
        import pandas as pd
        import glob
        from pathlib import Path
        import datetime
        
        results_dir = Path(results_dir)
        output_file = results_dir / "pipeline_output.csv"
        summary_file = results_dir / "pipeline_summary.txt"
        history_file = Path(os.path.join(results_dir.parent, "pipeline_history.csv"))
        combined_file = Path(os.path.join(results_dir.parent, "combined_features.csv"))
        
        print(f"Creating pipeline output CSV at {output_file}")
        print(f"Looking for feature files in {results_dir}")
        
        # Get list of all feature directories
        feature_dirs = []
        
        # Add specified feature directories if they exist
        if speech_features_dir and os.path.exists(speech_features_dir):
            feature_dirs.append(Path(speech_features_dir))
            print(f"Added speech features directory: {speech_features_dir}")
        if video_features_dir and os.path.exists(video_features_dir):
            feature_dirs.append(Path(video_features_dir))
            print(f"Added video features directory: {video_features_dir}")
        if multimodal_features_dir and os.path.exists(multimodal_features_dir):
            feature_dirs.append(Path(multimodal_features_dir))
            print(f"Added multimodal features directory: {multimodal_features_dir}")
        
        # Also look in results directory for any feature files
        feature_dirs.append(results_dir)
        
        # Find all feature CSV files
        feature_files = []
        for directory in feature_dirs:
            csv_files = list(directory.glob("**/*_features.csv")) + list(directory.glob("**/*.features.csv"))
            feature_files.extend(csv_files)
        
        print(f"Found {len(feature_files)} feature files")
        
        # Create a list to hold dataframes
        dfs = []
        
        # Read each file and extract the video name
        for file in feature_files:
            try:
                df = pd.read_csv(file)
                if len(df) == 0:
                    print(f"Skipping empty file: {file}")
                    continue
                
                # Ensure there's a video_name column
                if 'video_name' not in df.columns:
                    # Try to extract video name from filename
                    video_name = file.stem
                    if '_features' in video_name:
                        video_name = video_name.replace('_features', '')
                    if '.features' in video_name:
                        video_name = video_name.replace('.features', '')
                    
                    df['video_name'] = video_name
                
                # Add file type as a column
                file_type = "unknown"
                if "audio" in str(file) or "speech" in str(file):
                    file_type = "audio"
                elif "video" in str(file):
                    file_type = "video"
                elif "multimodal" in str(file):
                    file_type = "multimodal"
                
                df['feature_type'] = file_type
                df['feature_source'] = str(file)
                
                # If it's an aggregate file with only one row, replicate for all videos
                if len(df) == 1:
                    dfs.append(df)
                else:
                    # Skip non-aggregate files or handle as needed
                    print(f"Using file: {file} with {len(df)} rows")
                    if len(df) < 100:  # If it's a small file, include it
                        dfs.append(df)
                    else:
                        # For large files, compute aggregate statistics
                        agg_df = pd.DataFrame([{'video_name': df['video_name'].iloc[0]}])
                        
                        # Compute aggregate statistics for numerical columns
                        for col in df.columns:
                            if col not in ['frame_id', 'timestamp', 'video_name'] and pd.api.types.is_numeric_dtype(df[col]):
                                try:
                                    agg_df[f"{col}_mean"] = df[col].mean()
                                    agg_df[f"{col}_min"] = df[col].min()
                                    agg_df[f"{col}_max"] = df[col].max()
                                    agg_df[f"{col}_std"] = df[col].std()
                                except:
                                    pass
                        
                        dfs.append(agg_df)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                import traceback
                print(traceback.format_exc())
        
        # If no valid dataframes found, exit
        if not dfs:
            print("No valid feature files found. Cannot create pipeline output.")
            return False
        
        # Merge all dataframes
        try:
            # Group dataframes by video name
            video_dfs = {}
            for df in dfs:
                video_name = df['video_name'].iloc[0]
                if video_name not in video_dfs:
                    video_dfs[video_name] = []
                video_dfs[video_name].append(df)
            
            # Merge dataframes for each video
            merged_dfs = []
            for video_name, video_df_list in video_dfs.items():
                # Start with first dataframe
                merged_df = video_df_list[0].copy()
                
                # Add columns from other dataframes
                for df in video_df_list[1:]:
                    # Get columns not in merged_df
                    new_columns = [col for col in df.columns if col not in merged_df.columns]
                    for col in new_columns:
                        merged_df[col] = df[col].iloc[0] if len(df) == 1 else df[col]
                
                merged_dfs.append(merged_df)
            
            # Concatenate all merged dataframes
            final_df = pd.concat(merged_dfs, ignore_index=True)
            
            # Add timestamp column
            final_df['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save the pipeline output
            final_df.to_csv(output_file, index=False)
            print(f"Pipeline output saved to {output_file}")
            
            # Update the combined features file
            if os.path.exists(combined_file):
                combined_df = pd.read_csv(combined_file)
                combined_df = pd.concat([combined_df, final_df], ignore_index=True)
            else:
                combined_df = final_df
                
            combined_df.to_csv(combined_file, index=False)
            print(f"Combined features updated: {combined_file}")
            
            # Create a summary file
            with open(summary_file, 'w') as f:
                f.write(f"Pipeline Summary\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Features by source:\n")
                for file_type in final_df['feature_type'].unique():
                    type_cols = [col for col in final_df.columns if col not in ['frame_id', 'timestamp', 'video_name', 'feature_type', 'feature_source']]
                    f.write(f"- {file_type}: {len(type_cols)} features\n")
                
                f.write(f"\nVideo files processed:\n")
                for video in final_df['video_name'].unique():
                    f.write(f"- {video}\n")
                
                f.write(f"\nTotal features: {len(final_df.columns) - 4}\n")
            
            print(f"Summary file created: {summary_file}")
            
            # Update pipeline history
            history_entry = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'directory': str(results_dir),
                'videos_processed': len(final_df['video_name'].unique()),
                'features_extracted': len(final_df.columns) - 4
            }
            
            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file)
                history_df = pd.concat([history_df, pd.DataFrame([history_entry])], ignore_index=True)
            else:
                history_df = pd.DataFrame([history_entry])
                
            history_df.to_csv(history_file, index=False)
            print(f"Pipeline history updated: {history_file}")
            
            return True
            
        except Exception as e:
            print(f"Error merging dataframes: {e}")
            import traceback
            print(traceback.format_exc())
            return False
        
    except Exception as e:
        print(f"Error creating pipeline output: {e}")
        import traceback
        print(traceback.format_exc())
        return False
        
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
