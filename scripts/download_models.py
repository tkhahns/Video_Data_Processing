#!/usr/bin/env python3
"""
Script to download model files required for video feature extraction.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project directory to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import model downloader
from src.emotion_and_pose_recognition.models.model_downloader import download_models, MODEL_RESOURCES

def main():
    parser = argparse.ArgumentParser(description="Download model files for video feature extraction")
    parser.add_argument("--models", nargs="+", help="Specific models to download", 
                      choices=list(MODEL_RESOURCES.keys()) + ["all"])
    parser.add_argument("--output-dir", help="Directory to store downloaded models")
    args = parser.parse_args()
    
    # Set output directory if provided
    if args.output_dir:
        os.environ["VIDEO_MODELS_DIR"] = args.output_dir
    
    model_names = None
    if args.models and "all" not in args.models:
        model_names = args.models
    
    print(f"Starting download of {'all' if not model_names else ', '.join(model_names)} models...")
    results = download_models(model_names)
    
    # Print summary
    print("\nDownload Results:")
    for model_name, result in results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        path = result["path"] or "N/A"
        print(f"- {model_name}: {status}")
    
    # Overall success/failure
    if all(result["success"] for result in results.values()):
        print("\nAll models downloaded successfully!")
        return 0
    else:
        print("\nSome models failed to download. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
