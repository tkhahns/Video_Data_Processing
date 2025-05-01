"""
Main module for downloading models from various sources.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Handle imports differently when run as script vs. as module
if __name__ == "__main__":
    # Add the parent directory to sys.path for direct script execution
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, parent_dir)
    
    from src.download_models.utils import ensure_dir_exists, MODELS_DIR
    from src.download_models.parsers import parse_models_csv, is_huggingface_model, get_model_id
    from src.download_models.sources import download_from_website, download_model
else:
    # Use relative imports when imported as a module
    from .utils import ensure_dir_exists, MODELS_DIR
    from .parsers import parse_models_csv, is_huggingface_model, get_model_id
    from .sources import download_from_website, download_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_download(csv_path, output_dir=str(MODELS_DIR), model_types=None, force=False, dry_run=False):
    """Core download function that can be called from various entry points."""
    if model_types is None:
        model_types = ["all"]
    
    ensure_dir_exists(output_dir)
    
    models_df = parse_models_csv(csv_path)
    if models_df is None:
        return 1
    
    if "all" not in model_types:
        models_df = models_df[models_df["Type"].str.lower().isin([t.lower() for t in model_types])]
        logger.info(f"Filtered to {len(models_df)} models of types: {', '.join(model_types)}")
    
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0
    
    for _, row in models_df.iterrows():
        model_name = row.get("Name of the Model", "")
        model_type = row.get("Type", "")
        website = row.get("Website", "")
        
        if not model_name or model_name == "-":
            logger.info(f"Skipping row with no model name")
            continue
            
        model_dir = os.path.join(output_dir, model_name.replace(" ", "_").lower())
        if os.path.exists(model_dir) and not force:
            logger.info(f"Skipping {model_name} - directory already exists (use --force to override)")
            skipped_downloads += 1
            continue
            
        if dry_run:
            logger.info(f"[DRY RUN] Would download {model_name} from {website}")
            continue
        
        if download_from_website(website, model_name, model_type, output_dir):
            successful_downloads += 1
        elif is_huggingface_model(website, model_name):
            model_id = get_model_id(model_name, model_type)
            if download_model(model_id, model_name, output_dir):
                successful_downloads += 1
            else:
                failed_downloads += 1
        else:
            logger.warning(f"Unable to download {model_name} - no valid source identified")
            failed_downloads += 1
    
    if dry_run:
        logger.info(f"[DRY RUN] Would download {len(models_df)} models")
    else:
        logger.info(f"Download summary: {successful_downloads} successful, {failed_downloads} failed, {skipped_downloads} skipped")
    
    return 0 if failed_downloads == 0 else 1

def main():
    """Main function to download models from various sources."""
    parser = argparse.ArgumentParser(description="Download models from their source websites or Hugging Face")
    parser.add_argument(
        "--csv-path", 
        default="./models/reference_list_pre_trained_model.csv",
        help="Path to the CSV file containing model references"
    )
    parser.add_argument(
        "--output-dir", 
        default=str(MODELS_DIR),
        help="Directory to save the downloaded models"
    )
    parser.add_argument(
        "--model-types", 
        nargs="+", 
        default=["all"],
        help="Types of models to download (audio, video, image, etc.)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model directory exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    args = parser.parse_args()
    
    return run_download(
        args.csv_path,
        args.output_dir,
        args.model_types,
        args.force,
        args.dry_run
    )

if __name__ == "__main__":
    sys.exit(main())
