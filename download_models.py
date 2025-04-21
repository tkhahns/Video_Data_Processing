"""
Script to download models from Hugging Face for the Video Data Processing project.
"""
import os
import csv
import argparse
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models directory
MODELS_DIR = Path("./models/downloaded")

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def parse_models_csv(csv_path):
    """Parse the models reference CSV file."""
    try:
        models_df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(models_df)} models from CSV")
        return models_df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return None

def is_huggingface_model(website, model_name):
    """Check if the model is available on Hugging Face."""
    # Common patterns for Hugging Face models
    huggingface_patterns = [
        "huggingface.co",
        "hf.co",
    ]
    
    # Models we know are on Hugging Face even if not explicitly mentioned
    known_hf_models = [
        "BERT", "ALBERT", "DeBERTa", "ViT", "Whisper", "XLSR", "wav2vec"
    ]
    
    if website:
        for pattern in huggingface_patterns:
            if pattern in str(website).lower():
                return True
    
    if model_name:
        for model in known_hf_models:
            if model.lower() in str(model_name).lower():
                return True
    
    return False

def get_model_id(model_name, model_type):
    """Map model names to Hugging Face model IDs."""
    # Default mappings for common models
    model_mappings = {
        "ALBERT": "albert-base-v2",
        "DeBERTa": "microsoft/deberta-base",
        "XLSR": "facebook/wav2vec2-large-xlsr-53",
        "Sentence-BERT": "sentence-transformers/all-MiniLM-L6-v2",
        "WhisperX": "openai/whisper-small",
        "SimCSE": "princeton-nlp/sup-simcse-bert-base-uncased",
    }
    
    # Check for direct mapping
    if model_name in model_mappings:
        return model_mappings[model_name]
    
    # Try to infer based on model type and name
    if "bert" in model_name.lower() and "text" in model_type.lower():
        return "bert-base-uncased"
    elif "whisper" in model_name.lower():
        return "openai/whisper-base"
    
    # Return a placeholder if no mapping found
    logger.warning(f"No direct Hugging Face mapping for {model_name}, using generic ID")
    return None

def download_model(model_id, model_name, output_dir):
    """Download a model from Hugging Face."""
    if not model_id:
        logger.warning(f"Skipping {model_name} - no valid model ID")
        return False
    
    model_dir = os.path.join(output_dir, model_name.replace(" ", "_").lower())
    ensure_dir_exists(model_dir)
    
    try:
        logger.info(f"Downloading {model_name} (ID: {model_id}) to {model_dir}")
        
        # Try to download using AutoModel
        try:
            model = AutoModel.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            logger.info(f"Successfully downloaded {model_name} using AutoModel")
            return True
        except Exception as e:
            logger.warning(f"AutoModel download failed: {e}")
        
        # Fall back to snapshot_download
        try:
            snapshot_download(repo_id=model_id, local_dir=model_dir)
            logger.info(f"Successfully downloaded {model_name} using snapshot_download")
            return True
        except Exception as e:
            logger.warning(f"snapshot_download failed: {e}")
            
        logger.error(f"Failed to download {model_name}")
        return False
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        return False

def main():
    """Main function to download models."""
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument(
        "--csv-path", 
        default="./models/models_reference_list.csv",
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
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir_exists(args.output_dir)
    
    # Parse CSV file
    models_df = parse_models_csv(args.csv_path)
    if models_df is None:
        return
    
    # Filter models by type if specified
    if "all" not in args.model_types:
        models_df = models_df[models_df["Type"].str.lower().isin([t.lower() for t in args.model_types])]
        logger.info(f"Filtered to {len(models_df)} models of types: {', '.join(args.model_types)}")
    
    # Download each model
    successful_downloads = 0
    for _, row in models_df.iterrows():
        model_name = row.get("Name of the Model", "")
        model_type = row.get("Type", "")
        website = row.get("Website", "")
        
        if is_huggingface_model(website, model_name):
            model_id = get_model_id(model_name, model_type)
            if download_model(model_id, model_name, args.output_dir):
                successful_downloads += 1
        else:
            logger.info(f"Skipping {model_name} - not a Hugging Face model")
    
    logger.info(f"Downloaded {successful_downloads} models successfully")

if __name__ == "__main__":
    main()