"""
Script to download models from Hugging Face for the Video Data Processing project.
"""
import os
import csv
import argparse
import logging
import re
import subprocess
import urllib.request
import urllib.error
import shutil
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
    huggingface_patterns = [
        "huggingface.co",
        "hf.co",
    ]
    
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
    model_mappings = {
        "ALBERT": "albert-base-v2",
        "DeBERTa": "microsoft/deberta-base",
        "XLSR": "facebook/wav2vec2-large-xlsr-53",
        "Sentence-BERT": "sentence-transformers/all-MiniLM-L6-v2",
        "WhisperX": "openai/whisper-small",
        "SimCSE": "princeton-nlp/sup-simcse-bert-base-uncased",
    }
    
    if model_name in model_mappings:
        return model_mappings[model_name]
    
    if "bert" in model_name.lower() and "text" in model_type.lower():
        return "bert-base-uncased"
    elif "whisper" in model_name.lower():
        return "openai/whisper-base"
    
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
        
        try:
            model = AutoModel.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            logger.info(f"Successfully downloaded {model_name} using AutoModel")
            return True
        except Exception as e:
            logger.warning(f"AutoModel download failed: {e}")
        
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

def is_github_url(url):
    """Check if a URL is a GitHub repository."""
    if not url or url == "NA" or isinstance(url, float):
        return False
    
    github_patterns = [
        r"github\.com/[\w-]+/[\w-]+",
        r"github\.io/[\w-]+",
    ]
    
    for pattern in github_patterns:
        if re.search(pattern, str(url)):
            return True
    return False

def clone_github_repo(url, model_name, output_dir):
    """Clone a GitHub repository."""
    model_dir = os.path.join(output_dir, model_name.replace(" ", "_").lower())
    ensure_dir_exists(model_dir)
    
    if "github.io" in url:
        repo_name = re.search(r"([\w-]+)\.github\.io(?:/(.+))?", url)
        if repo_name:
            username = repo_name.group(1)
            project = repo_name.group(2) or username
            url = f"https://github.com/{username}/{project}"
    
    url = url.rstrip("/")
    if not url.endswith(".git"):
        url = url + ".git"
    
    try:
        logger.info(f"Cloning GitHub repository: {url} to {model_dir}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, model_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logger.info(f"Successfully cloned {model_name} from GitHub")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        return False

def is_direct_download_url(url):
    """Check if a URL is a direct download link."""
    if not url or url == "NA" or isinstance(url, float):
        return False
        
    download_patterns = [
        r"\.zip$", r"\.tar\.gz$", r"\.tgz$", r"\.pb$", 
        r"\.pth$", r"\.pt$", r"\.onnx$", r"\.bin$",
        r"download", r"releases/download"
    ]
    
    for pattern in download_patterns:
        if re.search(pattern, str(url).lower()):
            return True
    return False

def download_from_url(url, model_name, output_dir):
    """Download a model from a direct URL."""
    model_dir = os.path.join(output_dir, model_name.replace(" ", "_").lower())
    ensure_dir_exists(model_dir)
    
    try:
        filename = os.path.basename(url) or "model.bin"
        output_path = os.path.join(model_dir, filename)
        
        logger.info(f"Downloading from URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        logger.info(f"Successfully downloaded {model_name} to {output_path}")
        return True
    except urllib.error.URLError as e:
        logger.error(f"URL error: {e.reason}")
        return False
    except Exception as e:
        logger.error(f"Error downloading from URL: {e}")
        return False

def download_from_website(website, model_name, model_type, output_dir):
    """Try to download a model from its website."""
    if not website or website == "NA" or isinstance(website, float):
        logger.warning(f"No valid website provided for {model_name}")
        return False
    
    website = str(website).strip()
    
    if is_github_url(website):
        return clone_github_repo(website, model_name, output_dir)
    elif is_direct_download_url(website):
        return download_from_url(website, model_name, output_dir)
    elif is_huggingface_model(website, model_name):
        model_id = website.split('/')[-1] if '/' in website else get_model_id(model_name, model_type)
        return download_model(model_id, model_name, output_dir)
    else:
        logger.warning(f"Website format not recognized for {model_name}: {website}")
        return False

def main():
    """Main function to download models."""
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
    
    ensure_dir_exists(args.output_dir)
    
    models_df = parse_models_csv(args.csv_path)
    if models_df is None:
        return
    
    if "all" not in args.model_types:
        models_df = models_df[models_df["Type"].str.lower().isin([t.lower() for t in args.model_types])]
        logger.info(f"Filtered to {len(models_df)} models of types: {', '.join(args.model_types)}")
    
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
            
        model_dir = os.path.join(args.output_dir, model_name.replace(" ", "_").lower())
        if os.path.exists(model_dir) and not args.force:
            logger.info(f"Skipping {model_name} - directory already exists (use --force to override)")
            skipped_downloads += 1
            continue
            
        if args.dry_run:
            logger.info(f"[DRY RUN] Would download {model_name} from {website}")
            continue
        
        if download_from_website(website, model_name, model_type, args.output_dir):
            successful_downloads += 1
        elif is_huggingface_model(website, model_name):
            model_id = get_model_id(model_name, model_type)
            if download_model(model_id, model_name, args.output_dir):
                successful_downloads += 1
            else:
                failed_downloads += 1
        else:
            logger.warning(f"Unable to download {model_name} - no valid source identified")
            failed_downloads += 1
    
    if args.dry_run:
        logger.info(f"[DRY RUN] Would download {len(models_df)} models")
    else:
        logger.info(f"Download summary: {successful_downloads} successful, {failed_downloads} failed, {skipped_downloads} skipped")

if __name__ == "__main__":
    main()