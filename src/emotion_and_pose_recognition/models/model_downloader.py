"""
Utility for downloading model files from various sources.
"""

import os
import logging
import hashlib
import requests
import zipfile
import tarfile
import gdown
import torch
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download, login

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

# Define model resource URLs and file info
MODEL_RESOURCES = {
    "pare": {
        "hf_repo_id": "open-mmlab/pare",
        "hf_filename": "pare_checkpoint.pth",
        "local_path": "models/pare/pare_checkpoint.pth",
        "md5": None  # MD5 hash for verification (optional)
    },
    "vitpose": {
        "hf_repo_id": "open-mmlab/vitpose",
        "hf_filename": "vitpose_base_coco.pth",
        "local_path": "models/vitpose/vitpose_base_coco.pth",
        "md5": None
    },
    "psa": {
        "hf_repo_id": "open-mmlab/psa",
        "hf_filename": "psa_model.pth",
        "local_path": "models/psa/psa_model.pth",
        "md5": None
    },
    "rsn": {
        "hf_repo_id": "open-mmlab/rsn",
        "hf_filename": "rsn_model.pth",
        "local_path": "models/rsn/rsn_model.pth",
        "md5": None
    },
    "au_detector": {
        "hf_repo_id": "open-mmlab/au-detector",
        "hf_filename": "au_detector_model.pth",
        "local_path": "models/au_detector/au_detector_model.pth",
        "md5": None
    },
    "dan": {
        "hf_repo_id": "open-mmlab/dan",
        "hf_filename": "dan_model.pth",
        "local_path": "models/dan/dan_model.pth", 
        "md5": None
    },
    "fact": {
        "hf_repo_id": "open-mmlab/fact",
        "hf_filename": "fact_model.pth",
        "local_path": "models/fact/fact_model.pth",
        "md5": None
    },
    "eln": {
        "hf_repo_id": "open-mmlab/eln",
        "hf_filename": "eln_model.pth",
        "local_path": "models/eln/eln_model.pth",
        "md5": None
    }
}

def get_model_dir():
    """Get the directory where models are stored."""
    # Use an environment variable if set, otherwise use default
    models_dir = os.environ.get("VIDEO_MODELS_DIR", None)
    if not models_dir:
        # Use a directory in the project root
        project_root = Path(__file__).parent.parent.parent.parent
        models_dir = os.path.join(project_root, "resources", "models")
        
    # Create the directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def download_file(url, destination, description=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    desc = description or os.path.basename(destination)
    with open(destination, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))

def verify_checksum(file_path, expected_md5):
    """Verify file integrity using MD5 hash."""
    if not expected_md5:
        return True  # Skip verification if no hash provided
        
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5

def extract_archive(archive_path, extract_dir):
    """Extract zip or tar archive to specified directory."""
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r:') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        logger.warning(f"Unknown archive format: {archive_path}")
        return False
    return True

def download_from_huggingface(repo_id, filename, local_path):
    """Download a file from Hugging Face Hub."""
    try:
        # Check if HF token is available in the environment
        token = os.environ.get("HUGGINGFACE_TOKEN")
        
        # Try login with token if available
        if token:
            login(token=token, add_to_git_credential=False)
            logger.info(f"Logged in to Hugging Face with provided token")
        
        # Download the file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            use_auth_token=True if token else None,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False
        )
        
        # Rename if needed to match expected local path
        if file_path != local_path and os.path.exists(file_path):
            os.rename(file_path, local_path)
            
        return local_path
    except Exception as e:
        logger.error(f"Error downloading from HuggingFace: {e}")
        return None

def download_from_google_drive(file_id, destination):
    """Download a file from Google Drive."""
    try:
        gdown.download(id=file_id, output=destination, quiet=False)
        return destination
    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {e}")
        return None

def download_model(model_name):
    """Download a model by name from the configured source."""
    if model_name not in MODEL_RESOURCES:
        logger.error(f"Model {model_name} not found in resources")
        return None
        
    resource = MODEL_RESOURCES[model_name]
    models_dir = get_model_dir()
    local_path = os.path.join(models_dir, resource["local_path"])
    
    # Create the model's directory if needed
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(local_path):
        # Verify checksum if provided
        if resource.get("md5") and not verify_checksum(local_path, resource["md5"]):
            logger.warning(f"Checksum verification failed for {model_name}. Redownloading...")
        else:
            logger.info(f"Model {model_name} already exists at {local_path}")
            return local_path
    
    # Download based on the source type
    if "hf_repo_id" in resource and "hf_filename" in resource:
        logger.info(f"Downloading {model_name} model from Hugging Face Hub...")
        result = download_from_huggingface(
            resource["hf_repo_id"], 
            resource["hf_filename"], 
            local_path
        )
    elif "url" in resource:
        logger.info(f"Downloading {model_name} model from URL...")
        download_file(resource["url"], local_path, f"Downloading {model_name}")
        result = local_path if os.path.exists(local_path) else None
    elif "gdrive_id" in resource:
        logger.info(f"Downloading {model_name} model from Google Drive...")
        result = download_from_google_drive(resource["gdrive_id"], local_path)
    else:
        logger.error(f"No download source specified for {model_name}")
        return None
    
    # Verify the download
    if not result or not os.path.exists(local_path):
        logger.error(f"Failed to download {model_name}")
        return None
    
    # Extract if it's an archive
    if resource.get("is_archive") and any(local_path.endswith(ext) for ext in ['.zip', '.tar.gz', '.tgz', '.tar']):
        extract_dir = os.path.dirname(local_path)
        logger.info(f"Extracting {model_name} archive...")
        if not extract_archive(local_path, extract_dir):
            return None
    
    logger.info(f"Model {model_name} downloaded successfully")
    return local_path

def download_models(model_names=None):
    """Download multiple models by name."""
    if model_names is None:
        # Download all models if none specified
        model_names = list(MODEL_RESOURCES.keys())
    
    results = {}
    for model_name in model_names:
        logger.info(f"Downloading {model_name} model...")
        path = download_model(model_name)
        results[model_name] = {"success": path is not None, "path": path}
    
    return results
