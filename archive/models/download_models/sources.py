"""
Functions for downloading models from various sources.
"""
import os
import re
import logging
import subprocess
import urllib.request
import urllib.error
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer

from .utils import ensure_dir_exists
from .parsers import is_huggingface_model, get_model_id

logger = logging.getLogger(__name__)

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
