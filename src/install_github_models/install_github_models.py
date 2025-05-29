#!/usr/bin/env python3
"""
GitHub Model Installer for Video Data Processing
This script automates the installation of models from GitHub repositories.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import git
from tqdm import tqdm
import shutil
import json

# Configuration for all GitHub models
MODELS = [
    {
        "name": "FACT",
        "repo": "https://github.com/ZijiaLewisLu/CVPR2024-FACT",
        "description": "Frame-Action Cross-Attention for Action Segmentation",
        "install_method": "clone",
        "category": "video"
    },
    {
        "name": "GANimation",
        "repo": "https://github.com/albertpumarola/GANimation",
        "description": "Facial Expression Animation Generator",
        "install_method": "clone",
        "category": "emotion"
    },
    {
        "name": "ARBEx",
        "repo": "https://github.com/takihasan/ARBEx",
        "description": "Attentive Feature Extraction with Reliability Balancing",
        "install_method": "clone",
        "category": "emotion"
    },
    {
        "name": "PARE",
        "repo": "https://github.com/mkocabas/PARE",
        "description": "Part Attention Regressor for 3D Human Body Estimation",
        "install_method": "clone",
        "category": "pose"
    },
    {
        "name": "Polarized-Self-Attention",
        "repo": "https://github.com/DeLightCMU/PSA",
        "description": "Polarized Self-Attention for Pose Estimation",
        "install_method": "clone",
        "category": "pose"
    },
    {
        "name": "Residual-Steps-Network",
        "repo": "https://github.com/caiyuanhao1998/RSN",
        "description": "Residual Steps Network for Pose Estimation",
        "install_method": "clone",
        "category": "pose"
    },
    {
        "name": "SmoothNet",
        "repo": "https://github.com/cure-lab/SmoothNet",
        "description": "Human Motion Smoothing",
        "install_method": "clone",  # Changed from "pip" to "clone"
        "category": "motion"
    },
    {
        "name": "LaneGCN",
        "repo": "https://github.com/uber-research/LaneGCN",
        "description": "Motion Forecasting",
        "install_method": "clone",
        "category": "motion"
    },
    {
        "name": "RIFE",
        "repo": "https://github.com/hzwer/ECCV2022-RIFE",
        "description": "Real-Time Intermediate Flow Estimation",
        "install_method": "clone",
        "category": "motion"
    },
    {
        "name": "av_hubert",
        "repo": "https://github.com/facebookresearch/av_hubert",
        "description": "Audio-Visual HuBERT",
        "install_method": "clone",
        "category": "audiovisual"
    },
    {
        "name": "AudioStretchy",
        "repo": "https://github.com/twardoch/audiostretchy",
        "description": "Time Stretching for Audio",
        "install_method": "pip",
        "category": "audio"
    },
    {
        "name": "MELD",
        "repo": "https://github.com/declare-lab/MELD",
        "description": "Multimodal Emotion Recognition in Conversation",
        "install_method": "clone",
        "category": "emotion"
    },
    {
        "name": "VideoFinder",
        "repo": "https://github.com/win4r/VideoFinder-Llama3.2-vision-Ollama",
        "description": "Video Analysis with LLM Integration",
        "install_method": "clone",  # Changed from "pip" to "clone"
        "category": "video"
    },
    {
        "name": "CrowdFlow",
        "repo": "https://github.com/tsenst/CrowdFlow",
        "description": "Optical Flow for Crowd Analysis",
        "install_method": "clone",
        "category": "motion"
    },
    {
        "name": "intelligent-video-frame-extractor",
        "repo": "https://github.com/BSM0oo/intelligent-video-frame-extractor",
        "description": "Key Frame Extraction",
        "install_method": "clone",  # Changed from "pip" to "clone"
        "category": "video"
    },
    {
        "name": "heinsen-routing",
        "repo": "https://github.com/glassroom/heinsen_routing",
        "description": "Algorithm for Routing Vectors in Sequences",
        "install_method": "clone",
        "category": "text"
    },
    {
        "name": "ViTPose",
        "repo": "https://github.com/ViTAE-Transformer/ViTPose",
        "description": "Vision Transformer for Human Pose Estimation",
        "install_method": "clone", 
        "category": "pose"
    },
    {
        "name": "fairseq",
        "repo": "https://github.com/facebookresearch/fairseq",
        "description": "XLSR and S2T models",
        "install_method": "clone",
        "category": "speech",
        "note": "Requires omegaconf<2.1, conflicts with pyannote-audio/whisperx"
    }
]

class ModelInstaller:
    def __init__(self, args):
        self.args = args
        self.base_dir = Path(args.directory)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.installed_file = self.base_dir / "installed_models.json"
        self.installed_models = self._load_installed_models()
    
    def _load_installed_models(self):
        """Load the list of already installed models."""
        if self.installed_file.exists():
            with open(self.installed_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}
    
    def _save_installed_models(self):
        """Save the list of installed models."""
        with open(self.installed_file, 'w') as f:
            json.dump(self.installed_models, f, indent=2)
    
    def install(self):
        """Install all models based on the configuration."""
        print(f"Installing GitHub models to {self.base_dir}")
        
        # Create category directories
        categories = set(model["category"] for model in MODELS)
        for category in categories:
            category_dir = self.base_dir / category
            category_dir.mkdir(exist_ok=True)
        
        for model in tqdm(MODELS, desc="Installing models"):
            self._install_model(model)
        
        self._save_installed_models()
        print(f"\n✅ Installation complete. Models installed in {self.base_dir}\n")
    
    def _install_model(self, model):
        """Install a single model."""
        name = model["name"]
        repo = model["repo"]
        category = model["category"]
        install_method = model["install_method"]
        
        # Skip if already installed and update flag not set
        if name in self.installed_models and not self.args.update:
            tqdm.write(f"Skipping {name} (already installed)")
            return
        
        # Create a directory for the model in its category
        model_dir = self.base_dir / category / name
        
        try:
            if install_method == "clone":
                self._clone_repository(repo, model_dir, name)
            elif install_method == "pip":
                self._pip_install(repo, name)
            
            # Mark as installed
            self.installed_models[name] = {
                "path": str(model_dir),
                "repo": repo,
                "category": category,
                "install_method": install_method,
                "description": model.get("description", "")
            }
            
        except Exception as e:
            tqdm.write(f"❌ Failed to install {name}: {e}")
    
    def _clone_repository(self, repo, target_dir, name):
        """Clone a repository to the target directory."""
        if target_dir.exists() and self.args.update:
            tqdm.write(f"Updating {name}...")
            try:
                repo_obj = git.Repo(target_dir)
                repo_obj.remotes.origin.pull()
            except git.GitCommandError:
                tqdm.write(f"Cannot update {name}, attempting to clone again...")
                shutil.rmtree(target_dir, ignore_errors=True)
                git.Repo.clone_from(repo, target_dir)
        elif not target_dir.exists():
            tqdm.write(f"Cloning {name}...")
            git.Repo.clone_from(repo, target_dir)
    
    def _pip_install(self, repo, name):
        """Install a package directly from GitHub using pip."""
        tqdm.write(f"Installing {name} using pip...")
        cmd = [sys.executable, "-m", "pip", "install", f"git+{repo}"]
        if self.args.update:
            cmd.append("--upgrade")
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise Exception(f"Pip installation failed: {process.stderr}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Install models from GitHub for video data processing")
    parser.add_argument("-d", "--directory", default="github_models",
                      help="Directory to install models (default: github_models)")
    parser.add_argument("-u", "--update", action="store_true",
                      help="Update existing models")
    parser.add_argument("-l", "--list", action="store_true",
                      help="List available models")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        by_category = {}
        for model in MODELS:
            category = model["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(model)
        
        for category, models in sorted(by_category.items()):
            print(f"\n{category.upper()}:")
            for model in models:
                print(f"  - {model['name']}: {model['description']}")
                print(f"    Repository: {model['repo']}")
        return

    installer = ModelInstaller(args)
    installer.install()

if __name__ == "__main__":
    main()
