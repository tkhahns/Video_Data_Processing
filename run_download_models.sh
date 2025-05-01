#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing Model Downloader ==="
echo "This script will set up the environment and download required models."

# Parse command line arguments
model_type=""
force_download=false
dry_run=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-type)
      model_type="$2"
      shift 2
      ;;
    --force)
      force_download=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --help)
      echo -e "\nUsage: ./run_download_models.sh [options]"
      echo ""
      echo "Options:"
      echo "  --model-type TYPE    Download only models of specified type (audio, video, image)"
      echo "  --force              Force re-download even if model directory exists"
      echo "  --dry-run            Show what would be downloaded without actually downloading"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      shift
      ;;
  esac
done

# Change to the script's directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "\n[1/5] Creating virtual environment..."
    python -m venv .venv
else
    echo -e "\n[1/5] Virtual environment already exists."
fi

# Activate virtual environment
echo -e "\n[2/5] Activating virtual environment..."
source .venv/bin/activate

# Update pip and install dependencies
echo -e "\n[3/5] Updating pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface-hub pandas transformers gitpython

# Ensure models directory exists
echo -e "\n[4/5] Creating models directory if needed..."
mkdir -p ./models/downloaded

# Prepare download arguments
download_args=("./src/download_models.py")

if [ -n "$model_type" ]; then
    download_args+=("--model-types" "$model_type")
fi

if [ "$force_download" = true ]; then
    download_args+=("--force")
fi

if [ "$dry_run" = true ]; then
    download_args+=("--dry-run")
fi

# Run the download script
echo -e "\n[5/5] Downloading models..."
python "${download_args[@]}"

echo -e "\n✅ Download process completed!"
echo "You can now use the models for video data processing."