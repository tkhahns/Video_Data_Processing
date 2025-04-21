#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing Model Downloader ==="
echo "This script will set up the environment and download required models."

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
pip install huggingface-hub pandas transformers

# Ensure models directory exists
echo -e "\n[4/5] Creating models directory if needed..."
mkdir -p ./models/downloaded

# Run the download script
echo -e "\n[5/5] Downloading models..."
python download_models.py

echo -e "\n✅ Download process completed successfully!"
echo "You can now use the models for video data processing."
