#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing - Download to Timestamped Directory ==="
echo "This script downloads videos to a timestamped directory."

# Get the script's directory (project root)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create timestamped directory
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
OUTPUT_DIR="$PROJECT_ROOT/data/downloads_$TIMESTAMP"

echo -e "\nCreating timestamped output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Poetry is installed
if command -v poetry &>/dev/null; then
    echo -e "\n[1/3] Installing dependencies using Poetry..."
    
    # Install all dependencies from pyproject.toml
    echo "Installing base dependencies..."
    poetry install --no-interaction || { echo "Failed to install base dependencies"; exit 1; }
    
    # Install specific groups
    echo "Installing common dependencies..."
    poetry install --with common --no-interaction || echo "Warning: Some common dependencies failed to install"
    
    echo "Installing emotion recognition dependencies..."
    poetry install --with emotion --no-interaction || echo "Warning: Some emotion recognition dependencies failed to install"
    
    echo "Installing speech recognition dependencies..."
    poetry install --with speech --no-interaction || echo "Warning: Some speech recognition dependencies failed to install"
    
    echo "Installing download dependencies..."
    poetry install --with download --no-interaction || echo "Warning: Some download dependencies failed to install"
    
    echo "Dependencies installation completed."
    
    # Make sure the download script is executable
    echo -e "\n[2/3] Preparing download script..."
    chmod +x "$PROJECT_ROOT/scripts/macos/run_download_videos.sh"
    
    echo -e "\n[3/3] Running video downloader with Poetry..."
    # Pass all original arguments plus the output directory
    poetry run scripts/macos/run_download_videos.sh "$@" --output-dir "$OUTPUT_DIR"
else
    echo -e "\nPoetry not found. Installing without dependency management."
    
    # Make sure the download script is executable
    chmod +x "$PROJECT_ROOT/scripts/macos/run_download_videos.sh"
    
    # Run the script directly
    echo -e "\nRunning video downloader..."
    # Pass all original arguments plus the output directory
    ./scripts/macos/run_download_videos.sh "$@" --output-dir "$OUTPUT_DIR"
fi

echo -e "\nDownload completed. Videos saved in: $OUTPUT_DIR"
