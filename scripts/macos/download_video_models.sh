#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing: Model Downloader ==="
echo "This script downloads model files required for video feature extraction."

# Get the script's directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root
cd "$PROJECT_ROOT"

# Check for Hugging Face token in environment
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "No Hugging Face token found in environment. You may face download limitations."
    echo "To use your token, set the HUGGINGFACE_TOKEN environment variable."
    
    # Check if we're running in an interactive terminal
    if [ -t 0 ]; then
        echo -e "\nWould you like to enter your Hugging Face token now? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            read -sp "Enter your Hugging Face token: " token
            echo ""
            if [ -n "$token" ]; then
                export HUGGINGFACE_TOKEN="$token"
                echo "Token set for this session"
            fi
        fi
    fi
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "\nPoetry is not installed. Installing poetry is required for dependency management."
    echo "Please install Poetry with: curl -sSL https://install.python-poetry.org | python3 -"
    echo "All dependencies are defined in pyproject.toml"
    exit 1
else
    # Install dependencies using Poetry
    echo -e "\n[1/2] Installing dependencies with Poetry..."
    poetry install --with emotion --with common || {
        echo "Poetry installation had issues. Retrying with common dependencies only..."
        poetry install --with common
    }
fi

# Parse command line arguments
output_dir=""
models=""
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" == "--output-dir" ] && [ $i -lt $# ]; then
        i=$((i+1))
        output_dir="${!i}"
    elif [ "$arg" == "--models" ] && [ $i -lt $# ]; then
        i=$((i+1))
        models="${!i}"
    fi
    i=$((i+1))
done

# Build download command
cmd_args=()

if [ -n "$output_dir" ]; then
    cmd_args+=("--output-dir" "$output_dir")
fi

if [ -n "$models" ]; then
    cmd_args+=("--models" $models)
else
    cmd_args+=("--models" "all")
fi

# Run the download script
echo -e "\n[2/2] Downloading model files..."
poetry run python scripts/download_models.py "${cmd_args[@]}"

if [ $? -eq 0 ]; then
    echo -e "\nModel download completed successfully."
else
    echo -e "\nAn error occurred during model download."
    exit 1
fi
