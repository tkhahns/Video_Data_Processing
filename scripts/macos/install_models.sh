#!/bin/bash

# Exit on error
set -e

echo "===== Video Data Processing - Model Installer ====="
echo "This script installs all required models from GitHub repositories"
echo

# Get the script's directory 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Navigate to project root (two levels up from scripts/macos)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
cd "$PROJECT_ROOT"

# Check if Poetry is installed
if ! command -v poetry &>/dev/null; then
    echo "Poetry not found. Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies with Poetry..."
poetry install --no-interaction

# Check if we should update existing models
UPDATE_FLAG=""
if [ "$1" == "--update" ]; then
    UPDATE_FLAG="--update"
    echo "Update mode enabled - will update existing models"
fi

# Create directory for GitHub models if it doesn't exist
MODELS_DIR="$PROJECT_ROOT/github_models"
mkdir -p "$MODELS_DIR"

echo "Installing models from GitHub to $MODELS_DIR..."
poetry run python -m src.install_github_models.install_github_models --directory "$MODELS_DIR" $UPDATE_FLAG

echo
echo "To install a specific model category:"
echo "  poetry run python -m src.install_github_models.install_github_models --directory github_models/specific_category"
echo
echo "To list all available models:"
echo "  poetry run python -m src.install_github_models.install_github_models --list"
echo
echo "To update all models:"
echo "  ./scripts/macos/install_models.sh --update"
