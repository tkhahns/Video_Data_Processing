#!/bin/bash

# Exit on error
set -e

# Add trap to handle Ctrl+C gracefully
trap 'echo "Caught SIGINT — shutting down…"; kill 0; exit 1' SIGINT

echo "=== Video Data Processing - Feature Extraction ==="
echo "This script extracts features from separated speech audio files."

# Get the script's directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "\nPoetry is not installed. Installing poetry is required for dependency management."
    echo "Please install Poetry with: curl -sSL https://install.python-poetry.org | python3 -"
    echo "All dependencies are defined in pyproject.toml"
    exit 1
else
    # Install dependencies using Poetry
    echo -e "\n[1/2] Installing dependencies with Poetry..."
    poetry install --with speech --with common || {
        echo "Poetry installation had issues. Retrying with common dependencies only..."
        poetry install --with common
    }
fi

# Help message if --help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./run_feature_extraction.sh [options]"
    echo ""
    echo "Options:"
    echo "  --input-dir DIR      Directory containing input audio files (WAV files from separate_speech)"
    echo "  --output-dir DIR     Directory to save output features (default: ./output/features)"
    echo "  --output-csv FILE    Filename for the combined CSV output (default: all_features.csv)"
    echo "  --debug              Enable debug logging"
    echo "  --help               Show this help message"
    echo ""
    exit 0
fi

# Parse command line arguments
input_dir=""
output_dir="output/features"
output_csv="all_features.csv"
debug=""

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            input_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --output-csv)
            output_csv="$2"
            shift 2
            ;;
        --debug)
            debug="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if input directory is provided
if [ -z "$input_dir" ]; then
    echo "Error: Input directory must be specified with --input-dir"
    exit 1
fi

# Run the feature extraction
echo -e "\n[2/2] Running feature extraction..."
poetry run python -m src.output_features.main --input-dir "$input_dir" --output-dir "$output_dir" --output-csv "$output_csv" $debug

if [ $? -eq 0 ]; then
    echo -e "\nFeature extraction completed successfully."
    echo "Results saved to: $output_dir/$output_csv"
else
    echo -e "\nAn error occurred during feature extraction."
    exit 1
fi
