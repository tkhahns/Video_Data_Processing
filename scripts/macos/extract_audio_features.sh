#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing: Audio Feature Extraction ==="
echo "This script extracts features from audio files."

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
    echo -e "\nUsage: ./extract_audio_features.sh [options]"
    echo ""
    echo "Options:"
    echo "  --input-dir DIR      Directory containing input audio files"
    echo "  --audio FILE         Single audio file to process (can be specified multiple times)"
    echo "  --output-dir DIR     Directory to save feature files (default: ./output/audio_features)"
    echo "  --batch              Process all files without manual selection"
    echo "  --debug              Enable debug logging"
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive audio selection menu."
    exit 0
fi

# Look for input parameters in arguments
input_dir=""
output_dir=""
batch_mode=false
audio_files=()
other_args=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" == "--input-dir" ] && [ $i -lt $# ]; then
        i=$((i+1))
        input_dir="${!i}"
    elif [ "$arg" == "--output-dir" ] && [ $i -lt $# ]; then
        i=$((i+1))
        output_dir="${!i}"
    elif [ "$arg" == "--batch" ]; then
        batch_mode=true
    elif [ "$arg" == "--audio" ] && [ $i -lt $# ]; then
        i=$((i+1))
        audio_files+=("${!i}")
    else
        other_args+=("$arg")
    fi
    i=$((i+1))
done

# Run the audio feature extraction script
echo -e "\n[2/2] Running audio feature extraction..."

# Build command based on input parameters
cmd_args=()

# Add input directory if specified and no audio files were provided
if [ -n "$input_dir" ] && [ ${#audio_files[@]} -eq 0 ]; then
    echo "Using input directory: $input_dir"
    cmd_args+=("--input-dir" "$input_dir")
fi

# Add individual audio files if provided
for audio in "${audio_files[@]}"; do
    echo "Processing audio file: $audio"
    cmd_args+=("--audio" "$audio")
done

# Add output directory if specified
if [ -n "$output_dir" ]; then
    cmd_args+=("--output-dir" "$output_dir")
fi

# Add batch mode flag if specified
if [ "$batch_mode" = true ]; then
    echo "Running in batch mode - processing all files without manual selection"
    cmd_args+=("--batch")
fi

# Add other arguments
for arg in "${other_args[@]}"; do
    cmd_args+=("$arg")
done

# Use Poetry to run the script
if [ ${#cmd_args[@]} -eq 0 ] && [ ${#other_args[@]} -eq 0 ]; then
    echo "Entering interactive mode..."
    poetry run python -m src.speech_to_text.speech_features --interactive
else
    # Otherwise, pass all arguments to the script
    poetry run python -m src.speech_to_text.speech_features "${cmd_args[@]}"
fi

if [ $? -eq 0 ]; then
    echo -e "\nAudio feature extraction completed successfully."
else
    echo -e "\nAn error occurred during audio feature extraction."
    exit 1
fi
