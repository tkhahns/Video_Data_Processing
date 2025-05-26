#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing Speech Separation ==="
echo "This script extracts and isolates speech from video files."

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
    echo -e "\nUsage: ./run_separate_speech.sh [options] <video_file(s)>"
    echo ""
    echo "Options:"
    echo "  --input-dir DIR      Directory containing input video files (default: ./data/videos)"
    echo "  --output-dir DIR     Directory to save separated speech files (default: ./output/separated_speech)"
    echo "  --model MODEL        Speech separation model to use (sepformer, conv-tasnet)" 
    echo "  --file-type TYPE     Output file format: wav (1), mp3 (2), or both (3) (default: mp3)"
    echo "  --recursive          Process video files in subdirectories recursively"
    echo "  --debug              Enable debug logging"
    echo "  --interactive        Force interactive video selection mode"
    echo "  --batch              Run in batch mode, processing all files without manual selection"
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive video selection menu."
    exit 0
fi

# Look for input parameters in arguments
input_dir=""
output_dir=""
batch_mode=false
video_files=()
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
    elif [ "$arg" == "--video" ] && [ $i -lt $# ]; then
        i=$((i+1))
        video_files+=("${!i}")
    else
        other_args+=("$arg")
    fi
    i=$((i+1))
done

# Run the speech separation script
echo -e "\n[2/2] Running speech separation..."

# Build command based on input parameters
cmd_args=()

# Add input directory if specified and no video files were provided
if [ -n "$input_dir" ] && [ ${#video_files[@]} -eq 0 ]; then
    echo "Using input directory: $input_dir"
    cmd_args+=("--input-dir" "$input_dir")
fi

# Add individual video files if provided
if [ ${#video_files[@]} -gt 0 ]; then
    echo "Processing ${#video_files[@]} individual video files"
    for video in "${video_files[@]}"; do
        cmd_args+=("$video")
    done
fi

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
    poetry run python -m src.separate_speech --interactive
else
    # Otherwise, pass all arguments to the script
    poetry run python -m src.separate_speech "${cmd_args[@]}"
fi

if [ $? -eq 0 ]; then
    echo -e "\nSpeech separation process completed successfully."
else
    echo -e "\nAn error occurred during the speech separation process."
    exit 1
fi
