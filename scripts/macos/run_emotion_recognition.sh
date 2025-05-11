#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing Emotion Recognition ==="
echo "This script analyzes emotions in video files."

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
    poetry install --with emotion --with common || {
        echo "Poetry installation had issues. Retrying with common dependencies only..."
        poetry install --with common
    }
fi

# Help message if --help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./run_emotion_recognition.sh [options] <video_file(s)>"
    echo ""
    echo "Options:"
    echo "  --input-dir DIR      Directory containing input video files"
    echo "  --output-dir DIR     Directory to save emotion analysis results (default: ./output/emotions)"
    echo "  --batch              Process all videos in input directory"
    echo "  --interactive        Force interactive video selection mode"
    echo "  --debug              Enable debug logging"
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive video selection menu."
    exit 0
fi

# Look for input directory in arguments
input_dir=""
output_dir=""
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
    else
        other_args+=("$arg")
    fi
    i=$((i+1))
done

# Run the emotion recognition script
echo -e "\n[2/2] Running emotion recognition analysis..."

# Build command based on input parameters
cmd_args=()

# Add input directory if specified
if [ -n "$input_dir" ]; then
    echo "Using input directory: $input_dir"
    cmd_args+=("--input-dir" "$input_dir")
fi

# Add output directory if specified
if [ -n "$output_dir" ]; then
    cmd_args+=("--output-dir" "$output_dir")
fi

# Always add pose estimation by default (unless --no-pose is explicitly included)
NO_POSE_PRESENT=false
for arg in "${other_args[@]}"; do
  if [ "$arg" == "--no-pose" ]; then
    NO_POSE_PRESENT=true
    break
  fi
done

if [ "$NO_POSE_PRESENT" = false ]; then
    cmd_args+=("--with-pose")
fi

# Add other arguments
for arg in "${other_args[@]}"; do
    cmd_args+=("$arg")
done

# Use Poetry to run the script
if [ ${#cmd_args[@]} -eq 0 ] && [ ${#other_args[@]} -eq 0 ]; then
    echo "Entering interactive mode with pose estimation..."
    poetry run python -m src.emotion_recognition.cli --with-pose --interactive
else
    # Otherwise, pass all arguments to the script
    poetry run python -m src.emotion_recognition.cli "${cmd_args[@]}"
fi

if [ $? -eq 0 ]; then
    echo -e "\nEmotion recognition analysis completed successfully."
else
    echo -e "\nAn error occurred during emotion recognition analysis."
    exit 1
fi
