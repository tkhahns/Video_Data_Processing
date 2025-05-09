#!/bin/bash
# 
# Emotion Recognition Video Processing Script
#
# This script runs the emotion recognition module to detect and analyze 
# facial emotions in video files.
#
# Usage:
#   ./run_emotion_recognition.sh                       # Interactive mode (default)
#   ./run_emotion_recognition.sh process /path/to/video.mp4 --output /path/to/output.mp4
#   ./run_emotion_recognition.sh batch /input/dir /output/dir
#   ./run_emotion_recognition.sh interactive --input_dir /path/to/videos
#   ./run_emotion_recognition.sh check

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Project root: $PROJECT_ROOT"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to the PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    
    if ! command -v poetry &> /dev/null; then
        echo "Failed to install Poetry. Please install manually with:"
        echo "curl -sSL https://install.python-poetry.org | python3 -"
        exit 1
    fi
    echo "Poetry installed successfully."
fi

# Navigate to project root
cd "$PROJECT_ROOT" || exit 1

# Configure poetry for in-project virtualenv
poetry config virtualenvs.in-project true

# Set environment variables to suppress model download output
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore

# Install dependencies using Poetry with the emotion group
echo "Installing dependencies with Poetry..."
poetry install --with common --with emotion --no-root

# Make sure the default directories exist
DATA_DIR="${PROJECT_ROOT}/data/videos"
OUTPUT_DIR="${PROJECT_ROOT}/output/emotions"

if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Run the emotion recognition module with all arguments passed to this script
echo "Running Emotion Recognition module with body pose estimation enabled by default..."

# Check if --quiet or -q is already in the arguments
QUIET_PRESENT=false
NO_POSE_PRESENT=false

for arg in "$@"; do
  if [ "$arg" == "--quiet" ] || [ "$arg" == "-q" ]; then
    QUIET_PRESENT=true
  fi
  if [ "$arg" == "--no-pose" ]; then
    NO_POSE_PRESENT=true
  fi
done

# Prepare arguments
cmd_args=()
if [ "$NO_POSE_PRESENT" != "true" ]; then
  cmd_args+=("--with-pose")
fi
if [ "$QUIET_PRESENT" != "true" ]; then
  cmd_args+=("--quiet")  # Add quiet flag by default
fi
cmd_args+=("$@")  # Add all original arguments

# Run with prepared arguments
poetry run python -m src.emotion_recognition.cli "${cmd_args[@]}"

# Exit with the same code as the python command
exit $?
