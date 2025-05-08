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
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Install main dependencies individually to avoid conflicts
echo "Installing dependencies..."
pip3 install "numpy>=1.25.0" "pandas>=2.1.0"
pip3 install "matplotlib>=3.7.0"  # Install compatible matplotlib version first
pip3 install "opencv-python>=4.8.0.76"
pip3 install "tensorflow>=2.13.0" tf-keras  # Core dependencies for emotion recognition

# Check for main dependencies in requirements.txt first
echo "Installing dependencies from requirements.txt..."
# Fix the path to requirements.txt - use current directory
pip3 install -r requirements.txt

# Check for required DeepFace library
echo "Checking for DeepFace dependency..."
if ! python3 -c "import deepface" &> /dev/null; then
    echo "DeepFace not found. Attempting to install..."
    pip3 install deepface>=0.0.79
    
    # Verify installation was successful
    if ! python3 -c "import deepface" &> /dev/null; then
        echo "Error: Failed to install DeepFace. Please install manually with:"
        echo "pip install deepface>=0.0.79"
        exit 1
    fi
    echo "DeepFace installed successfully."
fi

# Check for TensorFlow version and install tf-keras if needed
echo "Checking TensorFlow compatibility..."
TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
if [[ $? -eq 0 ]]; then
    # Compare TensorFlow version
    MAJOR=$(echo $TF_VERSION | cut -d. -f1)
    MINOR=$(echo $TF_VERSION | cut -d. -f2)
    
    if [[ $MAJOR -eq 2 && $MINOR -ge 13 ]] || [[ $MAJOR -gt 2 ]]; then
        echo "TensorFlow $TF_VERSION detected, checking for tf-keras..."
        if ! python3 -c "import tf_keras" &> /dev/null; then
            echo "Installing tf-keras for TensorFlow compatibility..."
            pip3 install tf-keras
            
            # Verify installation
            if ! python3 -c "import tf_keras" &> /dev/null; then
                echo "Warning: Failed to install tf-keras. You may encounter errors."
            else
                echo "tf-keras installed successfully."
            fi
        else
            echo "tf-keras is already installed."
        fi
    fi
else
    echo "TensorFlow not detected, will attempt to install dependencies as needed."
fi

# Check for MediaPipe (for body pose estimation)
echo "Checking for MediaPipe dependency..."
if ! python3 -c "import mediapipe" &> /dev/null; then
    echo "MediaPipe not found. Attempting to install..."
    pip3 install mediapipe
    
    # Verify installation was successful
    if ! python3 -c "import mediapipe" &> /dev/null; then
        echo "Warning: Failed to install MediaPipe. Body pose estimation may not work properly."
    else
        echo "MediaPipe installed successfully."
    fi
else
    echo "MediaPipe is already installed."
fi

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

# Add the project root to PYTHONPATH to allow imports
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the emotion recognition module with all arguments passed to this script
echo "Running Emotion Recognition module with body pose estimation enabled by default..."

# Check if --no-pose is already in the arguments
NO_POSE_PRESENT=false
for arg in "$@"; do
  if [ "$arg" == "--no-pose" ]; then
    NO_POSE_PRESENT=true
    break
  fi
done

if [ "$NO_POSE_PRESENT" = true ]; then
    # If --no-pose is already in the arguments, don't add --with-pose
    python3 -m src.emotion_recognition.cli "$@"
else
    # Add --with-pose to arguments (it's now the default)
    python3 -m src.emotion_recognition.cli --with-pose "$@"
fi

# Exit with the same code as the python command
exit $?
