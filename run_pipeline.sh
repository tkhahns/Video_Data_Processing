#!/bin/bash
# 
# Complete Video Processing Pipeline Script
#
# This script runs the entire video processing pipeline in the following order:
# 1. Download videos from SharePoint
# 2. Process videos in parallel:
#    - Speech separation followed by speech-to-text transcription
#    - Emotion and body pose recognition

echo "=================================="
echo "Video Data Processing Pipeline"
echo "=================================="
echo "This script will run all processing steps in sequence."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Set PROJECT_ROOT to script directory (since script is in project root)
PROJECT_ROOT="$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed. Please install Python 3.12 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found $PYTHON_VERSION"

# Check for and create virtual environment if needed - but ONLY in project root
VENV_PATH="${PROJECT_ROOT}/.venv"
ALT_VENV_PATH="${PROJECT_ROOT}/venv"

# Always use absolute paths to avoid creating venvs in wrong locations
if [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "Found existing virtual environment at ${VENV_PATH}"
elif [ -f "${ALT_VENV_PATH}/bin/activate" ]; then
    echo "Found existing virtual environment at ${ALT_VENV_PATH}"
else
    echo "No virtual environment found. Creating one at ${VENV_PATH}..."
    # Change to project root directory before creating the venv
    pushd "${PROJECT_ROOT}" > /dev/null
    python3 -m venv .venv
    popd > /dev/null
    
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please check your Python installation."
        exit 1
    fi
    echo "Virtual environment created successfully at ${VENV_PATH}."
fi

# Activate virtual environment using absolute paths
if [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "Activating virtual environment from ${VENV_PATH}..."
    source "${VENV_PATH}/bin/activate"
elif [ -f "${ALT_VENV_PATH}/bin/activate" ]; then
    echo "Activating virtual environment from ${ALT_VENV_PATH}..."
    source "${ALT_VENV_PATH}/bin/activate"
fi

# Install dependencies from requirements.txt using absolute path
echo "Installing required dependencies..."
pip install -r "${PROJECT_ROOT}/requirements.txt"

# Make the scripts executable using absolute paths
chmod +x "${PROJECT_ROOT}/scripts/macos/run_download_videos.sh"
chmod +x "${PROJECT_ROOT}/scripts/macos/run_separate_speech.sh"
chmod +x "${PROJECT_ROOT}/scripts/macos/run_speech_to_text.sh"
chmod +x "${PROJECT_ROOT}/scripts/macos/run_emotion_recognition.sh"

# Create and ensure essential directories exist
DATA_DIR="${PROJECT_ROOT}/data"
VIDEOS_DIR="${DATA_DIR}/videos"
OUTPUT_DIR="${PROJECT_ROOT}/output"
mkdir -p "$DATA_DIR"
mkdir -p "$VIDEOS_DIR"
mkdir -p "$OUTPUT_DIR"
echo "Ensuring data directories exist: $VIDEOS_DIR"

# Create a timestamp for the pipeline run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PIPELINE_DIR="${OUTPUT_DIR}/pipeline_results_${TIMESTAMP}"
TIMESTAMPED_VIDEOS_DIR="${DATA_DIR}/videos_${TIMESTAMP}"

# Create pipeline directory
mkdir -p "$PIPELINE_DIR"
echo "Results will be stored in: $PIPELINE_DIR"

# Ask user about video source
echo -e "\n===== Video Source Selection ====="
echo "Please select how you want to provide input videos:"
echo "1. Use existing videos from ${VIDEOS_DIR}"
echo "2. Download new videos from SharePoint"
read -p "Enter your choice (1 or 2): " video_choice

case $video_choice in
    1)
        echo "Using existing videos from ${VIDEOS_DIR}..."
        VIDEO_DIR="$VIDEOS_DIR"
        
        # Check if directory has video files
        VIDEO_COUNT=$(find "$VIDEO_DIR" -type f -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" | wc -l)
        if [ "$VIDEO_COUNT" -eq 0 ]; then
            echo "No video files found in $VIDEO_DIR!"
            echo "Would you like to download videos instead? (y/n)"
            read -p "> " download_choice
            if [[ "$download_choice" =~ ^[Yy] ]]; then
                echo "Proceeding to video download..."
                video_choice=2
                mkdir -p "$TIMESTAMPED_VIDEOS_DIR"
                VIDEO_DIR="$TIMESTAMPED_VIDEOS_DIR" 
            else
                echo "Please add video files to $VIDEO_DIR and run the script again."
                exit 1
            fi
        else
            echo "Found $VIDEO_COUNT video file(s) in $VIDEO_DIR"
        fi
        ;;
    2)
        echo "Downloading videos to timestamped directory..."
        mkdir -p "$TIMESTAMPED_VIDEOS_DIR"
        VIDEO_DIR="$TIMESTAMPED_VIDEOS_DIR"
        ;;
    *)
        echo "Invalid choice. Using existing videos by default."
        VIDEO_DIR="$VIDEOS_DIR"
        ;;
esac

# Step 1: Download videos (only if selected)
if [ "$video_choice" -eq 2 ]; then
    echo -e "\n===== Step 1: Download Videos ====="
    chmod +x "${PROJECT_ROOT}/scripts/macos/run_download_videos.sh"
    "${PROJECT_ROOT}/scripts/macos/run_download_videos.sh" --output-dir "$VIDEO_DIR" "$@"
    DOWNLOAD_EXIT=$?

    if [ $DOWNLOAD_EXIT -ne 0 ]; then
        echo "Video download failed or was canceled (exit code $DOWNLOAD_EXIT)."
        echo "Pipeline will use any existing videos in ${VIDEOS_DIR} directory."
        VIDEO_DIR="$VIDEOS_DIR"
    else
        echo "Video download completed successfully to $VIDEO_DIR."
    fi
else
    echo -e "\n===== Step 1: Download Videos (Skipped) ====="
    DOWNLOAD_EXIT=0
fi

# Step 2a: Speech Separation (in background)
echo -e "\n===== Step 2a: Speech Separation ====="
./scripts/macos/run_separate_speech.sh --output-dir "${PIPELINE_DIR}/speech" "$VIDEO_DIR" &
SPEECH_PID=$!

# Step 2b: Emotion Recognition (in background)
echo -e "\n===== Step 2b: Emotion Recognition ====="
./scripts/macos/run_emotion_recognition.sh --output-dir "${PIPELINE_DIR}/emotions" "$VIDEO_DIR" &
EMOTION_PID=$!

# Wait for speech separation to complete
wait $SPEECH_PID
SPEECH_EXIT=$?

echo -e "\n===== Speech separation completed with status: $SPEECH_EXIT ====="

# Step 3: Speech-to-text (only if speech separation succeeded)
if [ $SPEECH_EXIT -eq 0 ]; then
    echo -e "\n===== Step 3: Speech to Text ====="
    ./scripts/macos/run_speech_to_text.sh --input-dir "${PIPELINE_DIR}/speech" --output-dir "${PIPELINE_DIR}/transcripts"
    STT_EXIT=$?
    echo -e "\n===== Speech to text completed with status: $STT_EXIT ====="
else
    echo "Speech separation failed. Skipping speech to text step."
    STT_EXIT=1
fi

# Wait for emotion recognition to complete
wait $EMOTION_PID
EMOTION_EXIT=$?

echo -e "\n===== Emotion recognition completed with status: $EMOTION_EXIT ====="

# Summarize results
echo -e "\n=================================="
echo "Pipeline Execution Summary"
echo "=================================="
echo "1. Download Videos: $([ $DOWNLOAD_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "2. Speech Separation: $([ $SPEECH_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "3. Speech to Text: $([ $STT_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "4. Emotion Recognition: $([ $EMOTION_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo -e "\nResults saved to: $PIPELINE_DIR"

# Set permissions for output files
chmod -R 755 "$PIPELINE_DIR"

# Overall success
if [ $SPEECH_EXIT -eq 0 ] && [ $STT_EXIT -eq 0 ] && [ $EMOTION_EXIT -eq 0 ]; then
    echo -e "\nPipeline completed successfully!"
    exit 0
else
    echo -e "\nPipeline completed with some errors. Check the logs for details."
    exit 1
fi
``