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

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed. Please install Python 3.12 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found $PYTHON_VERSION"

# Check for and create virtual environment if needed
if [ -f "./.venv/bin/activate" ]; then
    echo "Found existing virtual environment (.venv)"
elif [ -f "./venv/bin/activate" ]; then
    echo "Found existing virtual environment (venv)"
else
    echo "No virtual environment found. Creating one (.venv)..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please check your Python installation."
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

# Activate virtual environment
if [ -f "./.venv/bin/activate" ]; then
    echo "Activating virtual environment (.venv)..."
    source ./.venv/bin/activate
elif [ -f "./venv/bin/activate" ]; then
    echo "Activating virtual environment (venv)..."
    source ./venv/bin/activate
fi

# Install dependencies from requirements.txt
echo "Installing required dependencies..."
pip install -r requirements.txt

# Make the script executable
chmod +x scripts/macos/run_download_videos.sh
chmod +x scripts/macos/run_separate_speech.sh
chmod +x scripts/macos/run_speech_to_text.sh
chmod +x scripts/macos/run_emotion_recognition.sh

# Create a timestamp for the pipeline run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PIPELINE_DIR="pipeline_results_${TIMESTAMP}"

# Create pipeline directory
mkdir -p "$PIPELINE_DIR"
echo "Results will be stored in: $PIPELINE_DIR"

# Step 1: Download videos
echo -e "\n===== Step 1: Download Videos ====="
./scripts/macos/run_download_videos.sh --output-dir "${PIPELINE_DIR}/videos" "$@"
DOWNLOAD_EXIT=$?

if [ $DOWNLOAD_EXIT -ne 0 ]; then
    echo "Video download failed or was canceled (exit code $DOWNLOAD_EXIT)."
    echo "Pipeline will use any existing videos in data/videos directory."
    VIDEO_DIR="data/videos"
else
    echo "Video download completed successfully."
    VIDEO_DIR="${PIPELINE_DIR}/videos"
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
