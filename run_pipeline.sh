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

# Check if Poetry is installed, if not install it
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
fi

# Configure Poetry to create the virtualenv in the project directory
echo "Configuring Poetry..."
poetry config virtualenvs.in-project true

# Set environment variables to suppress model download output
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore

# Install base dependencies
echo "Installing common dependencies..."
poetry install --no-root

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
    
    # Install download dependencies
    echo "Installing download dependencies..."
    cd "$PROJECT_ROOT"
    poetry install --only download
    
    chmod +x "${PROJECT_ROOT}/scripts/macos/run_download_videos.sh"
    poetry run "${PROJECT_ROOT}/scripts/macos/run_download_videos.sh" --output-dir "$VIDEO_DIR" "$@"
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
(
    # Create subprocess with speech dependencies in an isolated environment
    echo "Installing speech dependencies..."
    cd "$PROJECT_ROOT"
    
    # Set environment variables in subprocess
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export TRANSFORMERS_VERBOSITY=error
    export TOKENIZERS_PARALLELISM=false
    export PYTHONWARNINGS=ignore
    
    poetry install --with common --with speech --no-root
    
    # Run speech separation with quiet flag
    poetry run python -m src.separate_speech --quiet --output-dir "${PIPELINE_DIR}/speech" "$VIDEO_DIR"
    SPEECH_EXIT=$?
    
    # If speech separation succeeds, run speech-to-text
    if [ $SPEECH_EXIT -eq 0 ]; then
        echo -e "\n===== Step 3: Speech to Text ====="
        poetry run python -m src.speech_to_text --quiet --input-dir "${PIPELINE_DIR}/speech" --output-dir "${PIPELINE_DIR}/transcripts"
        STT_EXIT=$?
        echo -e "\n===== Speech to text completed with status: $STT_EXIT ====="
    else
        echo "Speech separation failed. Skipping speech to text step."
        STT_EXIT=1
    fi
    
    # Save exit codes to files for the parent process to read
    echo $SPEECH_EXIT > "${PIPELINE_DIR}/.speech_exit"
    echo $STT_EXIT > "${PIPELINE_DIR}/.stt_exit"
) &
SPEECH_PID=$!

# Step 2b: Emotion Recognition (in background)
echo -e "\n===== Step 2b: Emotion Recognition ====="
(
    # Create subprocess with emotion dependencies in a separate isolated environment
    echo "Installing emotion recognition dependencies..."
    cd "$PROJECT_ROOT"
    
    # Set environment variables in subprocess
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export TRANSFORMERS_VERBOSITY=error
    export TOKENIZERS_PARALLELISM=false
    export PYTHONWARNINGS=ignore
    
    poetry install --with common --with emotion --no-root
    
    # Run emotion recognition with quiet flag
    poetry run python -m src.emotion_recognition.cli --quiet --output-dir "${PIPELINE_DIR}/emotions" "$VIDEO_DIR"
    EMOTION_EXIT=$?
    
    # Save exit code to file for the parent process to read
    echo $EMOTION_EXIT > "${PIPELINE_DIR}/.emotion_exit"
) &
EMOTION_PID=$!

# Wait for processes to complete
wait $SPEECH_PID
wait $EMOTION_PID

# Read exit codes
SPEECH_EXIT=$(cat "${PIPELINE_DIR}/.speech_exit")
STT_EXIT=$(cat "${PIPELINE_DIR}/.stt_exit")
EMOTION_EXIT=$(cat "${PIPELINE_DIR}/.emotion_exit")

# Clean up temporary files
rm -f "${PIPELINE_DIR}/.speech_exit" "${PIPELINE_DIR}/.stt_exit" "${PIPELINE_DIR}/.emotion_exit"

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