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
if ! poetry install --no-root; then
    echo "First poetry install attempt failed, retrying once..."
    poetry install --no-root
fi

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

# Ask user about processing steps to run
echo -e "\n===== Processing Steps Selection ====="
echo "Please select which processing steps you want to run:"
echo "1. Run all processing steps (recommended)"
echo "2. Select specific processing steps"
read -p "Enter your choice (1 or 2): " steps_choice

# Helper function to get validated yes/no input
get_yes_no_input() {
    local prompt=$1
    local response
    
    while true; do
        read -p "$prompt" response
        case "$response" in
            [Yy]* ) echo "true"; return ;;
            [Nn]* ) echo "false"; return ;;
            * ) echo "Invalid input. Please enter 'y' or 'n'." >&2 ;;
        esac
    done
}

# Initialize step flags (default to true/run everything)
RUN_SPEECH_SEPARATION=true
RUN_SPEECH_TO_TEXT=true
RUN_EMOTION_RECOGNITION=true

if [ "$steps_choice" -eq 2 ]; then
    # Let user select specific steps
    echo -e "\nSelect processing steps (y/n for each):"
    
    # Speech separation
    speech_response=$(get_yes_no_input "Run speech separation? (y/n): ")
    RUN_SPEECH_SEPARATION=$speech_response
    
    if [ "$RUN_SPEECH_SEPARATION" == "false" ]; then
        # Speech-to-Text requires Speech Separation
        RUN_SPEECH_TO_TEXT=false
        echo "Speech-to-Text will be skipped (requires Speech Separation)"
    else
        # Speech-to-text
        stt_response=$(get_yes_no_input "Run speech-to-text on separated audio? (y/n): ")
        RUN_SPEECH_TO_TEXT=$stt_response
    fi
    
    # Emotion recognition
    emotion_response=$(get_yes_no_input "Run emotion recognition? (y/n): ")
    RUN_EMOTION_RECOGNITION=$emotion_response
    
    # Verify at least one step is selected
    if [[ "$RUN_SPEECH_SEPARATION" == "false" && "$RUN_EMOTION_RECOGNITION" == "false" ]]; then
        echo "Error: You must select at least one processing step."
        echo "Defaulting to running all steps."
        RUN_SPEECH_SEPARATION=true
        RUN_SPEECH_TO_TEXT=true
        RUN_EMOTION_RECOGNITION=true
    else
        echo -e "\nSelected processing steps:"
        [[ "$RUN_SPEECH_SEPARATION" == "true" ]] && echo "- Speech Separation"
        [[ "$RUN_SPEECH_TO_TEXT" == "true" ]] && echo "- Speech-to-Text"
        [[ "$RUN_EMOTION_RECOGNITION" == "true" ]] && echo "- Emotion Recognition"
    fi
else
    echo "Running all processing steps."
fi

# Step 1: Download videos (only if selected)
if [ "$video_choice" -eq 2 ]; then
    echo -e "\n===== Step 1: Download Videos ====="
    
    # Install download dependencies
    echo "Installing download dependencies..."
    cd "$PROJECT_ROOT"
    if ! poetry install --only download; then
        echo "First poetry install attempt failed, retrying once..."
        poetry install --only download
    fi
    
    chmod +x "${PROJECT_ROOT}/scripts/macos/run_download_videos.sh"
    
    # Execute the download script with proper Poetry environment
    # Instead of passing all original arguments, just pass what's needed
    echo "Launching download interface..."
    DOWNLOAD_URL=""
    read -p "Enter SharePoint URL containing videos (leave empty for interactive mode): " DOWNLOAD_URL
    
    if [ -n "$DOWNLOAD_URL" ]; then
        # Call with URL if provided
        poetry run "${PROJECT_ROOT}/scripts/macos/run_download_videos.sh" --url "$DOWNLOAD_URL" --output-dir "$VIDEO_DIR"
    else
        # Call in interactive mode
        poetry run "${PROJECT_ROOT}/scripts/macos/run_download_videos.sh" --output-dir "$VIDEO_DIR"
    fi
    
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

# Set default exit status for unselected steps
SPEECH_EXIT=0
STT_EXIT=0
EMOTION_EXIT=0

# Function to run a command with Poetry, with retries
run_with_poetry() {
    local module=$1
    shift
    if ! poetry run python -m "$module" "$@"; then
        echo "Command failed, retrying once..."
        poetry run python -m "$module" "$@"
    fi
}

# Step 2a: Speech Separation (in background)
if [ "$RUN_SPEECH_SEPARATION" == true ]; then
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
        
        if ! poetry install --with common --with speech --no-root; then
            echo "First poetry install attempt failed, retrying once..."
            poetry install --with common --with speech --no-root
        fi
        
        # Run speech separation
        echo "Running speech separation..."
        run_with_poetry src.separate_speech --output-dir "${PIPELINE_DIR}/speech" "$VIDEO_DIR"
        SPEECH_EXIT=$?
        
        # If speech separation succeeds, run speech-to-text
        if [ $SPEECH_EXIT -eq 0 ]; then
            echo -e "\n===== Step 3: Speech to Text ====="
            echo "Running speech-to-text..."
            # Fix: Change --input-dir to a positional argument
            run_with_poetry src.speech_to_text "${PIPELINE_DIR}/speech" --output-dir "${PIPELINE_DIR}/transcripts"
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
else
    echo -e "\n===== Step 2a: Speech Separation (Skipped) ====="
    echo 0 > "${PIPELINE_DIR}/.speech_exit"
    echo 0 > "${PIPELINE_DIR}/.stt_exit"
fi

# Step 2b: Emotion Recognition (in background)
if [ "$RUN_EMOTION_RECOGNITION" == true ]; then
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
        
        if ! poetry install --with common --with emotion --no-root; then
            echo "First poetry install attempt failed, retrying once..."
            poetry install --with common --with emotion --no-root
        fi
        
        # Run emotion recognition
        echo "Running emotion recognition..."
        # Fixed command structure: command first, then input, then options
        run_with_poetry src.emotion_recognition.cli batch "$VIDEO_DIR" --output-dir "${PIPELINE_DIR}/emotions"
        EMOTION_EXIT=$?
        
        # Save exit code to file for the parent process to read
        echo $EMOTION_EXIT > "${PIPELINE_DIR}/.emotion_exit"
    ) &
    EMOTION_PID=$!
else
    echo -e "\n===== Step 2b: Emotion Recognition (Skipped) ====="
    echo 0 > "${PIPELINE_DIR}/.emotion_exit"
fi

# Wait for processes to complete (only if they were started)
if [ "$RUN_SPEECH_SEPARATION" == true ]; then
    wait $SPEECH_PID
fi

if [ "$RUN_EMOTION_RECOGNITION" == true ]; then
    wait $EMOTION_PID
fi

# Read exit codes
if [ -f "${PIPELINE_DIR}/.speech_exit" ]; then
    SPEECH_EXIT=$(cat "${PIPELINE_DIR}/.speech_exit")
fi

if [ -f "${PIPELINE_DIR}/.stt_exit" ]; then
    STT_EXIT=$(cat "${PIPELINE_DIR}/.stt_exit")
fi

if [ -f "${PIPELINE_DIR}/.emotion_exit" ]; then
    EMOTION_EXIT=$(cat "${PIPELINE_DIR}/.emotion_exit")
fi

# Clean up temporary files
rm -f "${PIPELINE_DIR}/.speech_exit" "${PIPELINE_DIR}/.stt_exit" "${PIPELINE_DIR}/.emotion_exit"

# Summarize results
echo -e "\n=================================="
echo "Pipeline Execution Summary"
echo "=================================="
echo "1. Download Videos: $([ $DOWNLOAD_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"

if [ "$RUN_SPEECH_SEPARATION" == true ]; then
    echo "2. Speech Separation: $([ $SPEECH_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
    
    if [ "$RUN_SPEECH_TO_TEXT" == true ]; then
        echo "3. Speech to Text: $([ $STT_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
    else
        echo "3. Speech to Text: SKIPPED"
    fi
else
    echo "2. Speech Separation: SKIPPED"
    echo "3. Speech to Text: SKIPPED"
fi

if [ "$RUN_EMOTION_RECOGNITION" == true ]; then
    echo "4. Emotion Recognition: $([ $EMOTION_EXIT -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
else
    echo "4. Emotion Recognition: SKIPPED"
fi

echo -e "\nResults saved to: $PIPELINE_DIR"

# Set permissions for output files
chmod -R 755 "$PIPELINE_DIR"

# Overall success - only check steps that were run
OVERALL_SUCCESS=true
if [ "$RUN_SPEECH_SEPARATION" == true ] && [ $SPEECH_EXIT -ne 0 ]; then
    OVERALL_SUCCESS=false
fi
if [ "$RUN_SPEECH_TO_TEXT" == true ] && [ $STT_EXIT -ne 0 ]; then
    OVERALL_SUCCESS=false
fi
if [ "$RUN_EMOTION_RECOGNITION" == true ] && [ $EMOTION_EXIT -ne 0 ]; then
    OVERALL_SUCCESS=false
fi

# Final status message
if [ "$OVERALL_SUCCESS" == true ]; then
    echo -e "\nPipeline completed successfully!"
    exit 0
else
    echo -e "\nPipeline completed with some errors. Check the logs for details."
    exit 1
fi