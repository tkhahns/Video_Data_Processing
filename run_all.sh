#!/bin/bash

# Exit on error
set -e

# Record start time
START_TIME=$(date +%s)

echo "=== Video Data Processing - Complete Pipeline ==="
echo "This script downloads videos to a timestamped directory and processes them."

# Get the script's directory (project root)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Setup function to delete token on exit
cleanup_token() {
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        echo "Clearing Hugging Face token from environment"
        unset HUGGINGFACE_TOKEN
    fi
    
    # Remove token file if it exists
    TOKEN_FILE="$HOME/.huggingface_token"
    if [ -f "$TOKEN_FILE" ]; then
        echo "Removing saved Hugging Face token"
        rm -f "$TOKEN_FILE"
    fi
}

# Register the cleanup function to run on script exit
trap cleanup_token EXIT

# Get Hugging Face token - temporary for this session only
echo -e "\n=== Hugging Face Authentication ==="
echo "This tool requires a Hugging Face token for accessing models."
echo "You can get your token from: https://huggingface.co/settings/tokens"
echo "Note: Your token will only be used for this session and will not be saved."

# Prompt for token
read -sp "Enter your Hugging Face token (input will be hidden): " HUGGINGFACE_TOKEN
echo ""

# Validate token is provided
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "No token provided. Some features may not work correctly."
else
    echo "Token received for this session"
fi

export HUGGINGFACE_TOKEN

# Create timestamped directory
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
DOWNLOADS_DIR="$PROJECT_ROOT/data/downloads_$TIMESTAMP"
RESULTS_DIR="$PROJECT_ROOT/output/pipeline_results_$TIMESTAMP"
SPEECH_OUTPUT_DIR="$RESULTS_DIR/speech"
TRANSCRIPT_OUTPUT_DIR="$RESULTS_DIR/transcripts"
EMOTIONS_AND_POSE_DIR="$RESULTS_DIR/emotions_and_pose"

echo -e "\nCreating timestamped directories:"
echo "- Downloaded Videos: $DOWNLOADS_DIR"
echo "- Pipeline results: $RESULTS_DIR"
echo "  |- Speech: speech/"
echo "  |- Transcripts: transcripts/"
echo "  |- Emotions and Pose: emotions_and_pose/"
mkdir -p "$DOWNLOADS_DIR" "$SPEECH_OUTPUT_DIR" "$TRANSCRIPT_OUTPUT_DIR" "$EMOTIONS_AND_POSE_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Poetry is installed
if command -v poetry &>/dev/null; then
    echo -e "\n[1/6] Installing dependencies using Poetry..."
    
    # Install all dependencies from pyproject.toml
    echo "Installing base dependencies..."
    poetry install --no-interaction || { echo "Failed to install base dependencies"; exit 1; }
    
    # Install specific groups
    echo "Installing common dependencies..."
    poetry install --with common --no-interaction || echo "Warning: Some common dependencies failed to install"
    
    echo "Installing emotion recognition dependencies..."
    poetry install --with emotion --no-interaction || echo "Warning: Some emotion recognition dependencies failed to install"
    
    echo "Installing speech recognition dependencies..."
    poetry install --with speech --no-interaction || echo "Warning: Some speech recognition dependencies failed to install"
    
    echo "Installing download dependencies..."
    poetry install --with download --no-interaction || echo "Warning: Some download dependencies failed to install"
    
    echo "Dependencies installation completed."
    
    # Make scripts executable
    echo -e "\n[2/6] Preparing scripts..."
    chmod +x "$PROJECT_ROOT/scripts/macos/run_download_videos.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/run_separate_speech.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/run_speech_to_text.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/run_emotion_and_pose_recognition.sh"
    
    # STEP 1: Run video downloader
    echo -e "\n[3/6] Running video downloader with Poetry..."
    # Pass all original arguments plus the output directory
    poetry run scripts/macos/run_download_videos.sh "$@" --output-dir "$DOWNLOADS_DIR"
    DOWNLOAD_EXIT=$?
    
    # STEP 2: Proceed if download was successful
    if [ $DOWNLOAD_EXIT -eq 0 ]; then
        # Check if any videos were downloaded
        VIDEO_COUNT=$(find "$DOWNLOADS_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.MP4" -o -name "*.MOV" -o -name "*.AVI" -o -name "*.MKV" \) | wc -l | tr -d '[:space:]')
        
        if [ "$VIDEO_COUNT" -eq 0 ]; then
            echo -e "\nNo videos were found in the directory: $DOWNLOADS_DIR"
            echo "Available files:"
            ls -la "$DOWNLOADS_DIR"
            echo "Pipeline halted - no videos to process."
            exit 1
        fi
        
        echo -e "\n${VIDEO_COUNT} videos found in the downloads directory:"
        find "$DOWNLOADS_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.MP4" -o -name "*.MOV" -o -name "*.AVI" -o -name "*.MKV" \) -exec basename {} \;
        
        # Ask user if they want to process all videos or select manually
        echo -e "\nHow would you like to process the downloaded videos?"
        echo "1. Process all downloaded videos automatically"
        echo "2. Select specific videos to process at each step"
        
        PROCESS_ALL=false
        read -p "Enter your choice (1 or 2): " choice
        if [ "$choice" == "1" ]; then
            PROCESS_ALL=true
            echo "Processing all videos automatically."
        else
            echo "You will be prompted to select videos at each step."
        fi
        
        # Set batch flag based on user choice
        BATCH_FLAG=""
        if [ "$PROCESS_ALL" = true ]; then
            BATCH_FLAG="--batch"
        fi
        
        # Create a semaphore file to track completion of parallel processes
        SEMAPHORE_FILE=$(mktemp)
        
        # Start emotion and pose recognition in parallel (background)
        echo -e "\n[4/6] Running emotion and pose recognition in parallel..."
        (
            poetry run scripts/macos/run_emotion_and_pose_recognition.sh --input-dir "$DOWNLOADS_DIR" --output-dir "$EMOTIONS_AND_POSE_DIR" $BATCH_FLAG
            EMOTION_EXIT=$?
            echo "EMOTION_EXIT=$EMOTION_EXIT" >> "$SEMAPHORE_FILE"
            echo -e "\nEmotion and pose recognition completed with exit code $EMOTION_EXIT"
        ) &
        EMOTION_PID=$!
        
        # Run speech processing pipeline sequentially
        echo -e "\n[5/6] Running speech separation..."
        poetry run scripts/macos/run_separate_speech.sh --input-dir "$DOWNLOADS_DIR" --output-dir "$SPEECH_OUTPUT_DIR" $BATCH_FLAG
        SPEECH_EXIT=$?
        
        # Only proceed with transcription if speech separation was successful
        if [ $SPEECH_EXIT -eq 0 ]; then
            echo -e "\n[6/6] Running speech-to-text on separated audio..."
            poetry run scripts/macos/run_speech_to_text.sh --input-dir "$SPEECH_OUTPUT_DIR" --output-dir "$TRANSCRIPT_OUTPUT_DIR" --diarize --extract-features $BATCH_FLAG
            TRANSCRIPT_EXIT=$?
        else
            echo -e "\nSpeech separation failed with exit code $SPEECH_EXIT"
            TRANSCRIPT_EXIT=1  # Set failure code for transcript since we couldn't process it
        fi
        
        # Wait for emotion recognition to complete
        echo "Waiting for emotion and pose recognition to complete..."
        wait $EMOTION_PID
        
        # Read the exit status of the emotion recognition process
        if [ -f "$SEMAPHORE_FILE" ]; then
            source "$SEMAPHORE_FILE"
            rm -f "$SEMAPHORE_FILE"
        else
            # If semaphore file doesn't exist, assume failure
            EMOTION_EXIT=1
        fi
        
        # Report the final status of all pipeline steps
        echo -e "\n===== Pipeline Execution Summary ====="
        echo "- Video Download: $([ $DOWNLOAD_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
        echo "- Speech Separation: $([ $SPEECH_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
        echo "- Speech-to-Text: $([ $TRANSCRIPT_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
        echo "- Emotion and Pose Recognition: $([ $EMOTION_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
        
        # Create pipeline_output.csv that merges all results
        echo -e "\n[+] Creating pipeline output CSV..."
        poetry run python -c "from src.speech_to_text.speech_features import create_pipeline_output; create_pipeline_output('$RESULTS_DIR')"
        CSV_EXIT=$?
        
        if [ $CSV_EXIT -eq 0 ]; then
            echo "✅ Pipeline output CSV created successfully"
            echo "- CSV output: $RESULTS_DIR/pipeline_output.csv"
            echo "- Summary: $RESULTS_DIR/pipeline_summary.txt"
        else
            echo "❌ Failed to create pipeline output CSV"
        fi

        # Calculate total process time
        END_TIME=$(date +%s)
        ELAPSED_TIME=$((END_TIME - START_TIME))
        HOURS=$((ELAPSED_TIME / 3600))
        MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
        SECONDS=$((ELAPSED_TIME % 60))
        
        # Format time with leading zeros for better readability
        TIME_FORMAT=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECONDS)
        
        echo -e "\nTotal process time: $TIME_FORMAT (HH:MM:SS)"
        
        echo -e "\nResults and outputs:"
        echo "- Downloaded videos: $DOWNLOADS_DIR"
        echo "- Separated speech: $SPEECH_OUTPUT_DIR"
        echo "- Transcripts: $TRANSCRIPT_OUTPUT_DIR"
        echo "- Emotion and pose analysis: $EMOTIONS_AND_POSE_DIR"
    else
        echo -e "\nVideo download failed or was canceled (exit code $DOWNLOAD_EXIT)."
        echo "Pipeline halted."
        
        # Show execution time even if the pipeline was halted
        END_TIME=$(date +%s)
        ELAPSED_TIME=$((END_TIME - START_TIME))
        MINUTES=$(( ELAPSED_TIME / 60 ))
        SECONDS=$(( ELAPSED_TIME % 60 ))
        echo "Execution time: ${MINUTES}m ${SECONDS}s"
    fi
else
    echo -e "\nPoetry not found. Cannot run the complete pipeline without Poetry."
    echo "Please install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi
