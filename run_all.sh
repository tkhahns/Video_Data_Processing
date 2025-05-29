#!/bin/bash

# Exit on error
set -e

# Record start time
START_TIME=$(date +%s)

echo "=== Video Data Processing - Complete Pipeline ==="
echo "This script processes videos from the data/ folder through the complete pipeline."

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

# Create timestamped directory for results
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
VIDEOS_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/output/pipeline_results_$TIMESTAMP"
SPEECH_OUTPUT_DIR="$RESULTS_DIR/speech"
TRANSCRIPT_OUTPUT_DIR="$RESULTS_DIR/transcripts"
EMOTIONS_AND_POSE_DIR="$RESULTS_DIR/emotions_and_pose"

echo -e "\nCreating timestamped directories:"
echo "- Source Videos: $VIDEOS_DIR"
echo "- Pipeline results: $RESULTS_DIR"
echo "  |- Speech: speech/"
echo "  |- Transcripts: transcripts/"
echo "  |- Emotions and Pose: emotions_and_pose/"
mkdir -p "$SPEECH_OUTPUT_DIR" "$TRANSCRIPT_OUTPUT_DIR" "$EMOTIONS_AND_POSE_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Poetry is installed
if command -v poetry &>/dev/null; then
    echo -e "\n[1/5] Installing dependencies using Poetry..."
    
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
    
    # Install GitHub models if requested
    if [ "$1" == "--with-models" ] || [ "$2" == "--with-models" ]; then
        echo "Installing GitHub models..."
        chmod +x "$PROJECT_ROOT/scripts/macos/install_models.sh"
        "$PROJECT_ROOT/scripts/macos/install_models.sh"
    fi
    
    echo "Dependencies installation completed."
    
    # Make scripts executable
    echo -e "\n[2/5] Preparing scripts..."
    chmod +x "$PROJECT_ROOT/scripts/macos/run_separate_speech.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/run_speech_to_text.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/run_emotion_and_pose_recognition.sh"
    
    # Check if any videos are available in the data directory
    VIDEO_COUNT=$(find "$VIDEOS_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.MP4" -o -name "*.MOV" -o -name "*.AVI" -o -name "*.MKV" \) | wc -l | tr -d '[:space:]')
    
    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo -e "\nNo videos were found in the directory: $VIDEOS_DIR"
        echo "Available files:"
        ls -la "$VIDEOS_DIR"
        echo "Pipeline halted - no videos to process."
        exit 1
    fi
    
    # Find all video files and store their paths
    VIDEO_FILES=($(find "$VIDEOS_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.MP4" -o -name "*.MOV" -o -name "*.AVI" -o -name "*.MKV" \)))
    
    echo -e "\n${VIDEO_COUNT} videos found in the data directory:"
    for ((i=0; i<${#VIDEO_FILES[@]}; i++)); do
        echo "[$((i+1))] $(basename "${VIDEO_FILES[$i]}")"
    done
    
    # Ask user if they want to process all videos or select specific ones
    echo -e "\nHow would you like to process the videos?"
    echo "1. Process all videos automatically"
    echo "2. Select specific videos to process at each step"
    
    PROCESS_ALL=false
    SELECTED_VIDEOS=()
    read -p "Enter your choice (1 or 2): " choice
    if [ "$choice" == "1" ]; then
        PROCESS_ALL=true
        SELECTED_VIDEOS=("${VIDEO_FILES[@]}")
        echo "Processing all videos automatically."
    else
        echo "Select videos to process (comma-separated numbers, e.g., 1,3,5):"
        read -p "Your selection: " selection
        
        # Parse the selection
        IFS=',' read -ra INDICES <<< "$selection"
        for index in "${INDICES[@]}"; do
            # Convert to 0-based index and check bounds
            idx=$((index-1))
            if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#VIDEO_FILES[@]}" ]; then
                SELECTED_VIDEOS+=("${VIDEO_FILES[$idx]}")
                echo "Selected: $(basename "${VIDEO_FILES[$idx]}")"
            else
                echo "Warning: Invalid selection $index, skipping"
            fi
        done
        
        # Check if any videos were selected
        if [ ${#SELECTED_VIDEOS[@]} -eq 0 ]; then
            echo "No valid videos selected. Pipeline halted."
            exit 1
        fi
        
        echo "You will process ${#SELECTED_VIDEOS[@]} selected videos."
    fi
    
    # Create a temporary file with the selected video paths for modules to use
    SELECTED_FILES_LIST=$(mktemp)
    for file in "${SELECTED_VIDEOS[@]}"; do
        echo "$file" >> "$SELECTED_FILES_LIST"
    done
    
    # Set batch flag based on user choice
    BATCH_FLAG=""
    if [ "$PROCESS_ALL" = true ]; then
        BATCH_FLAG="--batch"
    fi
    
    # Create a semaphore file to track completion of parallel processes
    SEMAPHORE_FILE=$(mktemp)
    
    # Start emotion and pose recognition in parallel (background)
    echo -e "\n[3/5] Running emotion and pose recognition in parallel..."
    (
        # Pass individual files directly to emotion recognition script
        if [ ${#SELECTED_VIDEOS[@]} -gt 0 ]; then
            if [ "$PROCESS_ALL" = true ]; then
                # In batch mode, just pass the directory
                poetry run scripts/macos/run_emotion_and_pose_recognition.sh --input-dir "$VIDEOS_DIR" --output-dir "$EMOTIONS_AND_POSE_DIR" $BATCH_FLAG
            else
                # For selected videos, pass them directly as arguments
                poetry run python -m src.emotion_and_pose_recognition.cli interactive --input_dir "$VIDEOS_DIR" --output_dir "$EMOTIONS_AND_POSE_DIR" --with-pose "${SELECTED_VIDEOS[@]}"
            fi
        else
            poetry run scripts/macos/run_emotion_and_pose_recognition.sh --input-dir "$VIDEOS_DIR" --output-dir "$EMOTIONS_AND_POSE_DIR"
        fi
        
        EMOTION_EXIT=$?
        echo "EMOTION_EXIT=$EMOTION_EXIT" >> "$SEMAPHORE_FILE"
        echo -e "\nEmotion and pose recognition completed with exit code $EMOTION_EXIT"
    ) &
    EMOTION_PID=$!
    
    # Run speech processing pipeline sequentially
    echo -e "\n[4/5] Running speech separation..."
    if [ ${#SELECTED_VIDEOS[@]} -gt 0 ]; then
        if [ "$PROCESS_ALL" = true ]; then
            # In batch mode, just pass the directory
            poetry run scripts/macos/run_separate_speech.sh --input-dir "$VIDEOS_DIR" --output-dir "$SPEECH_OUTPUT_DIR" $BATCH_FLAG
        else
            # For selected videos, pass specific files
            poetry run python -m src.separate_speech --output-dir "$SPEECH_OUTPUT_DIR" "${SELECTED_VIDEOS[@]}"
        fi
    else
        poetry run scripts/macos/run_separate_speech.sh --input-dir "$VIDEOS_DIR" --output-dir "$SPEECH_OUTPUT_DIR"
    fi
    SPEECH_EXIT=$?
    
    # Clean up temporary files
    rm -f "$SELECTED_FILES_LIST"
    
    # Report the final status of all pipeline steps
    echo -e "\n===== Pipeline Execution Summary ====="
    echo "- Speech Separation: $([ $SPEECH_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    echo "- Speech-to-Text: $([ $TRANSCRIPT_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    echo "- Emotion and Pose Recognition: $([ $EMOTION_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    
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
    echo "- Source videos: $VIDEOS_DIR"
    echo "- Separated speech: $SPEECH_OUTPUT_DIR"
    echo "- Transcripts: $TRANSCRIPT_OUTPUT_DIR"
    echo "- Emotion and pose analysis: $EMOTIONS_AND_POSE_DIR"
else
    echo -e "\nPoetry not found. Cannot run the complete pipeline without Poetry."
    echo "Please install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi
