#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing - Complete Pipeline ==="
echo "This script downloads videos to a timestamped directory and processes them."

# Get the script's directory (project root)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

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
    
    # STEP 2: Run speech separation if download was successful
    if [ $DOWNLOAD_EXIT -eq 0 ]; then
        echo -e "\n[4/6] Running speech separation on downloaded videos..."
        poetry run scripts/macos/run_separate_speech.sh --input-dir "$DOWNLOADS_DIR" --output-dir "$SPEECH_OUTPUT_DIR"
        SPEECH_EXIT=$?
        
        # STEP 3: Run speech-to-text if speech separation was successful
        if [ $SPEECH_EXIT -eq 0 ]; then
            echo -e "\n[5/6] Running speech-to-text on separated audio..."
            poetry run scripts/macos/run_speech_to_text.sh --input-dir "$SPEECH_OUTPUT_DIR" --output-dir "$TRANSCRIPT_OUTPUT_DIR"
            TRANSCRIPT_EXIT=$?
            
            # STEP 4: Run emotion and pose recognition on the original videos
            echo -e "\n[6/6] Running emotion and pose recognition on downloaded videos..."
            poetry run scripts/macos/run_emotion_and_pose_recognition.sh --input-dir "$DOWNLOADS_DIR" --output-dir "$EMOTIONS_AND_POSE_DIR"
            EMOTION_EXIT=$?
            
            # Report the final status of all pipeline steps
            echo -e "\n===== Pipeline Execution Summary ====="
            echo "- Video Download: $([ $DOWNLOAD_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
            echo "- Speech Separation: $([ $SPEECH_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
            echo "- Speech-to-Text: $([ $TRANSCRIPT_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
            echo "- Emotion and Pose Recognition: $([ $EMOTION_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
            
            echo -e "\nResults and outputs:"
            echo "- Downloaded videos: $DOWNLOADS_DIR"
            echo "- Separated speech: $SPEECH_OUTPUT_DIR"
            echo "- Transcripts: $TRANSCRIPT_OUTPUT_DIR"
            echo "- Emotion and pose analysis: $EMOTIONS_AND_POSE_DIR"
        else
            echo -e "\nSpeech separation failed with exit code $SPEECH_EXIT"
            echo "Downloaded videos are still available at: $DOWNLOADS_DIR"
        fi
    else
        echo -e "\nVideo download failed or was canceled (exit code $DOWNLOAD_EXIT)."
        echo "Pipeline halted."
    fi
else
    echo -e "\nPoetry not found. Cannot run the complete pipeline without Poetry."
    echo "Please install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi
