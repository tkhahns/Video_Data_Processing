#!/bin/bash

# Exit on error
set -e

# Record start time
START_TIME=$(date +%s)

echo "=== Video Data Processing - Complete Pipeline ==="
echo "This script processes existing videos in the data directory."

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

# Create timestamped directory for results only
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RESULTS_DIR="$PROJECT_ROOT/output/pipeline_results_$TIMESTAMP"
SPEECH_OUTPUT_DIR="$RESULTS_DIR/speech"
TRANSCRIPT_OUTPUT_DIR="$RESULTS_DIR/transcripts"
EMOTIONS_AND_POSE_DIR="$RESULTS_DIR/emotions_and_pose"

# Define the input videos directory (use existing data directory)
VIDEOS_DIR="$PROJECT_ROOT/data"

echo -e "\nLocations:"
echo "- Input Videos: $VIDEOS_DIR"
echo "- Pipeline results: $RESULTS_DIR"
echo "  |- Speech: speech/"
echo "  |- Transcripts: transcripts/"
echo "  |- Emotions and Pose: emotions_and_pose/"
mkdir -p "$SPEECH_OUTPUT_DIR" "$TRANSCRIPT_OUTPUT_DIR" "$EMOTIONS_AND_POSE_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Poetry is installed
if command -v poetry &>/dev/null; then
    # Get all video files with absolute paths first, before any user input
    echo -e "\nGathering video files..."
    VIDEO_FILES=()
    while IFS= read -r file; do
        # Get absolute path for each video file
        abs_path=$(realpath "$file")
        VIDEO_FILES+=("$abs_path")
        echo " - Found: $abs_path"
    done < <(find "$VIDEOS_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.MP4" -o -name "*.MOV" -o -name "*.AVI" -o -name "*.MKV" \))
    
    VIDEO_COUNT=${#VIDEO_FILES[@]}
    
    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo -e "\nNo videos found in $VIDEOS_DIR"
        echo "Please place video files in the data directory before running this script."
        exit 1
    fi
    
    echo -e "\n${VIDEO_COUNT} videos found in the data directory:"
    for video_path in "${VIDEO_FILES[@]}"; do
        echo " - $(basename "$video_path")"
    done

    # STEP 1: Get Hugging Face token - moved to top for better UX
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

    # STEP 2: Ask user if they want to process all videos or select manually
    echo -e "\nHow would you like to process the videos?"
    echo "1. Process all videos automatically"
    echo "2. Select specific videos to process at each step"
    
    PROCESS_ALL=false
    read -p "Enter your choice (1 or 2): " choice
    if [ "$choice" == "1" ]; then
        PROCESS_ALL=true
        echo "Processing all videos automatically."
        # Use all videos found
        SELECTED_VIDEOS=("${VIDEO_FILES[@]}")
        EMOTION_VIDEOS=("${VIDEO_FILES[@]}")
    else
        echo "You will be prompted to select videos for each step."
        
        # STEP 2a: Select videos for speech separation now
        echo -e "\n=== Select videos for speech separation ==="
        # Display list with numbers
        for i in "${!VIDEO_FILES[@]}"; do
            echo "[$((i+1))] $(basename "${VIDEO_FILES[$i]}")"
        done
        
        echo -e "\nEnter the numbers of videos to process (e.g., '1,3,5'), or 'all' for all videos:"
        read selection
        
        # Parse selection
        if [[ "$selection" == "all" ]]; then
            # Use all videos
            SELECTED_VIDEOS=("${VIDEO_FILES[@]}")
            echo "Selected all videos for speech separation."
        else
            # Parse numbers and select videos
            IFS=',' read -ra NUMS <<< "$selection"
            SELECTED_VIDEOS=()
            
            for num in "${NUMS[@]}"; do
                # Convert to array index (subtract 1)
                idx=$((num-1))
                
                # Validate index
                if [[ $idx -ge 0 && $idx -lt ${#VIDEO_FILES[@]} ]]; then
                    SELECTED_VIDEOS+=("${VIDEO_FILES[$idx]}")
                    echo "Selected: $(basename "${VIDEO_FILES[$idx]}")"
                else
                    echo "Warning: Invalid selection $num, skipping"
                fi
            done
        fi
        
        # STEP 2b: Select videos for emotion recognition now
        echo -e "\n=== Select videos for emotion and pose recognition ==="
        # Display list with numbers
        for i in "${!VIDEO_FILES[@]}"; do
            echo "[$((i+1))] $(basename "${VIDEO_FILES[$i]}")"
        done
        
        echo -e "\nEnter the numbers of videos to process (e.g., '1,3,5'), or 'all' for all videos:"
        read selection
        
        # Parse selection
        if [[ "$selection" == "all" ]]; then
            # Use all videos
            EMOTION_VIDEOS=("${VIDEO_FILES[@]}")
            echo "Selected all videos for emotion and pose recognition."
        else
            # Parse numbers and select videos
            IFS=',' read -ra NUMS <<< "$selection"
            EMOTION_VIDEOS=()
            
            for num in "${NUMS[@]}"; do
                # Convert to array index (subtract 1)
                idx=$((num-1))
                
                # Validate index
                if [[ $idx -ge 0 && $idx -lt ${#VIDEO_FILES[@]} ]]; then
                    EMOTION_VIDEOS+=("${VIDEO_FILES[$idx]}")
                    echo "Selected: $(basename "${VIDEO_FILES[$idx]}")"
                else
                    echo "Warning: Invalid selection $num, skipping"
                fi
            done
        fi
    fi
    
    # Set batch flag based on user choice
    BATCH_FLAG=""
    if [ "$PROCESS_ALL" = true ]; then
        BATCH_FLAG="--batch"
    fi
    
    # NOW START ACTUAL PROCESSING - all user input has been collected
    
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
    
    echo "Dependencies installation completed."
    
    # Make scripts executable
    echo -e "\n[2/6] Preparing scripts..."
    chmod +x "$PROJECT_ROOT/scripts/macos/run_separate_speech.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/run_speech_to_text.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/run_emotion_and_pose_recognition.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/extract_audio_features.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/extract_video_features.sh"
    chmod +x "$PROJECT_ROOT/scripts/macos/extract_multimodal_features.sh"
    
    # Create semaphore files to track completion of parallel processes
    SEMAPHORE_DIR=$(mktemp -d)
    TRANSCRIPT_SEMAPHORE="$SEMAPHORE_DIR/transcript"
    AUDIO_FEATURES_SEMAPHORE="$SEMAPHORE_DIR/audio_features"
    VIDEO_FEATURES_SEMAPHORE="$SEMAPHORE_DIR/video_features"
    MULTIMODAL_SEMAPHORE="$SEMAPHORE_DIR/multimodal"
    EMOTION_SEMAPHORE="$SEMAPHORE_DIR/emotion"
    
    # Step 1: FIRST run speech separation to ensure audio files exist
    echo -e "\n[3/6] Running speech separation..."
    
    # Now run speech separation with only the selected videos
    if [ ${#SELECTED_VIDEOS[@]} -eq 0 ]; then
        echo "No videos selected for speech separation. Skipping this step."
        SPEECH_EXIT=1
    else
        # Pass individual video files instead of input directory
        video_args=""
        for video_path in "${SELECTED_VIDEOS[@]}"; do
            video_args+=" --video \"$video_path\""
        done
        
        cmd="poetry run scripts/macos/run_separate_speech.sh $video_args --output-dir \"$SPEECH_OUTPUT_DIR\" $BATCH_FLAG"
        echo "Executing: $cmd"
        eval $cmd
        SPEECH_EXIT=$?
    fi
    
    # Only proceed with parallel processing if speech separation was successful
    if [ $SPEECH_EXIT -eq 0 ]; then
        echo -e "\n[4/6] Starting parallel processing tasks..."
        
        # Create directories for features
        AUDIO_FEATURES_DIR="$RESULTS_DIR/audio_features"
        VIDEO_FEATURES_DIR="$RESULTS_DIR/video_features"
        MULTIMODAL_FEATURES_DIR="$RESULTS_DIR/multimodal_features"
        mkdir -p "$AUDIO_FEATURES_DIR" "$VIDEO_FEATURES_DIR" "$MULTIMODAL_FEATURES_DIR"
        
        # 1. Start speech-to-text in parallel
        echo "Starting speech-to-text transcription in parallel..."
        (
            cmd="poetry run scripts/macos/run_speech_to_text.sh --input-dir \"$SPEECH_OUTPUT_DIR\" --output-dir \"$TRANSCRIPT_OUTPUT_DIR\" --diarize $BATCH_FLAG"
            echo "Executing: $cmd"
            eval $cmd
            TRANSCRIPT_EXIT=$?
            echo "TRANSCRIPT_EXIT=$TRANSCRIPT_EXIT" > "$TRANSCRIPT_SEMAPHORE"
            echo -e "\nSpeech-to-text completed with exit code $TRANSCRIPT_EXIT"
        ) &
        TRANSCRIPT_PID=$!
        
        # 2. Start audio feature extraction in parallel
        echo "Starting audio feature extraction in parallel..."
        (
            cmd="poetry run scripts/macos/extract_audio_features.sh --input-dir \"$SPEECH_OUTPUT_DIR\" --output-dir \"$AUDIO_FEATURES_DIR\" $BATCH_FLAG"
            echo "Executing: $cmd"
            eval $cmd
            AUDIO_FEATURES_EXIT=$?
            echo "AUDIO_FEATURES_EXIT=$AUDIO_FEATURES_EXIT" > "$AUDIO_FEATURES_SEMAPHORE"
            echo -e "\nAudio feature extraction completed with exit code $AUDIO_FEATURES_EXIT"
        ) &
        AUDIO_FEATURES_PID=$!
        
        # 3. Start emotion and pose recognition in parallel
        echo "Starting emotion and pose recognition in parallel..."
        (
            # Only proceed if videos were selected
            if [ ${#EMOTION_VIDEOS[@]} -eq 0 ]; then
                echo "No videos selected for emotion and pose recognition. Skipping this step."
                EMOTION_EXIT=1
            else
                # Pass individual video files and point to speech directory for audio sources
                video_args=""
                for video_path in "${EMOTION_VIDEOS[@]}"; do
                    # Get base filename for matching with separated speech
                    base_name=$(basename "$video_path" | sed 's/\.[^.]*$//')
                    video_args+=" --video \"$video_path\""
                    
                    # Check if corresponding audio file exists and add it specifically
                    audio_file="$SPEECH_OUTPUT_DIR/$base_name.wav"
                    if [ -f "$audio_file" ]; then
                        echo " - Found matching audio file for $base_name: $audio_file"
                        video_args+=" --audio-path \"$audio_file\""
                    fi
                done
                
                cmd="poetry run scripts/macos/run_emotion_and_pose_recognition.sh $video_args --output-dir \"$EMOTIONS_AND_POSE_DIR\" --feature-models all --speech-dir \"$SPEECH_OUTPUT_DIR\" $BATCH_FLAG"
                echo "Executing: $cmd"
                eval $cmd
                    
                EMOTION_EXIT=$?
            fi
            echo "EMOTION_EXIT=$EMOTION_EXIT" > "$EMOTION_SEMAPHORE"
            echo -e "\nEmotion and pose recognition completed with exit code $EMOTION_EXIT"
        ) &
        EMOTION_PID=$!
        
        # 4. Start video feature extraction in parallel  
        echo "Starting video feature extraction in parallel..."
        (
            # Only process if videos were selected
            if [ ${#EMOTION_VIDEOS[@]} -eq 0 ]; then
                echo "No videos selected for video feature extraction. Skipping this step."
                VIDEO_FEATURES_EXIT=1
            else
                # Build video arguments
                video_args=""
                for video_path in "${EMOTION_VIDEOS[@]}"; do
                    video_args+=" --video \"$video_path\""
                    echo " - Extracting video features from: $(basename "$video_path")"
                done
                
                # Extract visual-only features
                cmd="poetry run scripts/macos/extract_video_features.sh $video_args --output-dir \"$VIDEO_FEATURES_DIR\" $BATCH_FLAG"
                echo "Executing: $cmd"
                eval $cmd
                
                VIDEO_FEATURES_EXIT=$?
            fi
            echo "VIDEO_FEATURES_EXIT=$VIDEO_FEATURES_EXIT" > "$VIDEO_FEATURES_SEMAPHORE"
            echo -e "\nVideo feature extraction completed with exit code $VIDEO_FEATURES_EXIT"
        ) &
        VIDEO_FEATURES_PID=$!
        
        # 5. Start multimodal feature extraction in parallel
        echo "Starting multimodal feature extraction in parallel..."
        (
            # Only process if videos were selected
            if [ ${#EMOTION_VIDEOS[@]} -eq 0 ]; then
                echo "No videos selected for multimodal feature extraction. Skipping this step."
                MULTIMODAL_EXIT=1
            else
                # Build video-audio pair arguments
                video_args=""
                for video_path in "${EMOTION_VIDEOS[@]}"; do
                    # Get base filename for matching with separated speech
                    base_name=$(basename "$video_path" | sed 's/\.[^.]*$//')
                    video_args+=" --video \"$video_path\""
                    
                    # Check if corresponding audio file exists
                    audio_file="$SPEECH_OUTPUT_DIR/$base_name.wav"
                    if [ -f "$audio_file" ]; then
                        echo " - Found matching audio file for $base_name: $audio_file"
                        video_args+=" --audio-path \"$audio_file\""
                    fi
                    
                    echo " - Extracting multimodal features from: $base_name"
                done
                
                # Extract multimodal features
                cmd="poetry run scripts/macos/extract_multimodal_features.sh $video_args --speech-dir \"$SPEECH_OUTPUT_DIR\" --output-dir \"$MULTIMODAL_FEATURES_DIR\" $BATCH_FLAG"
                echo "Executing: $cmd"
                eval $cmd
                
                MULTIMODAL_EXIT=$?
            fi
            echo "MULTIMODAL_EXIT=$MULTIMODAL_EXIT" > "$MULTIMODAL_SEMAPHORE"
            echo -e "\nMultimodal feature extraction completed with exit code $MULTIMODAL_EXIT"
        ) &
        MULTIMODAL_PID=$!
        
        # Wait for all parallel processes to complete
        echo "Waiting for all parallel processes to complete..."
        wait $TRANSCRIPT_PID
        wait $AUDIO_FEATURES_PID
        wait $VIDEO_FEATURES_PID
        wait $MULTIMODAL_PID
        wait $EMOTION_PID
        
        # Read exit statuses
        [ -f "$TRANSCRIPT_SEMAPHORE" ] && source "$TRANSCRIPT_SEMAPHORE" || TRANSCRIPT_EXIT=1
        [ -f "$AUDIO_FEATURES_SEMAPHORE" ] && source "$AUDIO_FEATURES_SEMAPHORE" || AUDIO_FEATURES_EXIT=1
        [ -f "$VIDEO_FEATURES_SEMAPHORE" ] && source "$VIDEO_FEATURES_SEMAPHORE" || VIDEO_FEATURES_EXIT=1
        [ -f "$MULTIMODAL_SEMAPHORE" ] && source "$MULTIMODAL_SEMAPHORE" || MULTIMODAL_EXIT=1
        [ -f "$EMOTION_SEMAPHORE" ] && source "$EMOTION_SEMAPHORE" || EMOTION_EXIT=1
        
        # Clean up semaphore directory
        rm -rf "$SEMAPHORE_DIR"
    else
        echo -e "\nSpeech separation failed with exit code $SPEECH_EXIT"
        echo "Cannot proceed with subsequent steps that require separated speech."
        TRANSCRIPT_EXIT=1
        AUDIO_FEATURES_EXIT=1
        VIDEO_FEATURES_EXIT=1
        MULTIMODAL_EXIT=1
        EMOTION_EXIT=1
    fi
    
    # Report the final status of all pipeline steps
    echo -e "\n===== Pipeline Execution Summary ====="
    echo "- Speech Separation: $([ $SPEECH_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    echo "- Speech-to-Text: $([ $TRANSCRIPT_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    echo "- Audio Feature Extraction: $([ $AUDIO_FEATURES_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    echo "- Video Feature Extraction: $([ $VIDEO_FEATURES_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    echo "- Multimodal Feature Extraction: $([ $MULTIMODAL_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    echo "- Emotion and Pose Recognition: $([ $EMOTION_EXIT -eq 0 ] && echo "✅ Success" || echo "❌ Failed")"
    
    # Create pipeline_output.csv that merges all results
    echo -e "\n[5/6] Creating pipeline output CSV with combined features..."
    
    # Pass all feature directories to merge script
    AUDIO_FEATURES_DIR="$RESULTS_DIR/audio_features"
    VIDEO_FEATURES_DIR="$RESULTS_DIR/video_features"
    MULTIMODAL_FEATURES_DIR="$RESULTS_DIR/multimodal_features"
    
    poetry run python -c "from utils.merge_features import create_pipeline_output; create_pipeline_output('$RESULTS_DIR', speech_features_dir='$AUDIO_FEATURES_DIR', video_features_dir='$VIDEO_FEATURES_DIR', multimodal_features_dir='$MULTIMODAL_FEATURES_DIR')"
    CSV_EXIT=$?
    
    if [ $CSV_EXIT -eq 0 ]; then
        echo "✅ Pipeline output CSV created successfully"
        echo "- CSV output with combined features: $RESULTS_DIR/pipeline_output.csv"
        echo "- Summary report: $RESULTS_DIR/pipeline_summary.txt"
        echo "- Combined features across all runs: $PROJECT_ROOT/output/combined_features.csv"
        echo "- Pipeline history: $PROJECT_ROOT/output/pipeline_history.csv"
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
    echo "- Source videos: $VIDEOS_DIR"
    echo "- Separated speech: $SPEECH_OUTPUT_DIR"
    echo "- Transcripts: $TRANSCRIPT_OUTPUT_DIR"
    echo "- Audio features: $AUDIO_FEATURES_DIR"
    echo "- Video features: $VIDEO_FEATURES_DIR"
    echo "- Multimodal features: $MULTIMODAL_FEATURES_DIR"
    echo "- Emotion and pose analysis: $EMOTIONS_AND_POSE_DIR"
    echo "- Combined features (all runs): $PROJECT_ROOT/output/combined_features.csv"
fi