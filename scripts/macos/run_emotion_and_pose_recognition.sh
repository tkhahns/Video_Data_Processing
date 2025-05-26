#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing Emotion and Pose Recognition ==="
echo "This script analyzes emotions and body poses in video files."

# Get the script's directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root
cd "$PROJECT_ROOT"

# Check for Hugging Face token in environment - FIXED to handle non-interactive mode
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    # Check if script is running interactively
    if [ -t 0 ]; then
        echo -e "\n=== Hugging Face Authentication ==="
        echo "This tool requires a Hugging Face token for accessing models."
        echo "You can get your token from: https://huggingface.co/settings/tokens"
        echo "Note: Your token will only be used for this session and will not be saved."
        
        # Prompt for token
        read -sp "Enter your Hugging Face token (input will be hidden): " HUGGINGFACE_TOKEN
        echo ""
    else
        echo "Warning: Running in non-interactive mode without Hugging Face token."
        echo "Some features may not work correctly."
    fi
    
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "No token provided. Some features may not work correctly."
    else
        echo "Token received for this session"
    fi
    
    export HUGGINGFACE_TOKEN
fi

# Setup function to delete token on exit
cleanup_token() {
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        echo "Clearing Hugging Face token from environment"
        unset HUGGINGFACE_TOKEN
    fi
}

# Register the cleanup function to run on script exit
trap cleanup_token EXIT

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "\nPoetry is not installed. Installing poetry is required for dependency management."
    echo "Please install Poetry with: curl -sSL https://install.python-poetry.org | python3 -"
    echo "All dependencies are defined in pyproject.toml"
    exit 1
else
    # Install dependencies using Poetry
    echo -e "\n[1/2] Installing dependencies with Poetry..."
    poetry install --with emotion --with common || {
        echo "Poetry installation had issues. Retrying with common dependencies only..."
        poetry install --with common
    }
fi

# Help message if --help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./run_emotion_and_pose_recognition.sh [options] <video_file>"
    echo ""
    echo "Options:"
    echo "  --input-dir DIR      Directory containing input video files"
    echo "  --video FILE         Single video file to process"
    echo "  --audio-path FILE    Path to audio file for multimodal analysis"
    echo "  --speech-dir DIR     Directory containing separated speech audio files"
    echo "  --output-dir DIR     Directory to save emotion analysis results (default: ./output/emotions)"
    echo "  --batch              Process all videos in input directory without prompting"
    echo "  --interactive        Force interactive video selection mode"
    echo "  --single-speaker     Disable multi-speaker tracking (only track one person)"
    echo "  --debug              Enable debug logging"
    echo "  --help               Show this help message"
    echo ""
    echo "Models:"
    echo "  --feature-models     Specify which feature extraction models to use (space-separated):"
    echo "                       mediapipe pyfeat optical_flow av_hubert meld pare vitpose psa rsn au_detector dan eln all"
    echo ""
    echo "Note: Feature extraction is always enabled by default."
    echo "By default, the script processes videos with pose estimation, multi-speaker tracking,"
    echo "and uses all available feature models including multimodal ones."
    echo "If run without arguments, the script will show an interactive video selection menu."
    exit 0
fi

# Look for input parameters in arguments
input_dir=""
output_dir=""
batch_mode=false
video_files=()
audio_paths=()
speech_dir=""
other_args=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" == "--input-dir" ] && [ $i -lt $# ]; then
        i=$((i+1))
        input_dir="${!i}"
    elif [ "$arg" == "--output-dir" ] && [ $i -lt $# ]; then
        i=$((i+1))
        output_dir="${!i}"
    elif [ "$arg" == "--batch" ]; then
        batch_mode=true
    elif [ "$arg" == "--video" ] && [ $i -lt $# ]; then
        i=$((i+1))
        video_files+=("${!i}")
    elif [ "$arg" == "--audio-path" ] && [ $i -lt $# ]; then
        i=$((i+1))
        audio_paths+=("${!i}")
    elif [ "$arg" == "--speech-dir" ] && [ $i -lt $# ]; then
        i=$((i+1))
        speech_dir="${!i}"
    else
        other_args+=("$arg")
    fi
    i=$((i+1))
done

# Run the emotion and pose recognition script
echo -e "\n[2/2] Running emotion and pose recognition analysis..."
echo "Feature extraction is enabled - Video features will be extracted"

# Build command based on input parameters
cmd_args=()

# Add input directory if specified and no video files were provided
if [ -n "$input_dir" ] && [ ${#video_files[@]} -eq 0 ]; then
    echo "Using input directory: $input_dir"
    cmd_args+=("--input-dir" "$input_dir")
fi

# Add individual video files if provided
for ((i=0; i<${#video_files[@]}; i++)); do
    video="${video_files[$i]}"
    echo "Adding video file: $video"
    cmd_args+=("--video" "$video")
    
    # If we have a matching audio path for this video, add it
    if [ $i -lt ${#audio_paths[@]} ]; then
        echo "  with matching audio file: ${audio_paths[$i]}"
        cmd_args+=("--audio-path" "${audio_paths[$i]}")
    fi
done

# Add speech directory for the CLI to find matching audio files
if [ -n "$speech_dir" ]; then
    echo "Using speech directory: $speech_dir"
    cmd_args+=("--speech-dir" "$speech_dir")
fi

# Add output directory if specified
if [ -n "$output_dir" ]; then
    cmd_args+=("--output-dir" "$output_dir")
fi

# Always add extract-features flag
cmd_args+=("--extract-features")

# Add feature models if specified - use all models by default
if [ -n "$feature_models" ]; then
    echo "Using feature models: $feature_models"
    cmd_args+=("--feature-models" $feature_models)
else
    echo "Using all available feature models"
    cmd_args+=("--feature-models" "all")
fi

# Always add pose estimation by default (unless --no-pose is explicitly included)
NO_POSE_PRESENT=false
for arg in "${other_args[@]}"; do
  if [ "$arg" == "--no-pose" ]; then
    NO_POSE_PRESENT=true
    break
  fi
done

if [ "$NO_POSE_PRESENT" = false ]; then
    cmd_args+=("--with-pose")
fi

# Always use multi-speaker by default (unless --single-speaker is explicitly included)
SINGLE_SPEAKER_PRESENT=false
for arg in "${other_args[@]}"; do
  if [ "$arg" == "--single-speaker" ]; then
    SINGLE_SPEAKER_PRESENT=true
    break
  fi
done

if [ "$SINGLE_SPEAKER_PRESENT" = false ]; then
    cmd_args+=("--multi-speaker")
fi

# Add batch mode flag if specified
if [ "$batch_mode" = true ]; then
    echo "Running in batch mode - processing all files without manual selection"
    cmd_args+=("--batch")
fi

# Add other arguments
for arg in "${other_args[@]}"; do
    cmd_args+=("$arg")
done

# Use Poetry to run the script
if [ ${#cmd_args[@]} -eq 0 ] && [ ${#other_args[@]} -eq 0 ]; then
    echo "Entering interactive mode with pose estimation, multi-speaker tracking, and feature extraction..."
    poetry run python -m src.emotion_and_pose_recognition.cli --with-pose --multi-speaker --interactive --extract-features
else
    # Otherwise, pass all arguments to the script
    poetry run python -m src.emotion_and_pose_recognition.cli "${cmd_args[@]}"
fi

if [ $? -eq 0 ]; then
    echo -e "\nEmotion and pose recognition analysis completed successfully."
else
    echo -e "\nAn error occurred during emotion and pose recognition analysis."
    exit 1
fi