#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing: Multimodal Feature Extraction ==="
echo "This script extracts multimodal features from video and audio pairs."

# Get the script's directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root
cd "$PROJECT_ROOT"

# Check for Hugging Face token in environment
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo -e "\n=== Hugging Face Authentication ==="
    echo "This tool requires a Hugging Face token for accessing models."
    echo "You can get your token from: https://huggingface.co/settings/tokens"
    echo "Note: Your token will only be used for this session and will not be saved."
    
    # Prompt for token
    read -sp "Enter your Hugging Face token (input will be hidden): " HUGGINGFACE_TOKEN
    echo ""
    
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
    poetry install --with emotion --with speech --with common || {
        echo "Poetry installation had issues. Retrying with common dependencies only..."
        poetry install --with common
    }
fi

# Help message if --help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./extract_multimodal_features.sh [options]"
    echo ""
    echo "Options:"
    echo "  --video FILE         Video file to process (can be specified multiple times)"
    echo "  --audio-path FILE    Audio file to use with corresponding video (can be specified multiple times)"
    echo "  --speech-dir DIR     Directory containing separated speech audio files"
    echo "  --output-dir DIR     Directory to save feature files (default: ./output/multimodal_features)"
    echo "  --models LIST        Space-separated list of models to use: av_hubert meld all"
    echo "  --batch              Process all files without manual selection"
    echo "  --debug              Enable debug logging"
    echo "  --json-indent INT    Indentation level for JSON output files (default: 2)"
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive file selection menu."
    exit 0
fi

# Look for input parameters in arguments
output_dir=""
batch_mode=false
video_files=()
audio_paths=()
speech_dir=""
models=""
json_indent=2  # Default JSON indent
other_args=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" == "--speech-dir" ] && [ $i -lt $# ]; then
        i=$((i+1))
        speech_dir="${!i}"
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
    elif [ "$arg" == "--models" ] && [ $i -lt $# ]; then
        i=$((i+1))
        models="${!i}"
    elif [ "$arg" == "--json-indent" ] && [ $i -lt $# ]; then
        i=$((i+1))
        json_indent="${!i}"
    else
        other_args+=("$arg")
    fi
    i=$((i+1))
done

# Set environment variable for JSON indentation
export JSON_INDENT=$json_indent
echo "Using JSON indentation level: $json_indent"

# Run the multimodal feature extraction script
echo -e "\n[2/2] Running multimodal feature extraction..."

# Build command based on input parameters
cmd_args=()

# Add individual video and audio pairs if provided
for ((i=0; i<${#video_files[@]}; i++)); do
    video="${video_files[$i]}"
    echo "Processing video file: $video"
    cmd_args+=("--video" "$video")
    
    # If we have a matching audio path for this video, add it
    if [ $i -lt ${#audio_paths[@]} ]; then
        echo "  with matching audio file: ${audio_paths[$i]}"
        cmd_args+=("--audio-path" "${audio_paths[$i]}")
    fi
done

# Add speech directory for finding matching audio files
if [ -n "$speech_dir" ]; then
    echo "Using speech directory: $speech_dir"
    cmd_args+=("--speech-dir" "$speech_dir")
fi

# Add output directory if specified
if [ -n "$output_dir" ]; then
    cmd_args+=("--output-dir" "$output_dir")
fi

# Add feature models if specified - use multimodal models by default
if [ -n "$models" ]; then
    echo "Using multimodal models: $models"
    cmd_args+=("--models" $models)
else
    echo "Using default multimodal feature models"
    cmd_args+=("--models" "av_hubert meld")
fi

# Add batch mode flag if specified
if [ "$batch_mode" = true ]; then
    echo "Running in batch mode - processing all files without manual selection"
    cmd_args+=("--batch")
fi

# Add JSON indentation parameter
cmd_args+=("--json-indent" "$json_indent")

# Add other arguments
for arg in "${other_args[@]}"; do
    cmd_args+=("$arg")
done

# Use Poetry to run the script
if [ ${#cmd_args[@]} -eq 0 ] && [ ${#other_args[@]} -eq 0 ]; then
    echo "Entering interactive mode..."
    poetry run python -m src.emotion_and_pose_recognition.multimodal_features --interactive --json-indent "$json_indent"
else
    # Otherwise, pass all arguments to the script
    poetry run python -m src.emotion_and_pose_recognition.multimodal_features "${cmd_args[@]}"
fi

if [ $? -eq 0 ]; then
    echo -e "\nMultimodal feature extraction completed successfully."
else
    echo -e "\nAn error occurred during multimodal feature extraction."
    exit 1
fi
