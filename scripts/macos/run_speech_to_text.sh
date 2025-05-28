#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing Speech-to-Text ==="
echo "This script transcribes speech audio files to text."

# Get the script's directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "\nPoetry is not installed. Installing poetry is required for dependency management."
    echo "Please install Poetry with: curl -sSL https://install.python-poetry.org | python3 -"
    echo "All dependencies are defined in pyproject.toml"
    exit 1
else
    # Install dependencies using Poetry
    echo -e "\n[1/2] Installing dependencies with Poetry..."
    poetry install --with speech --with common || {
        echo "Poetry installation had issues. Retrying with common dependencies only..."
        poetry install --with common
    }
fi

# Install huggingface_hub CLI support
echo "Installing Hugging Face CLI dependencies..."
poetry run pip install --upgrade "huggingface_hub[cli]" > /dev/null || echo "Warning: Could not install huggingface_hub CLI"

# Check for Hugging Face token - prioritize environment variable
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "Using Hugging Face token from environment."
    
    # Ensure the token is properly configured for huggingface-hub
    poetry run python -c "from huggingface_hub import HfFolder; HfFolder.save_token('$HUGGINGFACE_TOKEN')" || \
        echo "Warning: Failed to save Hugging Face token to hub folder"
else
    # ONLY if not running in a subprocess (check for interactive terminal)
    if [ -t 0 ]; then
        echo "No Hugging Face token found in environment. Attempting login..."
        if ! poetry run huggingface-cli login; then
            echo "Hugging Face login failed or was canceled."
            echo "Speaker diarization might not work correctly."
        fi
    else
        echo "Running in non-interactive mode without Hugging Face token."
        echo "Speaker diarization might not work correctly."
    fi
fi

# Help message if --help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./run_speech_to_text.sh [options] <audio_file(s)>"
    echo ""
    echo "Options:"
    echo "  --input-dir DIR      Directory containing input audio files"
    echo "  --output-dir DIR     Directory to save transcription files (default: ./output/transcripts)"
    echo "  --model MODEL        Speech-to-text model to use (whisperx, xlsr)"
    echo "  --language LANG      Language code for transcription (default: en)"
    echo "  --output-format FMT  Output format: srt, txt, or both (default: srt)"
    echo "  --recursive          Process audio files in subdirectories recursively"
    echo "  --select             Force file selection prompt even when files are provided"
    echo "  --debug              Enable debug logging"
    echo "  --interactive        Force interactive audio selection mode"
    echo "  --no-diarize         Disable speaker diarization (speaker detection is enabled by default)"
    echo "  --batch              Enable batch mode to process all files without manual selection"
    echo "  --extract-features   Extract audio features and save as CSV files in the audio directory"
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive audio selection menu."
    exit 0
fi

# Look for input directory in arguments
input_dir=""
output_dir=""
no_diarize=false
batch_mode=false
extract_features=true  # Always extract features by default
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
    elif [ "$arg" == "--no-diarize" ]; then
        no_diarize=true
    elif [ "$arg" == "--batch" ]; then
        batch_mode=true
    elif [ "$arg" == "--extract-features" ]; then
        extract_features=true  # Redundant now but kept for backward compatibility
    else
        other_args+=("$arg")
    fi
    i=$((i+1))
done

# Run the speech-to-text script
echo -e "\n[2/2] Running speech-to-text transcription..."

# Build command based on input parameters
cmd_args=()

# Add input directory if specified
if [ -n "$input_dir" ]; then
    echo "Using input directory: $input_dir"
    cmd_args+=("--input-dir" "$input_dir")
fi

# Add output directory if specified
if [ -n "$output_dir" ]; then
    cmd_args+=("--output-dir" "$output_dir")
fi

# Add diarize flag by default unless explicitly disabled
if [ "$no_diarize" = false ]; then
    echo "Speaker detection (diarization) is enabled"
    cmd_args+=("--diarize")
else
    echo "Speaker detection (diarization) is disabled"
fi

# Add batch mode flag if specified
if [ "$batch_mode" = true ]; then
    echo "Running in batch mode - processing all files without manual selection"
    cmd_args+=("--batch")
fi

# Add feature extraction flag - now always enabled
echo "Audio feature extraction is enabled - CSV files will be created in the audio directory"
cmd_args+=("--extract-features")

# Add other arguments
for arg in "${other_args[@]}"; do
    cmd_args+=("$arg")
done

# Use Poetry to run the script
if [ ${#cmd_args[@]} -eq 0 ] && [ ${#other_args[@]} -eq 0 ]; then
    echo "Entering interactive mode..."
    # Add diarize flag to interactive mode too
    if [ "$no_diarize" = false ]; then
        poetry run python -m src.speech_to_text --interactive --diarize
    else
        poetry run python -m src.speech_to_text --interactive
    fi
else
    # Otherwise, pass all arguments to the script
    poetry run python -m src.speech_to_text "${cmd_args[@]}"
fi

if [ $? -eq 0 ]; then
    echo -e "\nSpeech-to-text transcription process completed successfully."
else
    echo -e "\nAn error occurred during the speech-to-text transcription process."
    exit 1
fi
