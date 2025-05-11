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
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive audio selection menu."
    exit 0
fi

# Look for input directory in arguments
input_dir=""
output_dir=""
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

# Add other arguments
for arg in "${other_args[@]}"; do
    cmd_args+=("$arg")
done

# Use Poetry to run the script
if [ ${#cmd_args[@]} -eq 0 ] && [ ${#other_args[@]} -eq 0 ]; then
    echo "Entering interactive mode..."
    poetry run python -m src.speech_to_text --interactive
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
