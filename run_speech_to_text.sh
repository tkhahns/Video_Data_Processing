#!/bin/bash

echo "=== Video Data Processing Speech-to-Text ==="
echo "This script transcribes speech audio files to text."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "\n[1/2] Creating virtual environment..."
    python -m venv .venv
else
    echo -e "\n[1/2] Using existing virtual environment."
fi

# Activate virtual environment
echo -e "\n[2/2] Activating virtual environment..."
source .venv/bin/activate

# Install required packages if needed
if ! python -c "import torch transformers colorama" &> /dev/null; then
    echo -e "\nInstalling required packages..."
    pip install torch transformers tqdm colorama
    pip install git+https://github.com/m-bain/whisperX.git
fi

# Help message if --help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./run_speech_to_text.sh [options] <audio_file(s)>"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR     Directory to save transcription files (default: ./output/transcripts)"
    echo "  --model MODEL        Speech-to-text model to use (whisperx, xlsr)"
    echo "  --language LANG      Language code for transcription (default: en)"
    echo "  --output-format FMT  Output format: srt, txt, or both (default: srt)"
    echo "  --recursive          Process audio files in subdirectories recursively"
    echo "  --select             Force file selection prompt even when files are provided"
    echo "  --debug              Enable debug logging"
    echo "  --interactive        Force interactive audio selection mode"
    echo "  --diarize            Enable speaker diarization (identifies different speakers in transcript)"
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive audio selection menu."
    exit 0
fi

# Run the speech-to-text script
echo -e "\nRunning speech-to-text transcription..."

# If no arguments are provided, use interactive mode
if [ $# -eq 0 ]; then
    echo "Entering interactive mode..."
    python -m src.speech_to_text --interactive --diarize
else
    # Otherwise, pass all arguments to the script with diarize flag
    python -m src.speech_to_text --diarize "$@"
fi

if [ $? -eq 0 ]; then
    echo -e "\nSpeech-to-text transcription process completed successfully."
else
    echo -e "\nAn error occurred during the speech-to-text transcription process."
    exit 1
fi
