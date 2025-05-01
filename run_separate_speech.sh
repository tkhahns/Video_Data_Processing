#!/bin/bash

echo "=== Video Data Processing Speech Separation ==="
echo "This script extracts and isolates speech from video files."

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
if ! python -c "import speechbrain tqdm pydub" &> /dev/null; then
    echo -e "\nInstalling required packages..."
    pip install speechbrain moviepy torchaudio tqdm pydub ffmpeg-python
fi

# Help message if --help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./run_separate_speech.sh [options] <video_file(s)>"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR     Directory to save separated speech files (default: ./output/separated_speech)"
    echo "  --model MODEL        Speech separation model to use (sepformer, conv-tasnet)" 
    echo "  --file-type TYPE     Output file format: wav (1), mp3 (2), or both (3) (default: mp3)"
    echo "  --recursive          Process video files in subdirectories recursively"
    echo "  --debug              Enable debug logging"
    echo "  --interactive        Force interactive video selection mode"
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive video selection menu."
    exit 0
fi

# Run the speech separation script
echo -e "\nRunning speech separation..."

# If no arguments are provided, use interactive mode
if [ $# -eq 0 ]; then
    echo "Entering interactive mode..."
    python -m src.separate_speech --interactive
else
    # Otherwise, pass all arguments to the script
    python -m src.separate_speech "$@"
fi

if [ $? -eq 0 ]; then
    echo -e "\nSpeech separation process completed successfully."
else
    echo -e "\nAn error occurred during the speech separation process."
    exit 1
fi
