#!/bin/bash

echo "=== Video Data Processing Emotion Recognition ==="
echo "This script detects and labels emotions in video files."

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
if ! python -c "import feat cv2 numpy tqdm" &> /dev/null; then
    echo -e "\nInstalling required packages..."
    pip install py-feat==0.5.1 opencv-python numpy tqdm
fi

# Help message if --help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./run_emotion_recognition.sh [options] <video_file(s)>"
    echo -e "\nOptions:"
    echo "  --output-dir DIR     Directory to save emotion recognition results (default: ./output/emotions)"
    echo "  --input DIR/FILE     Input directory or video file"
    echo "  --process-all        Process all video files in the input directory"
    echo "  --interval NUM       Frame interval in seconds (default: 1.0)"
    echo "  --threshold NUM      Confidence threshold for emotions (default: 0.5)"
    echo "  --device TYPE        Device to run on: cpu or cuda (default: cpu)"
    echo "  --recursive          Process video files in subdirectories recursively"
    echo "  --debug              Enable debug logging"
    echo "  --interactive        Force interactive video selection mode"
    echo "  --help               Show this help message"
    echo ""
    echo "If run without arguments, the script will show an interactive video selection menu."
    exit
fi

# Run the emotion recognition script
echo -e "\nRunning emotion recognition..."

if [ $# -eq 0 ]; then
    echo "Entering interactive mode..."
    python -m src.emotion_recognition_main --interactive
else
    # Otherwise, pass all arguments to the script
    python -m src.emotion_recognition_main "$@"
fi

if [ $? -eq 0 ]; then
    echo -e "\nEmotion recognition process completed successfully."
else
    echo -e "\nAn error occurred during the emotion recognition process."
    exit 1
fi
