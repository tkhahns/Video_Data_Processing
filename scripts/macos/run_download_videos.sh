#!/bin/bash

# Exit on error
set -e

echo "=== Video Data Processing SharePoint Downloader ==="
echo "This script simplifies downloading videos from SharePoint."

# Get the script's directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root (instead of script directory)
cd "$PROJECT_ROOT"

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

# Display help if help flag is provided
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo -e "\nUsage: ./run_download_videos.sh [options]"
    echo ""
    echo "Options:"
    echo "  --url URL             SharePoint folder URL containing videos (optional)"
    echo "  --output-dir PATH     Directory to save downloaded files (default: ./data/videos)"
    echo "  --list-only           Just list files without downloading"
    echo "  --debug               Enable debug mode with detailed logging"
    echo ""
    exit 0
fi

# Check if URL is provided in arguments
url_provided=false
script_args=()
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" == "--url" ] && [ $i -lt $# ]; then
        url_next=$((i+1))
        url="${!url_next}"
        url_provided=true
        i=$((i+2))
    else
        script_args+=("$arg")
        i=$((i+1))
    fi
done

# If URL wasn't provided in command line, prompt for it
if [ "$url_provided" = false ]; then
    echo -e "\nPlease enter the SharePoint URL containing the videos you want to download:"
    read -r url
    
    # Validate that a URL was entered
    while [ -z "$url" ]; do
        echo "URL cannot be empty. Please enter a valid SharePoint URL:"
        read -r url
    done
fi

# Run the download script with all arguments
echo -e "\nRunning SharePoint downloader..."
python src/download_videos/main.py --url "$url" "${script_args[@]}"

echo -e "\nDownload process completed."
