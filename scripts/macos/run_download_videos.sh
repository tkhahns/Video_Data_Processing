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

# Get the script's directory and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to the project root directory
cd "$PROJECT_ROOT"

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

# If URL was provided in arguments, validate it
if [ "$url_provided" = true ]; then
    if [[ ! "$url" =~ ^https?:// ]] || [[ ! "$url" =~ sharepoint\.com ]]; then
        echo "Warning: The provided URL might not be valid."
        read -p "Do you want to continue with this URL? (y/n): " continue_with_url
        if [[ ! "$continue_with_url" =~ ^[Yy] ]]; then
            url_provided=false
        fi
    fi
fi

# If URL wasn't provided or was rejected, get a valid URL
if [ "$url_provided" = false ]; then
    url=$(validate_url "Please enter the SharePoint URL containing the videos you want to download: ")
fi

# Run the download script with all arguments using Poetry
echo -e "\nRunning SharePoint downloader..."
poetry run python -m src.download_videos.main --url "$url" "${script_args[@]}"

echo -e "\nDownload process completed."
exit $?
