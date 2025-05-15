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
# If URL is not provided, prompt the user for it
if [ "$url_provided" = false ]; then
    read -p "Please enter the SharePoint folder URL: " url
    if [[ ! "$url" =~ ^https?:// ]] || [[ ! "$url" =~ sharepoint\.com ]]; then
        echo "Error: The provided URL is not valid. Please provide a valid SharePoint URL."
        exit 1
    fi
fi

# Run the download script with all arguments using Poetry
echo -e "\nRunning SharePoint downloader..."

# Add a check to see if the output directory is provided in script_args
output_dir_provided=false
for arg in "${script_args[@]}"; do
    if [ "$arg" == "--output-dir" ]; then
        output_dir_provided=true
        break
    fi
done

# Run the download script
poetry run python -m src.download_videos.main --url "$url" "${script_args[@]}"
DOWNLOAD_EXIT=$?

if [ $DOWNLOAD_EXIT -eq 0 ]; then
    # Determine the output directory that was used
    DEFAULT_VIDEOS_DIR="./data/videos"
    OUTPUT_DIR="$DEFAULT_VIDEOS_DIR"
    
    # Parse the output directory from args if provided
    i=0
    while [ $i -lt ${#script_args[@]} ]; do
        if [ "${script_args[$i]}" == "--output-dir" ] && [ $((i+1)) -lt ${#script_args[@]} ]; then
            OUTPUT_DIR="${script_args[$((i+1))]}"
            break
        fi
        i=$((i+1))
    done
    
    # Verify that files exist in the output directory
    VIDEO_COUNT=$(find "$OUTPUT_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.MP4" -o -name "*.MOV" -o -name "*.AVI" -o -name "*.MKV" \) | wc -l | tr -d '[:space:]')
    
    if [ "$VIDEO_COUNT" -gt 0 ]; then
        echo -e "\nSuccessfully downloaded ${VIDEO_COUNT} video files to: ${OUTPUT_DIR}"
        echo "Files available:"
        find "$OUTPUT_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.MP4" -o -name "*.MOV" -o -name "*.AVI" -o -name "*.MKV" \) -exec basename {} \;
    else
        echo -e "\nWarning: No video files were found in the output directory: ${OUTPUT_DIR}"
        echo "Available files (if any):"
        ls -la "$OUTPUT_DIR"
    fi
fi

echo -e "\nDownload process completed with exit code: $DOWNLOAD_EXIT"
exit $DOWNLOAD_EXIT
