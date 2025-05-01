## Video Data Processing Setup Instructions

These instructions will help you set up the environment for the Video Data Processing application on Windows.

---

## Prerequisites

Ensure you have **Python 3.12.10** installed on your system:

```bash
# Download and install Python 3.12.10 from the official Python website
# https://www.python.org/downloads/release/python-31210/

# Verify the Python version (in Command Prompt or PowerShell)
python --version    # → Python 3.12.10
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/tkhahns/Video_Data_Processing.git
cd Video_Data_Processing
```

### 2. Create and activate a virtual environment

```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

### 3. Update pip and install dependencies

```bash
# Update pip to the latest version
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install huggingface-hub
pip install huggingface-hub
```

---

## Fetch all models

You can download the required models using either of these methods:

### Option 1: Using the automated script (recommended)

```bash
# Open PowerShell and navigate to the project directory
# You may need to set execution policy first
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Run the script
.\run_download_models.ps1
```

This script automatically:
- Creates and activates the virtual environment
- Updates pip and installs dependencies
- Creates the models directory
- Downloads all required models

### Option 2: Running as a module

```bash
# With virtual environment activated
python -m src.download_models
```

### Option 3: Additional options

```bash
# Download only specific model types
python -m src.download_models --model-types audio video

# Force re-download of existing models
python -m src.download_models --force

# Preview what would be downloaded without downloading
python -m src.download_models --dry-run
```

---

## Troubleshooting

- If you encounter `ModuleNotFoundError` despite the package being installed, try:
  ```bash
  pip uninstall [package-name]
  pip install [package-name]
  ```

- If you're having permission issues, try running Command Prompt or PowerShell as Administrator

- If you still have import errors, verify you're using the correct Python interpreter:
  ```bash
  where python
  # Should point to your virtual environment's .venv\Scripts\python.exe
  ```

---

## Download Videos from SharePoint

You can download videos from SharePoint using the included browser automation tool:

### Option 1: Using the convenience script (recommended)

```bash
# Run the script with no arguments (you'll be prompted for the URL)
.\run_download_videos.ps1

# Or specify the URL directly
.\run_download_videos.ps1 --url "https://your-sharepoint-site.com/folder-with-videos"

# Additional options:
.\run_download_videos.ps1 --url "https://your-sharepoint-site.com/folder-with-videos" --output-dir "./my-videos"
.\run_download_videos.ps1 --list-only
.\run_download_videos.ps1 --debug
```

This script automatically:
- Activates the virtual environment
- Prompts for SharePoint URL if not provided
- Handles errors gracefully

### Option 2: Running as a module

```bash
# Using Python module syntax
python -m src.download_videos --url "https://your-sharepoint-site.com/folder-with-videos"

# Shorthand version
python src/download_videos --url "https://your-sharepoint-site.com/folder-with-videos"

# Additional options
python src/download_videos --url "https://your-sharepoint-site.com/folder-with-videos" --output-dir "./my-videos"
python src/download_videos --list-only --url "https://your-sharepoint-site.com/folder-with-videos"
python src/download_videos --debug --url "https://your-sharepoint-site.com/folder-with-videos"
```

### Option 3: Running the Python script directly

```bash
# Basic usage
python src/download_videos/main.py --url "https://your-sharepoint-site.com/folder-with-videos"

# Save to a specific directory
python src/download_videos/main.py --url "https://your-sharepoint-site.com/folder-with-videos" --output-dir "./my-videos"

# Just list files without downloading
python src/download_videos/main.py --url "https://your-sharepoint-site.com/folder-with-videos" --list-only

# Enable debug mode for troubleshooting
python src/download_videos/main.py --url "https://your-sharepoint-site.com/folder-with-videos" --debug
```

**Note:** The tool requires authentication to SharePoint. You'll need to sign in through the browser window that opens.