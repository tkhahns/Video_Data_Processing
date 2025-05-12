# Video Data Processing - Windows Setup Instructions

These instructions will help you set up the environment for the Video Data Processing application on Windows.

---

## Prerequisites

### Python 3.12 (Required)

This project requires Python 3.12 specifically:

```powershell
# Check Python version
python --version

# Install Python 3.12 from the official website
# https://www.python.org/downloads/windows/

# Ensure Python is added to PATH during installation

# Verify Python version
python --version  # Should show Python 3.12.x
```

### Poetry for Dependency Management

This project uses Poetry for dependency management:

```powershell
# Install Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Add Poetry to your PATH
# Poetry is typically installed to %APPDATA%\Python\Scripts
# You may need to add this directory to your PATH

# Verify Poetry installation
poetry --version

# Configure Poetry to use Python 3.12
poetry env use python3.12
```

### FFmpeg

FFmpeg is required for audio and video processing:

```powershell
# Install using Chocolatey (recommended)
choco install ffmpeg

# Or download from official website
# https://ffmpeg.org/download.html

# Verify installation
ffmpeg -version
```

---

## Setup Instructions

### 1. Clone the repository

```powershell
git clone https://github.com/tkhahns/Video_Data_Processing.git
cd Video_Data_Processing
```

### 2. Install dependencies with Poetry

```powershell
# Configure Poetry to use Python 3.12 for this project
poetry env use python3.12

# Install main dependencies
poetry install

# Install all feature groups
poetry install --with common
poetry install --with speech
poetry install --with emotion
poetry install --with download
```

---

## Complete Processing Pipeline

The `run_all.ps1` script provides a complete end-to-end pipeline that:
1. Downloads videos from SharePoint
2. Extracts speech from the videos
3. Transcribes the speech to text
4. Analyzes emotions and body poses in the videos

All outputs are organized in timestamped directories for easy tracking.

### Running the Complete Pipeline

```powershell
# Run the pipeline (will prompt for SharePoint URL)
.\run_all.ps1

# Or specify the SharePoint URL directly
.\run_all.ps1 --url "https://your-sharepoint-site.com/folder-with-videos"
```

### Pipeline Outputs

The pipeline creates the following directory structure:
- `data\downloads_TIMESTAMP\`: Original downloaded videos
- `output\pipeline_results_TIMESTAMP\`: Processing results
  - `speech\`: Extracted speech audio files
  - `transcripts\`: Text transcription files
  - `emotions_and_pose\`: Emotion and body pose analysis

---

## Individual Components

Each component can also be run separately if you need to process only specific steps.

### 1. Download Videos from SharePoint

The download module now uses a manual approach where you interact with SharePoint in the browser:

```powershell
# Run with Poetry (interactive mode)
poetry run .\scripts\windows\run_download_videos.ps1

# Specify URL directly
poetry run .\scripts\windows\run_download_videos.ps1 --url "https://your-sharepoint-site.com/folder-with-videos"

# Additional options
poetry run .\scripts\windows\run_download_videos.ps1 --url "URL" --output-dir ".\my-videos"
poetry run .\scripts\windows\run_download_videos.ps1 --list-only
poetry run .\scripts\windows\run_download_videos.ps1 --debug
```

This will:
1. Open a browser window with the SharePoint site
2. Display all available files 
3. Provide instructions for manual downloading:
   - For individual files: right-click and select "Download"
   - For multiple files: select files using checkboxes and download as ZIP
4. Monitor your system's Downloads folder for:
   - Downloaded video files - moved automatically to the output directory
   - ZIP files - extracted automatically to get video files, then ZIP is deleted
5. Automatically complete when all downloads are processed (no need to stop manually)

The process will automatically proceed to the next step in the pipeline once downloads are complete.

### 2. Extract Speech from Videos

```powershell
# Interactive mode
poetry run .\scripts\windows\run_separate_speech.ps1

# With specific input and output directories
poetry run .\scripts\windows\run_separate_speech.ps1 --input-dir ".\my-videos" --output-dir ".\my-speech"

# Additional options
poetry run .\scripts\windows\run_separate_speech.ps1 --file-type wav  # Output format: wav, mp3, or both
poetry run .\scripts\windows\run_separate_speech.ps1 --model sepformer  # Separation model
```

### 3. Transcribe Speech to Text

```powershell
# Interactive mode
poetry run .\scripts\windows\run_speech_to_text.ps1

# With specific input and output directories
poetry run .\scripts\windows\run_speech_to_text.ps1 --input-dir ".\my-speech" --output-dir ".\my-transcripts"

# Additional options
poetry run .\scripts\windows\run_speech_to_text.ps1 --language fr  # Language (default: en)
poetry run .\scripts\windows\run_speech_to_text.ps1 --output-format txt  # Output format: srt, txt, or both
poetry run .\scripts\windows\run_speech_to_text.ps1 --model whisperx  # Transcription model
```

### 4. Analyze Emotions and Body Poses

```powershell
# Interactive mode
poetry run .\scripts\windows\run_emotion_recognition.ps1

# With specific input and output directories
poetry run .\scripts\windows\run_emotion_recognition.ps1 --input-dir ".\my-videos" --output-dir ".\my-emotions"

# Emotion recognition always includes pose estimation by default
# To disable pose estimation (not recommended)
poetry run .\scripts\windows\run_emotion_recognition.ps1 --no-pose
```

---

## Advanced Usage

### Running as Python Modules

Each component can be run directly as a Python module:

```powershell
# Download videos
poetry run python -m src.download_videos --url "https://sharepoint-url.com" --output-dir ".\my-videos"

# Speech separation
poetry run python -m src.separate_speech --input-dir ".\my-videos" --output-dir ".\my-speech"

# Speech to text
poetry run python -m src.speech_to_text --input-dir ".\my-speech" --output-dir ".\my-transcripts"

# Emotion recognition
poetry run python -m src.emotion_recognition.cli --input-dir ".\my-videos" --output-dir ".\my-emotions" --with-pose
```

### Troubleshooting

#### PowerShell Execution Policy

If you get an error related to script execution policy:

```powershell
# Run PowerShell as Administrator and execute
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Missing Dependencies

If you encounter missing dependency errors:

```powershell
# Update Poetry dependencies
poetry update

# Ensure all feature groups are installed
poetry install --with common --with speech --with emotion --with download
```

#### Path Issues

Windows paths use backslashes (`\`), but many Python libraries also accept forward slashes (`/`). If you encounter path issues:

```powershell
# Always use double backslashes in PowerShell strings
$path = "C:\\Users\\username\\Documents"

# Or use raw strings with single backslashes
$path = 'C:\Users\username\Documents'
```

---

## Additional Resources

- [Project Documentation](https://github.com/username/Video_Data_Processing/docs)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Speech Processing Resources](https://github.com/speechbrain/speechbrain)
