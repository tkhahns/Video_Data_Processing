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

### Hugging Face Authentication

The pipeline requires a Hugging Face token for accessing AI models:

1. Create a free account at [Hugging Face](https://huggingface.co/join)
2. Generate a token at https://huggingface.co/settings/tokens
3. Keep your token handy to enter when prompted by the pipeline
4. **Important**: For security, tokens are never saved to disk and are only used during the current session

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
1. Prompts for your Hugging Face token (one-time, not saved)
2. Downloads videos from SharePoint
3. Runs emotion and pose recognition in parallel with speech processing for better performance
4. Extracts speech from the videos
5. Transcribes the speech to text
6. Reports the total processing time and results

All outputs are organized in timestamped directories for easy tracking.

### Running the Complete Pipeline

```powershell
# Run the pipeline (will prompt for Hugging Face token and SharePoint URL)
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

### 2. Extract Speech from Videos

```powershell
# Interactive mode
poetry run .\scripts\windows\run_separate_speech.ps1

# With specific input and output directories
poetry run .\scripts\windows\run_separate_speech.ps1 --input-dir ".\my-videos" --output-dir ".\my-speech"

# Batch mode (process all files without manual selection)
poetry run .\scripts\windows\run_separate_speech.ps1 --input-dir ".\my-videos" --output-dir ".\my-speech" --batch

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

# Batch mode
poetry run .\scripts\windows\run_speech_to_text.ps1 --input-dir ".\my-speech" --output-dir ".\my-transcripts" --batch

# With speaker diarization (enabled by default)
poetry run .\scripts\windows\run_speech_to_text.ps1 --input-dir ".\my-speech" --output-dir ".\my-transcripts" --diarize
```

### 4. Analyze Emotions and Body Poses

```powershell
# Interactive mode (will prompt for Hugging Face token)
poetry run .\scripts\windows\run_emotion_and_pose_recognition.ps1

# With specific input and output directories
poetry run .\scripts\windows\run_emotion_and_pose_recognition.ps1 --input-dir ".\my-videos" --output-dir ".\my-emotions"

# Batch mode
poetry run .\scripts\windows\run_emotion_and_pose_recognition.ps1 --input-dir ".\my-videos" --output-dir ".\my-emotions" --batch
```

---

## Advanced Usage

### Batch Processing

For automated processing without manual file selection:

```powershell
# Run entire pipeline in batch mode
.\run_all.ps1 --url "YOUR_URL" --batch

# Run individual components in batch mode
poetry run .\scripts\windows\run_separate_speech.ps1 --input-dir ".\my-videos" --batch
poetry run .\scripts\windows\run_speech_to_text.ps1 --input-dir ".\my-speech" --batch
poetry run .\scripts\windows\run_emotion_and_pose_recognition.ps1 --input-dir ".\my-videos" --batch
```

### Running as Python Modules

Each component can be run directly as a Python module:

```powershell
# Download videos
poetry run python -m src.download_videos --url "https://sharepoint-url.com" --output-dir ".\my-videos"

# Speech separation
poetry run python -m src.separate_speech --input-dir ".\my-videos" --output-dir ".\my-speech"

# Speech to text
poetry run python -m src.speech_to_text --input-dir ".\my-speech" --output-dir ".\my-transcripts"

# Emotion and pose recognition
poetry run python -m src.emotion_and_pose_recognition.cli --input-dir ".\my-videos" --output-dir ".\my-emotions" --with-pose
```

---

## Troubleshooting

### PowerShell Execution Policy

If you get an error related to script execution policy:

```powershell
# Run PowerShell as Administrator and execute
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Hugging Face Token Issues

If you encounter issues with Hugging Face authentication:

1. Ensure you have a valid token from https://huggingface.co/settings/tokens
2. The token is only used for the current session and is never saved to disk
3. If you need to use the token in multiple terminal sessions, you'll need to enter it each time

### Missing Dependencies

If you encounter missing dependency errors:

```powershell
# Update Poetry dependencies
poetry update

# Ensure all feature groups are installed
poetry install --with common --with speech --with emotion --with download
```

### Path Issues

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
- [Hugging Face](https://huggingface.co/) - AI model repository
