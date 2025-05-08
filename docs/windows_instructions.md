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

## Install ffmpeg

ffmpeg is required for audio processing and conversion. There are two ways to install it:

### Option 1: Using Chocolatey (recommended if you have Chocolatey installed)

```bash
# Install Chocolatey first if you don't have it
# Run this in an administrator PowerShell
# Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Then install ffmpeg
choco install ffmpeg
```

### Option 2: Manual installation

1. Download the latest static build from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows)
2. Extract the ZIP file to a folder (e.g., `C:\ffmpeg`)
3. Add ffmpeg to your PATH:
   - Search for "Environment Variables" in Windows search
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add the path to the bin folder (e.g., `C:\ffmpeg\bin`)
   - Click OK on all windows

### Option 3: Install Python wrapper only

You can also install just the Python wrapper, which will try to use ffmpeg if it's available:

```bash
pip install ffmpeg-python pydub
```

To verify ffmpeg is installed correctly:

```bash
ffmpeg -version
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

---

## Extract Speech from Videos

You can extract and separate speech from video files using the provided speech separation tool:

### Option 1: Using the convenience script (recommended)

```bash
# Run the script with no arguments (interactive mode)
.\run_separate_speech.ps1

# Or process specific video files
.\run_separate_speech.ps1 path\to\video.mp4

# Additional options:
.\run_separate_speech.ps1 --output-dir "./my-speech-output" path\to\video.mp4
.\run_separate_speech.ps1 --file-type wav  # Choose output format: wav, mp3, or both
.\run_separate_speech.ps1 --model sepformer
.\run_separate_speech.ps1 --detect-dialogues  # Enable dialogue detection
```

This script automatically:
- Activates the virtual environment
- Handles dependencies
- Processes videos through the speech separation model

### Option 2: Running as a module

```bash
# Using Python module syntax (interactive mode)
python -m src.separate_speech --interactive

# Process specific video files
python -m src.separate_speech path\to\video.mp4

# Additional options
python -m src.separate_speech path\to\video.mp4 --output-dir "./my-speech-output"
python -m src.separate_speech path\to\video.mp4 --file-type wav
python -m src.separate_speech path\to\video.mp4 --detect-dialogues  # Enable dialogue detection
python -m src.separate_speech path\to\video.mp4 --skip-no-speech  # Skip files without speech
```

### Option 3: Running the Python script directly

```bash
# Basic usage
python src/separate_speech/__main__.py path\to\video.mp4

# Advanced options
python src/separate_speech/__main__.py --output-dir "./my-speech-output" --file-type both path\to\video.mp4
python src/separate_speech/__main__.py --model sepformer --chunk-size 5 path\to\video.mp4
python src/separate_speech/__main__.py --detect-dialogues path\to\video.mp4  # Enable dialogue detection
```

The tool will:
1. Extract audio from the video files
2. Process through speech separation model
3. Save the isolated speech as audio files (WAV and/or MP3)
4. Optionally detect and extract dialogues from different speakers (with --detect-dialogues)

**Note:** 
- The first run will download the speech separation model (approximately 1GB), which may take some time depending on your internet connection.
- For dialogue detection, SpeechBrain will be installed automatically. This feature identifies different speakers and saves their speech as separate audio files.

### Dialogue Detection Prerequisites

To use dialogue detection, you need:

```bash
# Install SpeechBrain and associated dependencies
pip install speechbrain scikit-learn
```

If you encounter issues with dialogue detection:
- Ensure the output directory has write permissions
- For troubleshooting specific errors, check the log output

---

## Transcribe Speech to Text

You can transcribe speech audio files to text using the provided speech-to-text transcription tool:

### Option 1: Using the convenience script (recommended)

```bash
# Run the script with no arguments (interactive mode)
.\run_speech_to_text.ps1

# Or process specific audio files
.\run_speech_to_text.ps1 path\to\audio.wav

# Additional options:
.\run_speech_to_text.ps1 --output-dir "./my-transcripts" path\to\audio.mp3
.\run_speech_to_text.ps1 --language fr  # Specify language (default: en)
.\run_speech_to_text.ps1 --model whisperx  # Choose model (whisperx, xlsr)
.\run_speech_to_text.ps1 --select  # Force file selection even with files specified
```

This script automatically:
- Activates the virtual environment
- Installs required dependencies
- Processes audio files through the speech-to-text model

When run in interactive mode, the tool will:
1. Display a list of available audio files
2. Allow you to select specific files by number (e.g., "1,3,5") or choose "all"
3. Prompt you to choose an output format (SRT subtitles, TXT with timestamps, or both)
4. Process the selected files with the chosen settings

### Option 2: Running as a module

```bash
# Using Python module syntax (interactive mode)
python -m src.speech_to_text --interactive

# Process specific audio files
python -m src.speech_to_text path\to\audio.mp3

# Additional options
python -m src.speech_to_text path\to\audio.wav --output-dir "./my-transcripts"
python -m src.speech_to_text path\to\audio.mp3 --language es
python -m src.speech_to_text --output-format txt  # Options: srt, txt, both
```

### Option 3: Running the Python script directly

```bash
# Basic usage
python src/speech_to_text/__main__.py path\to\audio.mp3

# Advanced options
python src/speech_to_text/__main__.py --output-dir "./my-transcripts" --language fr path\to\audio.mp3
python src/speech_to_text/__main__.py --model xlsr --recursive path\to\audio\folder
python src/speech_to_text/__main__.py --output-format both  # Save as both SRT and TXT
```

The tool will:
1. Process the audio files through the selected speech recognition model
2. Create timestamped transcriptions of the spoken content
3. Save the results as SRT subtitles (.srt), plain text with timestamps (.txt), or both formats

**Note:** The first run will download the speech recognition model, which may take some time depending on your internet connection. If using a CPU-only system, the tool will automatically fall back to float32 precision for better compatibility.

---

## Detect Emotions and Body Poses in Videos

You can detect and analyze facial emotions and body poses in videos using the provided emotion recognition tool:

### Option 1: Using the convenience script (recommended)

```bash
# Run the script with no arguments (interactive mode)
.\run_emotion_recognition.ps1

# Or process specific video files
.\run_emotion_recognition.ps1 process path\to\video.mp4

# Additional options:
.\run_emotion_recognition.ps1 process path\to\video.mp4 --output path\to\output.mp4
.\run_emotion_recognition.ps1 batch input\directory output\directory
.\run_emotion_recognition.ps1 interactive --input_dir path\to\videos
.\run_emotion_recognition.ps1 check
.\run_emotion_recognition.ps1 --no-pose  # Disable body pose estimation
```

This script automatically:
- Activates the virtual environment
- Installs required dependencies (DeepFace, TensorFlow, MediaPipe, etc.)
- Creates the default directories if they don't exist
- Processes videos through the emotion recognition and pose estimation models

When run in interactive mode, the tool will:
1. Display a list of available video files (showing only filenames for clarity)
2. Allow you to select specific files by number (e.g., "1,3,5") or choose "all"
3. Prompt you to choose an output format (annotated video + log, or log only)
4. Process the selected files with the chosen settings

### Option 2: Running as a module

```bash
# Using Python module syntax (interactive mode)
python -m src.emotion_recognition.cli

# Process a single video file
python -m src.emotion_recognition.cli process path\to\video.mp4

# Disable body pose estimation
python -m src.emotion_recognition.cli --no-pose process path\to\video.mp4

# Additional options
python -m src.emotion_recognition.cli process path\to\video.mp4 --output path\to\output.mp4
python -m src.emotion_recognition.cli batch input\directory output\directory
python -m src.emotion_recognition.cli check
```

The tool will:
1. Detect faces in each frame of the video
2. Analyze emotions (happy, sad, angry, etc.) for each detected face
3. Detect and track body poses using MediaPipe
4. Generate an annotated video with emotion labels and pose landmarks
5. Create a CSV log file with emotion and pose details by timestamp
6. Create a JSON file with detailed pose data including joint positions and angles

**Note:** 
- Default input directory: `data\videos\`
- Default output directory: `output\emotions\`
- Body pose estimation is enabled by default and can be disabled with the `--no-pose` flag
- The tool uses DeepFace for emotion recognition and MediaPipe for pose estimation