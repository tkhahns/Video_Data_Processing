## Video Data Processing Setup Instructions

These instructions will help you set up the environment for the Video Data Processing application on macOS using Python 3.12.10 managed by pyenv.

---

## Prerequisites

Ensure you have **pyenv** installed and Python 3.12.10 set as your global interpreter:

```bash
# Install pyenv (if not already installed)
brew update
brew install pyenv

# Initialize pyenv in your shell (if not already configured)
# Add to your ~/.zprofile:
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

# Add to your ~/.zshrc:
eval "$(pyenv init -)"

# Install and set Python 3.12.10 globally
pyenv install --skip-existing 3.12.10
pyenv global 3.12.10

# Verify the Python version
python --version    # → Python 3.12.10
python3 --version   # → Python 3.12.10
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
# Create the virtual environment using the specified Python
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```  

### 3. Update pip and install dependencies

```bash
# Update pip to the latest version
pip install --upgrade pip

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
# Make the script executable
chmod +x run_download_models.sh

# Run the script
./run_download_models.sh
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

- For sentencepiece build errors:
  ```bash
  # Install without building isolation
  pip install --prefer-binary sentencepiece
  ```

- If you still have import errors, verify you're using the correct Python interpreter:
  ```bash
  which python
  # Should point to your virtual environment's .venv/bin/python
  ```

---

## Download Videos from SharePoint

You can download videos from SharePoint using the included browser automation tool:

### Option 1: Using the convenience script (recommended)

```bash
# Make the script executable (first time only)
chmod +x run_download_videos.sh

# Run the script with no arguments (you'll be prompted for the URL)
./run_download_videos.sh

# Or specify the URL directly
./run_download_videos.sh --url "https://your-sharepoint-site.com/folder-with-videos"

# Additional options:
./run_download_videos.sh --url "https://your-sharepoint-site.com/folder-with-videos" --output-dir "./my-videos"
./run_download_videos.sh --list-only
./run_download_videos.sh --debug
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

This will:
1. Open a browser window for SharePoint authentication
2. Find all available files in the specified folder
3. Present you with a list of files to choose from
4. Download your selected files

**Note:** The tool requires authentication to SharePoint. You'll need to sign in through the browser window that opens.

---

## Extract Speech from Videos

You can extract and separate speech from video files using the provided speech separation tool:

### Option 1: Using the convenience script (recommended)

```bash
# Make the script executable (first time only)
chmod +x run_separate_speech.sh

# Run the script with no arguments (interactive mode)
./run_separate_speech.sh

# Or process specific video files
./run_separate_speech.sh path/to/video.mp4

# Additional options:
./run_separate_speech.sh --output-dir "./my-speech-output" path/to/video.mp4
./run_separate_speech.sh --file-type wav  # Choose output format: wav, mp3, or both
./run_separate_speech.sh --model sepformer
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
python -m src.separate_speech path/to/video.mp4

# Additional options
python -m src.separate_speech path/to/video.mp4 --output-dir "./my-speech-output"
python -m src.separate_speech path/to/video.mp4 --file-type wav
```

### Option 3: Running the Python script directly

```bash
# Basic usage
python src/separate_speech/__main__.py path/to/video.mp4

# Advanced options
python src/separate_speech/__main__.py --output-dir "./my-speech-output" --file-type both path/to/video.mp4
python src/separate_speech/__main__.py --model sepformer --chunk-size 5 path/to/video.mp4
```

The tool will:
1. Extract audio from the video files
2. Process through speech separation model
3. Save the isolated speech as audio files (WAV and/or MP3)

**Note:** The first run will download the speech separation model (approximately 1GB), which may take some time depending on your internet connection.

---

## Install ffmpeg

ffmpeg is required for audio processing and conversion:

```bash
# Install ffmpeg using Homebrew
brew install ffmpeg
```

You can also install the Python wrapper for ffmpeg:

```bash
pip install ffmpeg-python pydub
```

This will:
1. Install the ffmpeg command-line tools
2. Install Python wrappers for easier integration

---

## Transcribe Speech to Text

You can transcribe speech audio files to text using the provided speech-to-text transcription tool:

### Option 1: Using the convenience script (recommended)

```bash
# Make the script executable (first time only)
chmod +x run_speech_to_text.sh

# Run the script with no arguments (interactive mode)
./run_speech_to_text.sh

# Or process specific audio files
./run_speech_to_text.sh path/to/audio.wav

# Additional options:
./run_speech_to_text.sh --output-dir "./my-transcripts" path/to/audio.mp3
./run_speech_to_text.sh --language fr  # Specify language (default: en)
./run_speech_to_text.sh --model whisperx  # Choose model (whisperx, xlsr)
./run_speech_to_text.sh --select  # Force file selection even with files specified
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
python -m src.speech_to_text path/to/audio.mp3

# Additional options
python -m src.speech_to_text path/to/audio.wav --output-dir "./my-transcripts"
python -m src.speech_to_text path/to/audio.mp3 --language es
python -m src.speech_to_text --output-format txt  # Options: srt, txt, both
```

### Option 3: Running the Python script directly

```bash
# Basic usage
python src/speech_to_text/__main__.py path/to/audio.mp3

# Advanced options
python src/speech_to_text/__main__.py --output-dir "./my-transcripts" --language fr path/to/audio.mp3
python src/speech_to_text/__main__.py --model xlsr --recursive path/to/audio/folder
python src/speech_to_text/__main__.py --output-format both  # Save as both SRT and TXT
```

The tool will:
1. Process the audio files through the selected speech recognition model
2. Create timestamped transcriptions of the spoken content
3. Save the results as SRT subtitles (.srt), plain text with timestamps (.txt), or both formats

**Note:** The first run will download the speech recognition model, which may take some time depending on your internet connection. If using a CPU-only system, the tool will automatically fall back to float32 precision for better compatibility.