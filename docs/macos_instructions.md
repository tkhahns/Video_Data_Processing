# Video Data Processing - macOS Setup Instructions

These instructions will help you set up the environment for the Video Data Processing application on macOS.

---

## Prerequisites

### Python 3.12 (Required)

This project requires Python 3.12 specifically:

```bash
# Check Python version
python3 --version

# Install Python 3.12 using Homebrew
brew update
brew install python@3.12

# Make Python 3.12 the default
echo 'alias python=python3.12' >> ~/.zshrc
echo 'alias python3=python3.12' >> ~/.zshrc
source ~/.zshrc

# Verify Python version
python --version  # Should show Python 3.12.x
```

### Poetry for Dependency Management

This project uses Poetry for dependency management:

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to your PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify Poetry installation
poetry --version

# Configure Poetry to use Python 3.12
poetry env use python3.12
```

### FFmpeg

FFmpeg is required for audio and video processing:

```bash
# Install using Homebrew
brew install ffmpeg
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/tkhahns/Video_Data_Processing.git
cd Video_Data_Processing
```

### 2. Install dependencies with Poetry

```bash
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

### 3. Set up execution permissions

```bash
# Make all scripts executable
chmod +x run_all.sh
chmod +x scripts/macos/run_download_videos.sh
chmod +x scripts/macos/run_separate_speech.sh
chmod +x scripts/macos/run_speech_to_text.sh
chmod +x scripts/macos/run_emotion_recognition.sh
```

---

## Complete Processing Pipeline

The `run_all.sh` script provides a complete end-to-end pipeline that:
1. Downloads videos from SharePoint
2. Extracts speech from the videos
3. Transcribes the speech to text
4. Analyzes emotions and body poses in the videos

All outputs are organized in timestamped directories for easy tracking.

### Running the Complete Pipeline

```bash
# Run the pipeline (will prompt for SharePoint URL)
./run_all.sh

# Or specify the SharePoint URL directly
./run_all.sh --url "https://your-sharepoint-site.com/folder-with-videos"
```

### Pipeline Outputs

The pipeline creates the following directory structure:
- `data/downloads_TIMESTAMP/`: Original downloaded videos
- `output/pipeline_results_TIMESTAMP/`: Processing results
  - `speech/`: Extracted speech audio files
  - `transcripts/`: Text transcription files
  - `emotions_and_pose/`: Emotion and body pose analysis

---

## Individual Components

Each component can also be run separately if you need to process only specific steps.

### 1. Download Videos from SharePoint

```bash
# Run with Poetry (interactive mode)
poetry run scripts/macos/run_download_videos.sh

# Specify URL directly
poetry run scripts/macos/run_download_videos.sh --url "https://your-sharepoint-site.com/folder-with-videos"

# Additional options
poetry run scripts/macos/run_download_videos.sh --url "URL" --output-dir "./my-videos"
poetry run scripts/macos/run_download_videos.sh --list-only
poetry run scripts/macos/run_download_videos.sh --debug
```

### 2. Extract Speech from Videos

```bash
# Interactive mode
poetry run scripts/macos/run_separate_speech.sh

# With specific input and output directories
poetry run scripts/macos/run_separate_speech.sh --input-dir "./my-videos" --output-dir "./my-speech"

# Additional options
poetry run scripts/macos/run_separate_speech.sh --file-type wav  # Output format: wav, mp3, or both
poetry run scripts/macos/run_separate_speech.sh --model sepformer  # Separation model
```

### 3. Transcribe Speech to Text

```bash
# Interactive mode
poetry run scripts/macos/run_speech_to_text.sh

# With specific input and output directories
poetry run scripts/macos/run_speech_to_text.sh --input-dir "./my-speech" --output-dir "./my-transcripts"

# Additional options
poetry run scripts/macos/run_speech_to_text.sh --language fr  # Language (default: en)
poetry run scripts/macos/run_speech_to_text.sh --output-format txt  # Output format: srt, txt, or both
poetry run scripts/macos/run_speech_to_text.sh --model whisperx  # Transcription model
```

### 4. Analyze Emotions and Body Poses

```bash
# Interactive mode
poetry run scripts/macos/run_emotion_recognition.sh

# With specific input and output directories
poetry run scripts/macos/run_emotion_recognition.sh --input-dir "./my-videos" --output-dir "./my-emotions"

# Emotion recognition always includes pose estimation by default
# To disable pose estimation (not recommended)
poetry run scripts/macos/run_emotion_recognition.sh --no-pose
```

---

## Advanced Usage

### Running as Python Modules

Each component can be run directly as a Python module:

```bash
# Download videos
poetry run python -m src.download_videos --url "https://sharepoint-url.com" --output-dir "./my-videos"

# Speech separation
poetry run python -m src.separate_speech --input-dir "./my-videos" --output-dir "./my-speech"

# Speech to text
poetry run python -m src.speech_to_text --input-dir "./my-speech" --output-dir "./my-transcripts"

# Emotion recognition
poetry run python -m src.emotion_recognition.cli --input-dir "./my-videos" --output-dir "./my-emotions" --with-pose
```

### Customizing the Pipeline

The complete pipeline creates a unique timestamped directory for each run, making it easy to track different processing sessions. If you need to customize the pipeline:

1. Edit the `run_all.sh` script to modify the directory structure or processing steps
2. Use environment variables to configure specific components
3. Create your own pipeline script based on the existing ones

---

## Troubleshooting

### Missing Dependencies

If you encounter missing dependency errors:

```bash
# Update Poetry dependencies
poetry update

# Ensure all feature groups are installed
poetry install --with common --with speech --with emotion --with download
```

### SharePoint Download Issues

For SharePoint download problems:

1. Check that the URL points directly to a folder with videos
2. Examine the `sharepoint_page.png` screenshot created during download attempts
3. Try running with `--debug` for more detailed logging
4. Ensure your SharePoint credentials are correct

### Audio Processing Issues

If speech separation or transcription fails:

1. Verify that FFmpeg is properly installed: `ffmpeg -version`
2. Check that the input video files have valid audio tracks
3. Try processing a different video file to isolate the issue

### Emotion Recognition Issues

For emotion detection problems:

1. Ensure TensorFlow is properly installed in the Poetry environment
2. Check that the video files have visible faces for emotion detection
3. Try running with a smaller video file first to validate the setup

---

## Additional Resources

- [Project Documentation](https://github.com/username/Video_Data_Processing/docs)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Speech Processing Resources](https://github.com/speechbrain/speechbrain)