# Video Data Processing

A comprehensive toolkit for processing video data, extracting speech, generating transcripts, and analyzing emotions and body poses.

## Key Features

- **SharePoint Integration**: Download videos directly from SharePoint
- **Speech Separation**: Extract clean speech from videos with background noise
- **Speech-to-Text Transcription**: Convert speech to accurate text with speaker identification
- **Emotion & Pose Recognition**: Analyze facial emotions and body poses in videos
- **Sequential Processing Pipeline**: Process videos through all steps automatically
- **Batch Processing**: Process multiple videos without manual intervention

## Getting Started

### Prerequisites

- Python 3.12
- Poetry (dependency management)
- FFmpeg (audio/video processing)
- Hugging Face account (for AI model access)

### Installation

```bash
# Clone the repository
git clone https://github.com/tkhahns/Video_Data_Processing.git
cd Video_Data_Processing

# Install dependencies
poetry install
poetry install --with common --with speech --with emotion --with download
```

### Quick Start

The simplest way to use this toolkit is through the all-in-one pipeline script:

```bash
# macOS/Linux
./run_all.sh

# Windows
.\run_all.ps1
```

This will:
1. Prompt for your Hugging Face token (used in-memory only, never saved to disk)
2. Guide you through downloading videos from SharePoint
3. Process the videos through all pipeline stages
4. Output results in timestamped directories

## Documentation

Detailed documentation for each platform:

- [macOS Instructions](docs/macos_instructions.md)
- [Windows Instructions](docs/windows_instructions.md)

## Pipeline Workflow

1. **Download Videos**: Download videos from SharePoint or use existing files
2. **Speech Separation**: Extract clean speech audio from videos
3. **Transcription**: Convert speech to text with speaker identification
4. **Emotion & Pose Recognition**: Analyze facial emotions and body language

## Components

Each component can be used individually:

```bash
# Speech separation
poetry run scripts/macos/run_separate_speech.sh --input-dir "./my-videos"

# Speech-to-text
poetry run scripts/macos/run_speech_to_text.sh --input-dir "./my-speech"

# Emotion and pose recognition
poetry run scripts/macos/run_emotion_and_pose_recognition.sh --input-dir "./my-videos"
```

## Batch Processing

For automated processing of multiple videos:

```bash
# Run the complete pipeline in batch mode
./run_all.sh --batch

# Run individual components in batch mode
poetry run scripts/macos/run_separate_speech.sh --input-dir "./my-videos" --batch
poetry run scripts/macos/run_speech_to_text.sh --input-dir "./my-speech" --batch
poetry run scripts/macos/run_emotion_and_pose_recognition.sh --input-dir "./my-videos" --batch
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
