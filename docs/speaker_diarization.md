# Speaker Diarization Guide

Speaker diarization identifies different speakers in audio and labels them in transcripts. This guide explains how to set up and use this feature.

## Quick Setup (Recommended)

Run our setup script to download and validate all required models:

```bash
# On macOS/Linux
./scripts/download_diarization_models.sh

# On Windows PowerShell
.\scripts\download_diarization_models.ps1
```

The script will:
1. Install required dependencies
2. Guide you through Hugging Face login
3. Download and validate the diarization models
4. Verify that you have accepted the necessary licenses

## Manual Setup Process

If you prefer to set up manually or are troubleshooting issues:

### 1. Login to Hugging Face

```bash
# Login to Hugging Face
huggingface-cli login
```

### 2. Accept Required Model Licenses

You must accept both of these model licenses:

1. Visit [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) and click "Accept license"
2. Visit [pyannote/segmentation](https://huggingface.co/pyannote/segmentation) and click "Accept license"

### 3. Verify Downloads

After accepting the licenses, download the models:

```bash
python -c "from huggingface_hub import snapshot_download, HfFolder; token = HfFolder.get_token(); snapshot_download('pyannote/speaker-diarization', token=token); snapshot_download('pyannote/segmentation', token=token)"
```

## Troubleshooting

If you see "License acceptance required" errors:

1. **Are you logged in correctly?** Run `huggingface-cli whoami` to check
2. **Have you accepted BOTH model licenses?** You need both models
3. **Is your token valid?** Try logging in again with `huggingface-cli login`
4. **Wait a few minutes** after accepting licenses - changes may take time to propagate
5. **Clear cache:** Remove `~/.cache/huggingface` directory and re-download models

## Using Speaker Diarization

Speaker diarization is enabled by default:

```bash
# macOS/Linux
./scripts/macos/run_speech_to_text.sh --input-dir "./my-audio"

# Windows
.\scripts\windows\run_speech_to_text.ps1 --input-dir ".\my-audio"
```

To disable it:

```bash
# Add the --no-diarize flag
./scripts/macos/run_speech_to_text.sh --no-diarize
```
