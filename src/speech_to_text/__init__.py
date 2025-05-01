"""
Speech-to-text transcription package.
Provides tools for transcribing speech audio files to text.
"""

__version__ = "1.0.0"

from pathlib import Path

# Default paths
DEFAULT_MODELS_DIR = Path("./models/downloaded")
DEFAULT_OUTPUT_DIR = Path("./output/transcripts")
DEFAULT_AUDIO_DIR = Path("./output/separated_speech")  # Updated to use separated speech output

# Default settings
DEFAULT_MODEL = "whisperx"
DEFAULT_LANGUAGE = "en"
DEFAULT_SEGMENT_SIZE = 30  # Default segment size in seconds
SUPPORTED_MODELS = ["whisperx", "xlsr", "s2t-fairseq", "whisper"]
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac"]
