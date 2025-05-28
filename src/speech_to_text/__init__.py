"""
Speech-to-text transcription package.
"""

from pathlib import Path

# Define default paths
DEFAULT_MODELS_DIR = Path("./resources/models")
DEFAULT_OUTPUT_DIR = Path("./output/transcripts")
DEFAULT_AUDIO_DIR = Path("./output/separated_speech")

# Define defaults for transcription
DEFAULT_MODEL = "whisperx"
DEFAULT_LANGUAGE = "en"
DEFAULT_SEGMENT_SIZE = 30  # in seconds

# Define supported models
SUPPORTED_MODELS = ["whisperx", "whisper", "xlsr", "s2t-fairseq"]

# Define supported audio formats for better format compatibility
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']

# Import main classes/functions for easier access
from .main import main
from .transcription import transcribe_audio
from .speech_features import find_audio_files, extract_audio_features
