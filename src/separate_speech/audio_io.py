"""
Audio I/O operations for speech separation.
"""
import os
import logging
import subprocess
import torch
import torchaudio
import tempfile
import numpy as np
from pathlib import Path
import tqdm
from .utils import check_ffmpeg_dependencies

# Try importing from utils package
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)

def ensure_directory_exists(filepath):
    """
    Create directory for the given filepath if it doesn't exist.
    
    Args:
        filepath: Path to the file (including filename)
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def convert_wav_to_mp3(wav_path, mp3_path):
    """Convert WAV file to MP3 format using pydub or ffmpeg."""
    if not check_ffmpeg_dependencies():
        logger.error("MP3 conversion skipped due to missing ffmpeg/ffprobe.")
        return False

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="192k")
        return True
    except ImportError:
        logger.warning("Pydub not installed, trying ffmpeg directly...")
        try:
            # Try using ffmpeg directly
            cmd = ["ffmpeg", "-y", "-i", wav_path, "-b:a", "192k", mp3_path]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"Error converting to MP3: {e}")
            return False

def save_audio(waveform, sample_rate, output_path, file_type="mp3"):
    """
    Save audio waveform to file.
    
    Args:
        waveform: Audio waveform tensor [channels, samples]
        sample_rate: Sample rate of the audio
        output_path: Path to save the audio (without extension)
        file_type: Output format ("wav", "mp3", or "both")
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure correct shape for torchaudio
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
            
        if file_type in ["wav", "both"]:
            # Ensure directory exists
            wav_path = str(output_path) + ".wav"
            ensure_directory_exists(wav_path)
            
            # Save as WAV using torchaudio
            torchaudio.save(wav_path, waveform.cpu(), sample_rate)
            logger.info(f"Saved WAV audio to {wav_path}")
            
        if file_type in ["mp3", "both"]:
            # Ensure directory exists for mp3 file
            mp3_path = str(output_path) + ".mp3"
            ensure_directory_exists(mp3_path)
            
            # Try to save as MP3 using various methods
            try:
                # Method 1: Try using pydub
                # Save temporary WAV file first
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav.close()
                torchaudio.save(temp_wav.name, waveform.cpu(), sample_rate)
                
                # Convert to MP3
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_wav(temp_wav.name)
                audio_segment.export(mp3_path, format="mp3", bitrate="192k")
                
                # Clean up temp file
                os.unlink(temp_wav.name)
                logger.info(f"Saved MP3 audio to {mp3_path}")
                
            except ImportError:
                # Method 2: Try direct ffmpeg conversion
                try:
                    # Save temporary WAV file first
                    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_wav.close()
                    torchaudio.save(temp_wav.name, waveform.cpu(), sample_rate)
                    
                    # Convert to MP3 using ffmpeg
                    subprocess.run([
                        "ffmpeg", "-y", "-i", temp_wav.name, 
                        "-codec:a", "libmp3lame", "-qscale:a", "2",
                        mp3_path
                    ], check=True, capture_output=True)
                    
                    # Clean up temp file
                    os.unlink(temp_wav.name)
                    logger.info(f"Saved MP3 audio to {mp3_path} using ffmpeg")
                    
                except Exception as e:
                    logger.error(f"Failed to convert to MP3: {e}")
                    return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        return False
