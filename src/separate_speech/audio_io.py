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
        # Validate input waveform
        if waveform is None or torch.isnan(waveform).any():
            logger.error("Invalid waveform data contains NaN values")
            return False
            
        # Check for extremely small or zero values
        if torch.max(torch.abs(waveform)) < 1e-6:
            logger.warning("Waveform has very low amplitude, might be silent")
        
        # Ensure correct shape for torchaudio
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        
        # Normalize audio to prevent clipping
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0.99:
            waveform = waveform * (0.95 / max_val)
            logger.info(f"Normalized audio to prevent clipping (max value: {max_val:.4f})")
            
        if file_type in ["wav", "both"]:
            # Ensure directory exists
            wav_path = str(output_path) + ".wav"
            ensure_directory_exists(wav_path)
            
            try:
                # Save as WAV using torchaudio
                torchaudio.save(wav_path, waveform.cpu(), sample_rate)
                
                # Verify the file was created and is valid
                if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100:
                    logger.error(f"WAV file was not created properly or is too small: {wav_path}")
                    return False
                    
                logger.info(f"Saved WAV audio to {wav_path} ({os.path.getsize(wav_path)/1024:.1f} KB)")
            except Exception as e:
                logger.error(f"Error saving WAV file: {e}")
                return False
            
        if file_type in ["mp3", "both"]:
            # Ensure directory exists for mp3 file
            mp3_path = str(output_path) + ".mp3"
            ensure_directory_exists(mp3_path)
            
            # Always save to a temporary WAV file first for reliable conversion
            try:
                temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav_path = temp_wav.name
                temp_wav.close()
                
                # Save to temporary WAV
                torchaudio.save(temp_wav_path, waveform.cpu(), sample_rate)
                
                # Verify the temp WAV file exists and is valid
                if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) < 100:
                    logger.error("Temporary WAV file is invalid or too small")
                    if os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)
                    return False
                
                # Try a more reliable MP3 conversion method using FFmpeg directly
                try:
                    # Use a more explicit FFmpeg command with careful settings
                    cmd = [
                        "ffmpeg", "-y",                      # Force overwrite
                        "-f", "wav",                         # Input format
                        "-i", temp_wav_path,                 # Input file
                        "-af", "aresample=resampler=soxr",   # Use high-quality resampler
                        "-ar", str(sample_rate),             # Output sample rate
                        "-ac", "1",                          # Mono output
                        "-b:a", "192k",                      # Bitrate
                        "-codec:a", "libmp3lame",            # MP3 encoder
                        "-q:a", "2",                         # Quality level (0-9, lower is better)
                        mp3_path                             # Output file
                    ]
                    
                    # Run FFmpeg with progress tracking
                    logger.info(f"Converting to MP3 with FFmpeg: {' '.join(cmd)}")
                    process = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    
                    # Check if MP3 file exists and has reasonable size
                    if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 1000:
                        logger.info(f"Saved MP3 audio to {mp3_path} ({os.path.getsize(mp3_path)/1024:.1f} KB)")
                    else:
                        logger.error(f"MP3 file is missing or too small: {mp3_path}")
                        return False
                        
                except subprocess.SubprocessError as e:
                    logger.error(f"FFmpeg conversion error: {e}")
                    
                    # Try pydub as fallback
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_wav(temp_wav_path)
                        audio_segment = audio_segment.set_frame_rate(sample_rate)
                        audio_segment = audio_segment.set_channels(1)  # Mono
                        audio_segment.export(mp3_path, format="mp3", bitrate="192k")
                        logger.info(f"Saved MP3 using Pydub fallback: {mp3_path}")
                    except Exception as pydub_error:
                        logger.error(f"Both FFmpeg and Pydub failed: {pydub_error}")
                        return False
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)
                        
            except Exception as e:
                logger.error(f"Error during MP3 conversion: {e}")
                # Try to clean up temp files
                if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
                return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
