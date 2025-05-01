"""
Audio input/output functionality.
"""
import os
import logging
import subprocess
import torch
import torchaudio
from .utils import check_ffmpeg_dependencies

logger = logging.getLogger(__name__)

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
    """Save audio waveform to file.
    
    Args:
        waveform: The audio data to save
        sample_rate: Sample rate of the audio
        output_path: Path to save the audio without extension
        file_type: Type of file to save - "wav", "mp3", or both ("both")
    """
    try:
        # Ensure output path has no extension
        if output_path.lower().endswith('.wav') or output_path.lower().endswith('.mp3'):
            output_path = os.path.splitext(output_path)[0]
        
        # Define paths
        wav_path = output_path + '.wav'
        mp3_path = output_path + '.mp3'
        
        # Ensure waveform has the right shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Check audio statistics before saving
        audio_max = waveform.abs().max().item()
        audio_mean = waveform.abs().mean().item()
        logger.info(f"Audio statistics: max_amplitude={audio_max:.6f}, mean_amplitude={audio_mean:.6f}")
        
        if audio_max < 0.01:
            logger.warning("WARNING: Audio amplitude is very low, output may be inaudible!")
        
        # Always save as WAV (it's needed for MP3 conversion anyway)
        logger.info(f"Saving WAV file: {wav_path}")
        torchaudio.save(wav_path, waveform, sample_rate)
        
        # Convert to MP3 if requested
        if file_type.lower() in ["mp3", "both"]:
            logger.info(f"Converting to MP3: {mp3_path}")
            success = convert_wav_to_mp3(wav_path, mp3_path)
            
            if success:
                logger.info(f"Successfully saved MP3: {mp3_path}")
                # Remove WAV file if not keeping both
                if file_type.lower() != "both" and file_type.lower() != "wav":
                    os.remove(wav_path)
                    logger.info("Removed temporary WAV file")
            else:
                logger.warning(f"MP3 conversion failed, keeping WAV file: {wav_path}")
        
        # Log which files were kept
        if file_type.lower() == "wav" or file_type.lower() == "both" or (file_type.lower() == "mp3" and not success):
            logger.info(f"Saved WAV file: {wav_path}")
            
        return True
            
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        return False
