"""
Audio extraction from video files.
"""
import os
import tempfile
import logging
import torch
import torchaudio
from moviepy import VideoFileClip
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_audio_from_video(video_path, output_path=None, sample_rate=16000):
    """Extract audio from video file."""
    logger.info(f"Extracting audio from {video_path}")
    
    if output_path is None:
        # Create a temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    try:
        # Using MoviePy to extract audio
        video = VideoFileClip(video_path)
        # Fix for MoviePy version compatibility
        if hasattr(video.audio, 'write_audiofile'):
            if 'verbose' in video.audio.write_audiofile.__code__.co_varnames:
                video.audio.write_audiofile(output_path, fps=sample_rate, 
                                          nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
            else:
                # For newer versions that don't have verbose parameter
                video.audio.write_audiofile(output_path, fps=sample_rate, 
                                          nbytes=2, codec='pcm_s16le', logger=None)
        else:
            logger.error("Video has no audio track")
            return None, None, None
        
        logger.info(f"Audio extracted to {output_path}")
        
        # Load the extracted audio for processing
        waveform, sr = torchaudio.load(output_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
            
        return waveform, sample_rate, output_path
    
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return None, None, None

def find_video_files(input_paths=None, recursive=True, default_dir=None):
    """Find video files in the specified paths or default directory."""
    video_files = []
    
    # If no input paths provided, use data directory
    if not input_paths:
        default_dir = default_dir or Path('data')
        input_paths = [str(default_dir)]
    
    # Process each input path
    for input_path in input_paths:
        if os.path.isfile(input_path):
            # Single file
            if input_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_files.append(input_path)
            else:
                logger.warning(f"Skipping non-video file: {input_path}")
        elif os.path.isdir(input_path):
            # Directory
            if recursive:
                for root, _, files in os.walk(input_path):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                            video_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(input_path):
                    file_path = os.path.join(input_path, file)
                    if os.path.isfile(file_path) and file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                        video_files.append(file_path)
    
    return video_files
