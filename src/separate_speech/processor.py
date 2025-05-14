"""
Video processing and speech separation workflow.
"""
import os
import logging
import tqdm
import torch
import sys
from pathlib import Path
from . import separation
from . import extraction
from . import audio_io
from .utils import clean_memory, ensure_dir_exists
import tempfile

# Try importing from utils package
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)

def process_file(video_path, output_dir, model_name, models_dir, chunk_size_sec=10, file_type="mp3", skip_no_speech=False, min_speech_seconds=1.0):
    """
    Process a single video file for speech separation.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output files
        model_name: Name of the speech separation model to use
        models_dir: Directory containing downloaded models
        chunk_size_sec: Size of audio chunks to process in seconds
        file_type: Output format ("wav", "mp3", "both")
        skip_no_speech: If True, skip files where no speech is detected
        min_speech_seconds: Minimum seconds of speech needed to process file
        
    Returns:
        Tuple of (success, has_speech)
        - success: True if processing completed successfully
        - has_speech: True if speech was detected in the file
    """
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_path = os.path.join(output_dir, f"{video_name}_speech")  # No extension, added later
    
    # Ensure the output directory exists
    ensure_dir_exists(output_dir)
    
    # Extract audio from video
    logger.info(f"Processing {video_filename}")
    with tqdm.tqdm(total=4, desc=f"Processing {video_filename}", unit="step") as pbar:
        pbar.set_description("Extracting audio")
        waveform, sample_rate, temp_audio_path = extraction.extract_audio_from_video(video_path)
        if waveform is None:
            return False, False
        pbar.update(1)
        
        # Get audio file size and log
        audio_duration = waveform.shape[1] / sample_rate
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Load separation model
        pbar.set_description("Loading speech separation model")
        model = separation.load_speech_separation_model(model_name, models_dir)
        if model is None:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False, False
        pbar.update(1)
        
        # Separate speech using chunked processing
        pbar.set_description("Separating speech")
        separated_speech = separation.separate_speech_chunked(waveform, model, sample_rate, chunk_size_sec)
        
        # Clean up model to free memory
        del model
        clean_memory()
        
        if separated_speech is None:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False, False
        pbar.update(1)
        
        # Always save the full separated speech file
        pbar.set_description(f"Saving speech audio")
        audio_io.save_audio(separated_speech, sample_rate, output_path, file_type)
        
        # Clean up temp file from initial extraction
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        pbar.update(1)
        
        # Report success and speech detected
        return True, True
