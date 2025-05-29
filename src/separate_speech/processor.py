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

def process_file(video_path, output_dir, model_name, models_dir, chunk_size_sec=10, file_type="wav", skip_no_speech=False, min_speech_seconds=1.0):
    """Process a single video file for speech separation."""
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    # Remove spaces and special characters from output filename
    safe_video_name = video_name.replace(" ", "_").replace("(", "").replace(")", "")
    output_path = os.path.join(output_dir, f"{safe_video_name}_speech")  # No extension, added later
    
    # Ensure the output directory exists
    ensure_dir_exists(output_dir)
    
    # Extract audio from video
    logger.info(f"Processing {video_filename}")
    with tqdm.tqdm(total=4, desc=f"Processing {video_filename}", unit="step") as pbar:
        pbar.set_description("Extracting audio")
        waveform, sample_rate, temp_audio_path = extraction.extract_audio_from_video(video_path)
        if waveform is None:
            logger.error(f"Failed to extract audio from {video_filename}")
            return False, False
        pbar.update(1)
        
        # Get audio file size and log
        audio_duration = waveform.shape[1] / sample_rate
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Load separation model
        pbar.set_description("Loading speech separation model")
        model = separation.load_speech_separation_model(model_name, models_dir)
        if model is None:
            logger.error(f"Failed to load speech separation model: {model_name}")
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
            logger.error(f"Speech separation failed for {video_filename}")
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False, False
        
        # Validate the separated speech before saving
        if torch.isnan(separated_speech).any():
            logger.error(f"Separated speech contains NaN values for {video_filename}")
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False, False
            
        # Check if output is completely silent
        if torch.max(torch.abs(separated_speech)) < 1e-5:
            logger.warning(f"Separated speech is nearly silent for {video_filename}")
            
        pbar.update(1)
        
        # Save the separated speech file
        pbar.set_description(f"Saving speech audio")
        save_success = audio_io.save_audio(separated_speech, sample_rate, output_path, file_type)
        
        # Clean up temp file from initial extraction
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        pbar.update(1)
        
        if not save_success:
            logger.error(f"Failed to save audio file for {video_filename}")
            return False, False
        
        # Verify the output file exists
        expected_output = f"{output_path}.{file_type if file_type != 'both' else 'mp3'}"
        if not os.path.exists(expected_output) or os.path.getsize(expected_output) < 1000:
            logger.error(f"Output file validation failed: {expected_output}")
            return False, False
            
        logger.info(f"Successfully processed {video_filename}")
        # Report success and speech detected
        return True, True
