"""
Video processing and speech separation workflow.
"""
import os
import logging
import tqdm
from . import separation
from . import extraction
from . import audio_io
from .utils import clean_memory

logger = logging.getLogger(__name__)

def process_file(video_path, output_dir, model_name, models_dir, chunk_size_sec=10, file_type="mp3"):
    """Process a single video file for speech separation."""
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_path = os.path.join(output_dir, f"{video_name}_speech")  # No extension, added later
    
    # Extract audio from video
    logger.info(f"Processing {video_filename}")
    with tqdm.tqdm(total=4, desc=f"Processing {video_filename}", unit="step") as pbar:
        pbar.set_description("Extracting audio")
        waveform, sample_rate, temp_audio_path = extraction.extract_audio_from_video(video_path)
        if waveform is None:
            return False
        pbar.update(1)
        
        # Get audio file size and log
        audio_duration = waveform.shape[1] / sample_rate
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Load separation model
        pbar.set_description("Loading model")
        model = separation.load_speech_separation_model(model_name, models_dir)
        if model is None:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False
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
            return False
        pbar.update(1)
        
        # Save separated speech
        file_type_desc = "WAV" if file_type == "wav" else "MP3" if file_type == "mp3" else "WAV+MP3"
        pbar.set_description(f"Saving {file_type_desc} audio")
        success = audio_io.save_audio(separated_speech, sample_rate, output_path, file_type)
        
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        pbar.update(1)
        
        return success
