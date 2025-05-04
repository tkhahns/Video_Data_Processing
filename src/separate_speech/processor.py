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

def check_diarization_prerequisites():
    """Check if all prerequisites for diarization are met (pyannote or speechbrain)."""
    try:
        import pyannote.audio
        logger.info("Found pyannote.audio for diarization.")
        return True
    except ImportError:
        try:
            import speechbrain
            version = getattr(speechbrain, "__version__", "unknown")
            logger.info(f"Found speechbrain version {version} for fallback diarization.")
            return True
        except ImportError:
            logger.error("Neither pyannote.audio nor speechbrain is installed. Cannot perform dialogue detection.")
            logger.error("Install with: pip install pyannote.audio==2.1.1 or pip install speechbrain")
            return False

def process_file(video_path, output_dir, model_name, models_dir, chunk_size_sec=10, file_type="mp3", detect_dialogues=False, skip_no_speech=True, min_speech_seconds=1.0):
    """
    Process a single video file for speech separation.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output files
        model_name: Name of the speech separation model to use
        models_dir: Directory containing downloaded models
        chunk_size_sec: Size of audio chunks to process in seconds
        file_type: Output format ("wav", "mp3", "both")
        detect_dialogues: If True, detect dialogues and save them separately
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
    with tqdm.tqdm(total=5 if detect_dialogues else 4, desc=f"Processing {video_filename}", unit="step") as pbar:
        pbar.set_description("Extracting audio")
        waveform, sample_rate, temp_audio_path = extraction.extract_audio_from_video(video_path)
        if waveform is None:
            return False, False
        pbar.update(1)
        
        # Get audio file size and log
        audio_duration = waveform.shape[1] / sample_rate
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Check for speech presence if requested
        if skip_no_speech and temp_audio_path:
            pbar.set_description("Checking for speech presence")
            # Import the diarization module here to avoid circular imports
            from . import diarization
            speech_detected, speech_duration = diarization.detect_speech_presence(
                temp_audio_path, min_speech_seconds=min_speech_seconds
            )
            
            if not speech_detected:
                logger.info(f"Skipping {video_filename} - no significant speech detected")
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                
                # If we want to save empty output files to indicate the file was processed
                # but had no speech, uncomment and modify the following:
                # no_speech_dir = os.path.join(output_dir, "no_speech_detected")
                # ensure_dir_exists(no_speech_dir)
                # with open(os.path.join(no_speech_dir, f"{video_name}.txt"), "w") as f:
                #     f.write(f"No speech detected in {video_filename}\n")
                #     f.write(f"Audio duration: {audio_duration:.2f} seconds\n")
                #     f.write(f"Speech duration: {speech_duration:.2f} seconds\n")
                
                return True, False  # Processing successful but no speech detected
            
            logger.info(f"Speech detected: {speech_duration:.2f} seconds of speech in {audio_duration:.2f} seconds audio")
        
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
        
        # Always save the full separated speech file first
        pbar.set_description(f"Saving full speech audio")
        audio_io.save_audio(separated_speech, sample_rate, output_path, file_type)
        
        # If dialogue detection is requested, process further
        dialogue_success = False
        if detect_dialogues:
            # Check prerequisites before attempting diarization
            if not check_diarization_prerequisites():
                logger.error("Prerequisites for dialogue detection not met.")
                pbar.update(1)
            else:
                # Save temp file for diarization
                pbar.set_description("Preparing for dialogue detection")
                try:
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_file.close()
                    
                    # Save the speech as WAV for diarization
                    import torchaudio
                    torchaudio.save(temp_file.name, separated_speech.cpu(), sample_rate)
                    
                    # Import diarization module - correctly import locally to avoid circular imports
                    from . import diarization
                    
                    # Load diarization model
                    pbar.set_description("Loading diarization model")
                    diarization_model = diarization.load_diarization_model(device="cpu")  # CPU is more reliable for pyannote
                    
                    if diarization_model is None:
                        logger.error("Failed to load diarization model, skipping dialogue detection")
                    else:
                        # Perform diarization
                        segments, speech_detected = diarization.perform_diarization(
                            temp_file.name, diarization_model, 
                            min_speakers=1, max_speakers=8, 
                            min_speech_seconds=min_speech_seconds
                        )
                        
                        # Check if segments were returned and speech was detected
                        if not segments or not speech_detected:
                            logger.warning(f"No valid speech segments found in {video_name}")
                        else:
                            # Clean up diarization model
                            del diarization_model
                            clean_memory()
                            
                            # Calculate dialogue output directory (subdirectory for dialogues)
                            dialogue_output_dir = os.path.join(output_dir, f"{video_name}_dialogues")
                            
                            # Extract dialogue segments
                            pbar.set_description("Extracting dialogue segments")
                            logger.info(f"Extracting {len(segments)} dialogue segments")
                            saved_files = diarization.extract_dialogue_segments(
                                separated_speech, sample_rate, segments, dialogue_output_dir, 
                                video_name, save_speakers_separately=True
                            )
                            
                            if saved_files:
                                logger.info(f"Extracted {len(saved_files)} dialogue segments from {video_filename}")
                                dialogue_success = True
                            else:
                                logger.warning(f"No dialogue segments were saved for {video_filename}")
                    
                    # Clean up temp file
                    if 'temp_file' in locals() and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                        
                except ImportError as e:
                    logger.error(f"ImportError during dialogue detection: {e}")
                    logger.error("Make sure speechbrain is installed: pip install speechbrain")
                except Exception as e:
                    logger.error(f"Error during dialogue detection: {e}")
                
                pbar.update(1)
        
        # Clean up temp file from initial extraction
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        pbar.update(1)
        
        # Report success and speech detected
        return True, True

def process_file_with_dialogue_detection(video_path, output_dir, model_name, models_dir, chunk_size_sec=10, file_type="mp3"):
    """
    Process a single video file with dialogue detection.
    Convenience wrapper around process_file with dialogue detection enabled.
    """
    return process_file(video_path, output_dir, model_name, models_dir, chunk_size_sec, file_type, detect_dialogues=True)
