"""
Script to extract and separate speech from video files.
"""
import os
import sys
import argparse
import logging
import tempfile
import subprocess
import gc
from pathlib import Path
import numpy as np
import torch
import torchaudio
from moviepy import VideoFileClip
import tqdm  # Import tqdm module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODELS_DIR = Path("./models/downloaded")
DEFAULT_OUTPUT_DIR = Path("./output/separated_speech")  # Changed output location
DEFAULT_VIDEOS_DIR = Path("./data/videos")
DEFAULT_MODEL = "sepformer"
DEFAULT_CHUNK_SIZE = 10  # Default chunk size in seconds

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def check_ffmpeg_dependencies():
    """Check if ffmpeg and ffprobe are installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        logger.info("ffmpeg and ffprobe are available.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("ffmpeg or ffprobe is not installed. Please install them to enable MP3 conversion.")
        logger.error("Install ffmpeg using: apt install ffmpeg (Linux) or brew install ffmpeg (macOS).")
        return False

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

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        logger.warning("psutil not installed. Cannot monitor memory usage.")
        return 0

def load_speech_separation_model(model_name, models_dir):
    """Load the specified speech separation model."""
    # Check memory before loading model
    mem_before = get_memory_usage()
    logger.info(f"Memory usage before loading model: {mem_before:.2f} MB")
    
    logger.info(f"Loading {model_name} speech separation model")
    
    model_path = os.path.join(models_dir, model_name.lower().replace(" ", "_"))
    
    # If model directory doesn't exist, try downloading it
    if not os.path.exists(model_path):
        logger.warning(f"Model directory {model_path} not found")
        try:
            if model_name.lower() == "sepformer":
                # SpeechBrain-specific loading
                import torch
                from speechbrain.pretrained import SepformerSeparation
                
                model = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-whamr",
                    savedir=model_path,
                    run_opts={"device": "cpu"}  # Force CPU to save memory
                )
                logger.info(f"Successfully loaded SepFormer model from speechbrain")
                
                # Check memory after loading
                mem_after = get_memory_usage()
                logger.info(f"Memory usage after loading model: {mem_after:.2f} MB (increased by {mem_after - mem_before:.2f} MB)")
                
                return model
            elif model_name.lower() == "conv-tasnet":
                # Try to load Conv-TasNet
                import torch
                import librosa
                
                # This would need the appropriate model loading code
                logger.error(f"Loading for Conv-TasNet not implemented yet")
                return None
            else:
                logger.error(f"Unknown model: {model_name}")
                return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    else:
        # Model exists locally, load it
        try:
            if model_name.lower() == "sepformer":
                from speechbrain.pretrained import SepformerSeparation
                
                model = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-whamr",
                    savedir=model_path,
                    run_opts={"device": "cpu"}  # Force CPU to save memory
                )
                
                # Check memory after loading
                mem_after = get_memory_usage()
                logger.info(f"Memory usage after loading model: {mem_after:.2f} MB (increased by {mem_after - mem_before:.2f} MB)")
                
                return model
            else:
                # Load other model types as needed
                logger.error(f"Loading for {model_name} not implemented yet")
                return None
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            return None

def separate_speech(audio, model):
    """Separate speech from the mixed audio using the loaded model."""
    logger.info("Processing audio chunk for speech separation")
    
    try:
        if hasattr(model, "separate_batch"):  # SpeechBrain SepFormer interface
            # Ensure proper shape [batch, time] for model input
            if audio.dim() == 2 and audio.size(0) == 1:  # [1, time]
                waveform = audio.squeeze(0).unsqueeze(0)  # Convert from [1, time] to [1, time]
            elif audio.dim() == 1:  # [time]
                waveform = audio.unsqueeze(0)  # Convert to [1, time]
            else:
                # Unexpected shape, try to adapt
                logger.warning(f"Unexpected audio shape: {audio.shape}, attempting to reshape")
                waveform = audio.view(1, -1)  # Reshape to [1, time]
            
            # Store expected output length
            expected_length = waveform.shape[1]
            
            # Perform separation
            est_sources = model.separate_batch(waveform)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # est_sources will have shape [batch, n_sources, time]
            # We'll take the first source as the speech and ensure it has shape [1, time]
            if est_sources.dim() == 3:
                # Normal case: [batch, n_sources, time]
                separated_speech = est_sources[0, 0, :].unsqueeze(0)  # Shape [1, time]
            elif est_sources.dim() == 2:
                # Alternative case: [n_sources, time] or [batch, time]
                separated_speech = est_sources[0, :].unsqueeze(0)  # Shape [1, time]
            elif est_sources.dim() == 1:
                # Just a single time dimension
                separated_speech = est_sources.unsqueeze(0)  # Shape [1, time]
            else:
                logger.error(f"Unexpected output shape from model: {est_sources.shape}")
                logger.warning("Using original audio as fallback")
                return audio
            
            # Ensure output has the correct shape [1, time]
            # This should match the chunk's time dimension
            if separated_speech.shape[1] != expected_length:
                logger.warning(f"Output shape {separated_speech.shape} doesn't match input shape {waveform.shape}")
                
                # Critical: Handle severe shape mismatch - if output is very small compared to input
                # This is likely causing the silent output issue
                if separated_speech.shape[1] < expected_length * 0.1:  # If output is less than 10% of expected
                    logger.warning("Severe shape mismatch detected. Using original audio as fallback.")
                    return audio  # Use the original audio instead
                
                # Try to fix the shape
                if separated_speech.numel() == waveform.shape[1]:
                    separated_speech = separated_speech.view(1, waveform.shape[1])
                elif separated_speech.shape[1] < waveform.shape[1]:
                    # Pad with zeros
                    logger.warning("Output shorter than expected. Padding with zeros.")
                    padded = torch.zeros((1, waveform.shape[1]), device=separated_speech.device)
                    padded[:, :separated_speech.shape[1]] = separated_speech
                    separated_speech = padded
            
            # Check if the output is mostly silence (zeros or very low values)
            if separated_speech.abs().mean() < 0.001:  # Check if average amplitude is very low
                logger.warning("Separated speech appears to be mostly silence. Using original audio instead.")
                return audio  # Use the original audio instead
                
            # Normalize the output to ensure it has sufficient amplitude
            max_val = separated_speech.abs().max()
            if max_val > 0:
                gain_factor = min(0.9 / max_val, 3.0)  # Boost amplitude but avoid excessive gain
                if gain_factor > 1.1:  # Only apply if gain is significant
                    logger.info(f"Applying gain factor of {gain_factor:.2f} to increase volume")
                    separated_speech = separated_speech * gain_factor
            
            # Clean up to free memory
            del est_sources
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return separated_speech
        else:
            logger.error("Model interface not supported")
            return None
    
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory error during speech separation")
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"Out of memory error: {e}")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.error(f"Error during speech separation: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during speech separation: {e}")
        return None

def separate_speech_chunked(audio, model, sample_rate, chunk_size_sec=DEFAULT_CHUNK_SIZE, overlap_sec=1):
    """Separate speech from mixed audio in chunks to manage memory."""
    logger.info(f"Performing speech separation with chunked processing (chunk size: {chunk_size_sec}s)")
    
    try:
        # Get audio length in samples
        audio_length = audio.shape[1]
        chunk_size = int(chunk_size_sec * sample_rate)  # Convert seconds to samples
        overlap_size = int(overlap_sec * sample_rate)  # Overlap in samples
        
        # If audio is shorter than a single chunk, process it directly
        if audio_length <= chunk_size:
            return separate_speech(audio, model)
        
        # Initialize output tensor for the full separated speech
        separated_speech = torch.zeros((1, audio_length), device=audio.device)
        
        # Calculate number of chunks for progress bar
        total_chunks = (audio_length + (chunk_size - overlap_size) - 1) // (chunk_size - overlap_size)
        
        # Process audio in chunks with overlap
        with tqdm.tqdm(total=total_chunks, desc="Separating speech", unit="chunk") as pbar:
            for start_idx in range(0, audio_length, chunk_size - overlap_size):
                # Clear memory from previous iterations
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
                # Calculate end index for this chunk
                end_idx = min(start_idx + chunk_size, audio_length)
                
                # Extract chunk
                chunk = audio[:, start_idx:end_idx]
                chunk_length = chunk.shape[1]
                
                # Process chunk
                separated_chunk = separate_speech(chunk, model)
                
                if separated_chunk is None:
                    logger.error(f"Failed to separate chunk at {start_idx//sample_rate}s-{end_idx//sample_rate}s")
                    continue
                
                # Verify shape of separated_chunk
                if separated_chunk.shape[1] != chunk_length:
                    # Try to reshape if possible
                    if separated_chunk.numel() == chunk_length:
                        separated_chunk = separated_chunk.view(1, chunk_length)
                    elif separated_chunk.numel() < chunk_length:
                        # Pad with zeros
                        padded = torch.zeros((1, chunk_length), device=separated_chunk.device)
                        padded[:, :separated_chunk.numel()] = separated_chunk.view(1, -1)
                        separated_chunk = padded
                    elif separated_chunk.numel() > chunk_length:
                        # Trim
                        separated_chunk = separated_chunk[:, :chunk_length]
                
                # Apply cross-fade for overlapping regions (except for first chunk)
                if start_idx > 0:
                    # Calculate overlap region
                    overlap_start = start_idx
                    overlap_end = min(start_idx + overlap_size, audio_length)
                    overlap_length = overlap_end - overlap_start
                    
                    # Ensure we don't try to access beyond the length of separated_chunk
                    if overlap_length > separated_chunk.shape[1]:
                        overlap_length = separated_chunk.shape[1]
                    
                    # Create linear cross-fade weights
                    fade_in = torch.linspace(0, 1, overlap_length).view(1, -1).to(audio.device)
                    fade_out = 1 - fade_in
                    
                    # Apply cross-fade in the overlap region
                    try:
                        separated_speech[:, overlap_start:overlap_end] = (
                            fade_out * separated_speech[:, overlap_start:overlap_end] + 
                            fade_in * separated_chunk[:, :overlap_length]
                        )
                        
                        # Copy the non-overlapping part
                        remaining_length = separated_chunk.shape[1] - overlap_length
                        if remaining_length > 0 and (overlap_end + remaining_length) <= audio_length:
                            separated_speech[:, overlap_end:overlap_end+remaining_length] = separated_chunk[:, overlap_length:overlap_length+remaining_length]
                    except RuntimeError as e:
                        logger.error(f"Error merging chunks at overlap region: {e}")
                        # Try an alternative approach - just copy without cross-fade
                        copy_length = min(end_idx - start_idx, separated_chunk.shape[1])
                        separated_speech[:, start_idx:start_idx+copy_length] = separated_chunk[:, :copy_length]
                else:
                    # First chunk, just copy directly
                    copy_length = min(end_idx, separated_chunk.shape[1])
                    separated_speech[:, :copy_length] = separated_chunk[:, :copy_length]
                
                # Update progress bar
                pbar.update(1)
        
        return separated_speech
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory error during speech separation")
        # Try to free memory
        torch.cuda.empty_cache()
        # If chunking already failed, try with smaller chunk size
        if chunk_size_sec > 2:
            logger.info(f"Retrying with smaller chunk size ({chunk_size_sec/2}s)")
            return separate_speech_chunked(audio, model, sample_rate, chunk_size_sec=chunk_size_sec/2)
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Try with smaller chunk size
            if chunk_size_sec > 2:
                logger.info(f"Retrying with smaller chunk size ({chunk_size_sec/2}s)")
                return separate_speech_chunked(audio, model, sample_rate, chunk_size_sec=chunk_size_sec/2)
        else:
            logger.error(f"Runtime error during speech separation: {e}")
            # Try again with smaller chunks and no overlap
            if chunk_size_sec > 2:
                logger.info(f"Retrying with smaller chunk size ({chunk_size_sec/2}s) and no overlap")
                return separate_speech_chunked(audio, model, sample_rate, chunk_size_sec=chunk_size_sec/2, overlap_sec=0)
        return None
    except Exception as e:
        logger.error(f"Error during chunked speech separation: {e}")
        return None

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

def process_file(video_path, output_dir, model_name, models_dir, chunk_size_sec=DEFAULT_CHUNK_SIZE, file_type="mp3"):
    """Process a single video file for speech separation."""
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_path = os.path.join(output_dir, f"{video_name}_speech")  # No extension, added later
    
    # Extract audio from video
    logger.info(f"Processing {video_filename}")
    with tqdm.tqdm(total=4, desc=f"Processing {video_filename}", unit="step") as pbar:
        pbar.set_description("Extracting audio")
        waveform, sample_rate, temp_audio_path = extract_audio_from_video(video_path)
        if waveform is None:
            return False
        pbar.update(1)
        
        # Get audio file size and log
        audio_duration = waveform.shape[1] / sample_rate
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Load separation model
        pbar.set_description("Loading model")
        model = load_speech_separation_model(model_name, models_dir)
        if model is None:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False
        pbar.update(1)
        
        # Separate speech using chunked processing
        pbar.set_description("Separating speech")
        separated_speech = separate_speech_chunked(waveform, model, sample_rate, chunk_size_sec)
        
        # Clean up model to free memory
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if separated_speech is None:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return False
        pbar.update(1)
        
        # Save separated speech
        file_type_desc = "WAV" if file_type == "wav" else "MP3" if file_type == "mp3" else "WAV+MP3"
        pbar.set_description(f"Saving {file_type_desc} audio")
        success = save_audio(separated_speech, sample_rate, output_path, file_type)
        
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        pbar.update(1)
        
        return success

def find_video_files(input_paths=None, recursive=False):
    """Find video files in the specified paths or default directory."""
    video_files = []
    
    # If no input paths provided, use default videos directory
    if not input_paths:
        input_paths = [str(DEFAULT_VIDEOS_DIR)]
    
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

def select_videos_interactively(videos_dir=None, recursive=False):
    """Display a list of available videos and prompt user to select."""
    # Find video files in the specified directory or default
    input_paths = [videos_dir] if videos_dir else [str(DEFAULT_VIDEOS_DIR)]
    all_videos = find_video_files(input_paths, recursive)
    
    if not all_videos:
        logger.error(f"No video files found in {input_paths[0]}")
        return [], None
    
    # Display the list of available videos
    print("\n=== Available Video Files ===")
    for i, video_path in enumerate(all_videos, 1):
        # Get file size in MB
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"[{i}] {os.path.basename(video_path)} ({size_mb:.1f} MB)")
    
    # Prompt for selection
    while True:
        print("\nOptions:")
        print("- Enter numbers (e.g., '1,3,5') to select specific videos")
        print("- Enter 'all' to process all videos")
        print("- Enter 'q' to quit")
        
        selection = input("\nSelect videos to process: ").strip().lower()
        
        if selection == 'q':
            return [], None
            
        if selection == 'all':
            selected_videos = all_videos
            break
        
        try:
            # Parse comma-separated indices
            indices = [int(idx.strip()) for idx in selection.split(',')]
            selected_videos = []
            
            for idx in indices:
                if 1 <= idx <= len(all_videos):
                    selected_videos.append(all_videos[idx-1])
                else:
                    print(f"Error: {idx} is not a valid video number")
                    break
            else:
                # If no break occurred in the loop
                if selected_videos:
                    break
                print("No valid videos selected. Please try again.")
                
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")
    
    # Now prompt for file type selection
    print("\n=== Select Output File Format ===")
    print("[1] MP3 format (default)")
    print("[2] WAV format")
    print("[3] Both WAV and MP3")
    
    while True:
        file_type_selection = input("\nSelect file format [1-3]: ").strip()
        
        if not file_type_selection:
            # Default to MP3
            file_type = "mp3"
            break
        elif file_type_selection in ["1", "mp3"]:
            file_type = "mp3"
            break
        elif file_type_selection in ["2", "wav"]:
            file_type = "wav"
            break
        elif file_type_selection in ["3", "both"]:
            file_type = "both"
            break
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")
    
    return selected_videos, file_type

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract and separate speech from video files")
    parser.add_argument(
        "input",
        nargs="*",
        help="Input video file(s) or directory. If not provided, interactive selection will be used."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save separated speech files"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["sepformer", "conv-tasnet", "dual-path-rnn"],
        help="Speech separation model to use"
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing downloaded models"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process video files in subdirectories recursively"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive video selection mode"
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=DEFAULT_CHUNK_SIZE,
        help="Size of audio chunks to process in seconds"
    )
    parser.add_argument(
        "--file-type",
        type=str,
        choices=["wav", "mp3", "both", "1", "2", "3"],
        default="mp3",
        help="Output file format: wav (1), mp3 (2), or both (3)"
    )
    
    args = parser.parse_args()
    
    # Process file type argument
    file_type = args.file_type
    if file_type == "1":
        file_type = "wav"
    elif file_type == "2":
        file_type = "mp3"
    elif file_type == "3":
        file_type = "both"
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    ensure_dir_exists(args.output_dir)
    
    # Check for dependencies
    try:
        import speechbrain
        import moviepy
        import tqdm
        # Try to import pydub for MP3 conversion
        try:
            from pydub import AudioSegment
        except ImportError:
            # Check if ffmpeg is available as fallback
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                logger.info("Using ffmpeg for MP3 conversion")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("Neither pydub nor ffmpeg found. MP3 conversion may not work.")
                logger.warning("Install with: pip install pydub")
                logger.warning("Or install ffmpeg: apt install ffmpeg / brew install ffmpeg")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install speechbrain moviepy torchaudio tqdm pydub")
        return 1
    
    # Collect video files - either from arguments or interactive selection
    video_files = []
    file_type_from_interactive = None
    
    # Use interactive mode if no input args or --interactive flag
    if not args.input or args.interactive:
        # Interactive video selection
        video_files, file_type_from_interactive = select_videos_interactively(
            videos_dir=None if not args.input else args.input[0],
            recursive=args.recursive
        )
    else:
        # Use provided input arguments
        video_files = find_video_files(args.input, args.recursive)
    
    if not video_files:
        logger.error("No video files selected for processing")
        return 1
    
    # Process file type argument - interactive selection overrides command line if provided
    file_type = file_type_from_interactive or args.file_type
    if file_type == "1":
        file_type = "wav"
    elif file_type == "2":
        file_type = "mp3"
    elif file_type == "3":
        file_type = "both"
    
    # Process each video file
    successful = 0
    total_files = len(video_files)
    
    # Show overall progress
    with tqdm.tqdm(total=total_files, desc="Overall progress", unit="file") as overall_pbar:
        for i, video_path in enumerate(video_files):
            overall_pbar.set_description(f"File {i+1}/{total_files}")
            
            # Clear memory before processing each file
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if process_file(video_path, args.output_dir, args.model, args.models_dir, args.chunk_size, file_type):
                successful += 1
                overall_pbar.set_postfix(success_rate=f"{successful}/{i+1}")
            else:
                logger.error(f"Failed to process {os.path.basename(video_path)}")
            
            overall_pbar.update(1)
    
    print(f"\n✅ Processed {successful}/{total_files} videos successfully")
    print(f"🎵 Audio files saved to: {args.output_dir}")
    
    # Update message based on file type
    if file_type == "wav":
        print("Files were saved in WAV format.")
    elif file_type == "mp3":
        print("Files were saved in MP3 format.")
    else:
        print("Files were saved in both WAV and MP3 formats.")
    
    return 0 if successful == total_files else 1

if __name__ == "__main__":
    sys.exit(main())
