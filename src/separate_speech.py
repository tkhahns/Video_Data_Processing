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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODELS_DIR = Path("./models/downloaded")
DEFAULT_OUTPUT_DIR = Path("./data/separated_speech")
DEFAULT_VIDEOS_DIR = Path("./data/videos")  # Default videos directory
DEFAULT_MODEL = "sepformer"  # Default model from our model list
DEFAULT_CHUNK_SIZE = 10  # Default chunk size in seconds

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

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
            
            # Log shapes for debugging
            logger.debug(f"Input shape: {audio.shape}, Model input shape: {waveform.shape}")
            
            # Check memory
            mem_before = get_memory_usage()
            logger.debug(f"Memory before separation: {mem_before:.2f} MB")
            
            # Perform separation
            est_sources = model.separate_batch(waveform)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Check memory after separation
            mem_after = get_memory_usage()
            logger.debug(f"Memory after separation: {mem_after:.2f} MB (delta: {mem_after-mem_before:.2f} MB)")
            
            # Log shape for debugging
            logger.debug(f"Estimated sources shape: {est_sources.shape}")
            
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
                return None
            
            # Ensure output has the correct shape [1, time]
            # This should match the chunk's time dimension
            if separated_speech.shape[1] != waveform.shape[1]:
                logger.warning(f"Output shape {separated_speech.shape} doesn't match input shape {waveform.shape}")
                # Try to fix the shape
                if separated_speech.numel() == waveform.shape[1]:
                    separated_speech = separated_speech.view(1, waveform.shape[1])
            
            # Final check
            logger.debug(f"Final separated speech shape: {separated_speech.shape}")
            
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
        
        # Process audio in chunks with overlap
        for start_idx in range(0, audio_length, chunk_size - overlap_size):
            # Clear memory from previous iterations
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            # Calculate end index for this chunk
            end_idx = min(start_idx + chunk_size, audio_length)
            
            # Extract chunk
            chunk = audio[:, start_idx:end_idx]
            chunk_length = chunk.shape[1]
            logger.debug(f"Processing chunk {start_idx//sample_rate}s-{end_idx//sample_rate}s ({chunk_length/sample_rate:.1f}s)")
            
            # Process chunk
            mem_before = get_memory_usage()
            separated_chunk = separate_speech(chunk, model)
            mem_after = get_memory_usage()
            
            if separated_chunk is None:
                logger.error(f"Failed to separate chunk {start_idx//sample_rate}s-{end_idx//sample_rate}s")
                continue
            
            # Verify shape of separated_chunk
            if separated_chunk.shape[1] != chunk_length:
                logger.warning(f"Shape mismatch: chunk {chunk.shape}, separated {separated_chunk.shape}")
                # Try to reshape if possible
                if separated_chunk.numel() == chunk_length:
                    separated_chunk = separated_chunk.view(1, chunk_length)
                elif separated_chunk.numel() < chunk_length:
                    # Pad with zeros
                    logger.warning(f"Padding output to match expected length")
                    padded = torch.zeros((1, chunk_length), device=separated_chunk.device)
                    padded[:, :separated_chunk.numel()] = separated_chunk.view(1, -1)
                    separated_chunk = padded
                elif separated_chunk.numel() > chunk_length:
                    # Trim
                    logger.warning(f"Trimming output to match expected length")
                    separated_chunk = separated_chunk[:, :chunk_length]
            
            logger.debug(f"Chunk processed. Memory: {mem_after:.2f}MB (delta: {mem_after-mem_before:.2f}MB)")
            logger.debug(f"Separated chunk shape: {separated_chunk.shape}")
            
            # Apply cross-fade for overlapping regions (except for first chunk)
            if start_idx > 0:
                # Calculate overlap region
                overlap_start = start_idx
                overlap_end = min(start_idx + overlap_size, audio_length)
                overlap_length = overlap_end - overlap_start
                
                # Ensure we don't try to access beyond the length of separated_chunk
                if overlap_length > separated_chunk.shape[1]:
                    logger.warning(f"Overlap length {overlap_length} exceeds separated chunk length {separated_chunk.shape[1]}")
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
                    logger.warning("Falling back to direct copy without cross-fade")
                    copy_length = min(end_idx - start_idx, separated_chunk.shape[1])
                    separated_speech[:, start_idx:start_idx+copy_length] = separated_chunk[:, :copy_length]
            else:
                # First chunk, just copy directly
                copy_length = min(end_idx, separated_chunk.shape[1])
                separated_speech[:, :copy_length] = separated_chunk[:, :copy_length]
        
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
            logger.error(f"Out of memory error: {e}")
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

def save_audio(waveform, sample_rate, output_path):
    """Save audio waveform to file."""
    logger.info(f"Saving separated speech to {output_path}")
    
    try:
        # Ensure waveform has the right shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Save as WAV
        torchaudio.save(output_path, waveform, sample_rate)
        return True
    
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        return False

def process_file(video_path, output_dir, model_name, models_dir, chunk_size_sec=DEFAULT_CHUNK_SIZE):
    """Process a single video file for speech separation."""
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    output_path = os.path.join(output_dir, f"{video_name}_speech.wav")
    
    # Extract audio from video
    waveform, sample_rate, temp_audio_path = extract_audio_from_video(video_path)
    if waveform is None:
        return False
    
    # Get audio file size and log
    audio_duration = waveform.shape[1] / sample_rate
    logger.info(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Load separation model
    model = load_speech_separation_model(model_name, models_dir)
    if model is None:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return False
    
    # Separate speech using chunked processing
    separated_speech = separate_speech_chunked(waveform, model, sample_rate, chunk_size_sec)
    
    # Clean up model to free memory
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if separated_speech is None:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return False
    
    # Save separated speech
    success = save_audio(separated_speech, sample_rate, output_path)
    
    # Clean up temp file
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
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
        return []
    
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
            return []
            
        if selection == 'all':
            return all_videos
        
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
                    return selected_videos
                print("No valid videos selected. Please try again.")
                
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")

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
    
    args = parser.parse_args()
    
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
        # Try to import psutil for memory monitoring
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not installed. Memory monitoring will be limited.")
            logger.warning("Install with: pip install psutil")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install speechbrain moviepy torchaudio psutil")
        return 1
    
    # Collect video files - either from arguments or interactive selection
    video_files = []
    
    # Use interactive mode if no input args or --interactive flag
    if not args.input or args.interactive:
        # Interactive video selection
        video_files = select_videos_interactively(
            videos_dir=None if not args.input else args.input[0],
            recursive=args.recursive
        )
    else:
        # Use provided input arguments
        video_files = find_video_files(args.input, args.recursive)
    
    if not video_files:
        logger.error("No video files selected for processing")
        return 1
    
    # Process each video file
    successful = 0
    for video_path in video_files:
        logger.info(f"Processing {video_path}")
        # Clear memory before processing each file
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if process_file(video_path, args.output_dir, args.model, args.models_dir, args.chunk_size):
            successful += 1
        else:
            logger.error(f"Failed to process {video_path}")
    
    logger.info(f"Processed {successful}/{len(video_files)} videos successfully")
    return 0 if successful == len(video_files) else 1

if __name__ == "__main__":
    sys.exit(main())
