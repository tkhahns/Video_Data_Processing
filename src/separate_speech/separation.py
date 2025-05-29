"""
Speech separation functionality.
"""
import os
import logging
import torch
import tqdm
import gc
from .utils import get_memory_usage

logger = logging.getLogger(__name__)

def load_speech_separation_model(model_name, models_dir):
    """Load the specified speech separation model."""
    # Check memory before loading model
    mem_before = get_memory_usage()
    logger.info(f"Memory usage before loading model: {mem_before:.2f} MB")
    
    logger.info(f"Loading {model_name} speech separation model")
    
    try:
        if model_name.lower() == "sepformer":
            from speechbrain.pretrained import SepformerSeparation
            
            # Use SepFormer from speechbrain's pretrained models
            model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr",
                run_opts={"device": "cpu"}  # Force CPU to save memory
            )
            
            # Check memory after loading
            mem_after = get_memory_usage()
            logger.info(f"Memory usage after loading model: {mem_after:.2f} MB (increased by {mem_after - mem_before:.2f} MB)")
            
            return model
        elif model_name.lower() == "conv-tasnet":
            # Try to load Conv-TasNet
            logger.error(f"Loading for Conv-TasNet not implemented yet")
            return None
        else:
            logger.error(f"Unknown model: {model_name}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
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

def separate_speech_chunked(audio, model, sample_rate, chunk_size_sec=10, overlap_sec=1):
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

def separate_speech_chunked(waveform, model, sample_rate, chunk_size_sec=10):
    """
    Separate speech from noise using a pre-trained model with chunking.
    
    Args:
        waveform: Input audio waveform
        model: Loaded speech separation model
        sample_rate: Sample rate of the audio
        chunk_size_sec: Size of audio chunks to process in seconds
        
    Returns:
        Separated speech waveform
    """
    # Split waveform into chunks
    chunk_size = int(chunk_size_sec * sample_rate)
    chunks = [waveform[:, i:i+chunk_size] for i in range(0, waveform.size(1), chunk_size)]
    
    # Process each chunk
    output_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            # Process chunk
            with torch.no_grad():
                est_sources = model.separate_batch(chunk)
            
            # Check for NaN values in output
            if torch.isnan(est_sources).any():
                logger.warning(f"NaN values detected in model output for chunk {i+1}/{len(chunks)}")
                logger.warning("Using original audio chunk as fallback")
                output_chunks.append(chunk)
                continue
            
            # Verify the output shape matches expectations
            if est_sources.shape[1] == chunk.shape[1]:
                # Shape is correct, extract the speech component (first source)
                separated_speech = est_sources[:, 0]
                
                # Additional validation - check if output is silent or very quiet
                if torch.max(torch.abs(separated_speech)) < 1e-4:
                    logger.warning(f"Output for chunk {i+1}/{len(chunks)} is nearly silent, using original audio")
                    output_chunks.append(chunk)
                else:
                    output_chunks.append(separated_speech)
            else:
                # Shape mismatch, log the error and use original chunk as fallback
                logger.warning(f"Shape mismatch on chunk {i+1}/{len(chunks)}: Expected shape {chunk.shape[1]}, " 
                              f"got {est_sources.shape[1]}. Using original audio for this chunk.")
                output_chunks.append(chunk)
        except Exception as e:
            # Handle any exceptions during separation
            logger.error(f"Error processing chunk {i+1}/{len(chunks)}: {e}")
            logger.error("Using original audio chunk as fallback")
            output_chunks.append(chunk)  # Use original chunk as fallback
            continue
    
    # Concatenate all processed chunks
    return torch.cat(output_chunks, dim=1)
