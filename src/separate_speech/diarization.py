"""
Speaker diarization functionality for detecting different speakers in audio.
Uses open source models that don't require authentication tokens.
"""
import os
import torch
import numpy as np

from utils import init_logging
logger = init_logging.get_logger(__name__)

def load_diarization_model(device="cpu"):
    """
    Load an open source speaker diarization model (SpeechBrain).
    
    Args:
        device: Device to run the model on (cpu/cuda)
    
    Returns:
        The loaded diarization model or None if loading fails
    """
    try:
        # Log memory usage before loading model
        try:
            from .utils import get_memory_usage
            mem_before = get_memory_usage()
            logger.info(f"Memory usage before loading diarization model: {mem_before:.2f} MB")
        except ImportError:
            pass
        
        logger.info("Loading SpeechBrain speaker diarization model...")
        
        # Import SpeechBrain
        from speechbrain.pretrained import EncoderClassifier
        
        # Load the speaker embedding model
        embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/speechbrain/spkrec-ecapa-voxceleb"),
            run_opts={"device": device}
        )
        
        logger.info("Successfully loaded SpeechBrain speaker embedding model")
            
        # Log memory usage after loading
        try:
            mem_after = get_memory_usage()
            logger.info(f"Memory usage after loading diarization model: {mem_after:.2f} MB (increased by {mem_after - mem_before:.2f} MB)")
        except (NameError, ImportError):
            pass
            
        return embedding_model
        
    except ImportError as e:
        logger.error(f"Failed to import speechbrain: {e}")
        logger.error("Please install speechbrain with: pip install speechbrain")
        return None
    except Exception as e:
        logger.error(f"Error loading diarization model: {e}")
        return None

def load_pyannote_pipeline(model_name="pyannote/speaker-diarization", device="cpu"):
    """
    Load pyannote.audio speaker diarization pipeline.
    Returns pipeline or None if not available.
    """
    try:
        from pyannote.audio import Pipeline
        logger.info(f"Loading pyannote.audio pipeline: {model_name}")
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=None)
        pipeline.to(device)
        logger.info("Successfully loaded pyannote.audio pipeline")
        return pipeline
    except Exception as e:
        logger.warning(f"Could not load pyannote.audio pipeline: {e}")
        return None

def segment_audio(waveform, sample_rate, segment_length_sec=5.0, overlap_sec=1.0):
    """
    Segment audio into chunks for speaker diarization.
    
    Args:
        waveform: Audio waveform tensor [channels, samples]
        sample_rate: Sample rate of the audio
        segment_length_sec: Length of each segment in seconds
        overlap_sec: Overlap between segments in seconds
        
    Returns:
        List of segments as (start_sample, end_sample) tuples
    """
    segment_length = int(segment_length_sec * sample_rate)
    overlap_length = int(overlap_sec * sample_rate)
    stride = segment_length - overlap_length
    
    # Get audio length
    audio_length = waveform.shape[1]
    
    # Create segments
    segments = []
    for start in range(0, audio_length - overlap_length, stride):
        end = min(start + segment_length, audio_length)
        if end - start < 0.5 * segment_length:  # Skip very short segments
            continue
        segments.append((start, end))
        if end == audio_length:
            break
    
    return segments

def cluster_embeddings(embeddings, method="agglomerative", min_speakers=1, max_speakers=8):
    """
    Cluster speaker embeddings to identify different speakers.
    
    Args:
        embeddings: Speaker embeddings (n_segments, embedding_dim)
        method: Clustering method ("agglomerative" or "kmeans")
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        
    Returns:
        Labels for each segment (n_segments,)
    """
    try:
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.metrics import silhouette_score
        
        # Convert to numpy for clustering
        if torch.is_tensor(embeddings):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Handle different embedding shapes
        if len(embeddings_np.shape) == 3:
            logger.info(f"Reshaping embeddings from {embeddings_np.shape} to 2D array for clustering")
            embeddings_np = np.mean(embeddings_np, axis=1)
        elif len(embeddings_np.shape) > 3:
            original_shape = embeddings_np.shape
            embeddings_np = embeddings_np.reshape(original_shape[0], -1)
            
        # If only one segment, assign single speaker
        if embeddings_np.shape[0] <= 1:
            return np.zeros(embeddings_np.shape[0], dtype=int)
        
        # Normalize embeddings
        from sklearn.preprocessing import normalize
        try:
            embeddings_np = normalize(embeddings_np)
        except ValueError:
            if len(embeddings_np.shape) > 2:
                embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
                embeddings_np = normalize(embeddings_np)
        
        # Try different numbers of clusters
        best_score = -1
        best_labels = None
        best_n_clusters = min_speakers
        
        # Cap max_speakers to number of segments
        max_speakers = min(max_speakers, embeddings_np.shape[0])
        
        # Only try clustering if we have enough segments
        if embeddings_np.shape[0] >= 2:
            for n_clusters in range(min_speakers, max_speakers + 1):
                try:
                    if method == "agglomerative":
                        try:
                            clustering = AgglomerativeClustering(
                                n_clusters=n_clusters,
                                affinity="cosine",
                                linkage="average"
                            )
                        except TypeError:
                            clustering = AgglomerativeClustering(n_clusters=n_clusters)
                    else:  # kmeans
                        try:
                            clustering = KMeans(n_clusters=n_clusters, n_init=10)
                        except TypeError:
                            clustering = KMeans(n_clusters=n_clusters, n_init="auto")
                    
                    labels = clustering.fit_predict(embeddings_np)
                    
                    # If only one cluster or all segments in same cluster, no need for silhouette
                    if n_clusters == 1 or len(np.unique(labels)) <= 1:
                        score = 0
                    else:
                        try:
                            score = silhouette_score(embeddings_np, labels, metric="cosine")
                        except:
                            try:
                                score = silhouette_score(embeddings_np, labels)
                            except:
                                score = 0
                                
                    # Update best if higher score
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_n_clusters = n_clusters
                except Exception as e:
                    logger.warning(f"Clustering with {n_clusters} clusters failed: {e}")
                    continue
                    
            logger.info(f"Selected {best_n_clusters} speakers with silhouette score {best_score:.4f}")
        else:
            # Not enough segments for clustering
            best_labels = np.zeros(embeddings_np.shape[0], dtype=int)
        
        return best_labels
        
    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        # Return basic clustering as fallback
        return np.zeros(len(embeddings), dtype=int)

def detect_speech_presence(audio_file, threshold=0.3, min_speech_seconds=1.0):
    """
    Detect if there is any human speech in an audio file.
    
    Args:
        audio_file: Path to audio file
        threshold: Speech probability threshold (0.0-1.0)
        min_speech_seconds: Minimum duration of speech required (in seconds)
        
    Returns:
        Tuple of (speech_detected (bool), speech_duration (float in seconds))
    """
    try:
        import torchaudio
        
        logger.info(f"Checking for speech presence in {audio_file}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)
        audio_duration = waveform.shape[1] / sample_rate
        
        # Try to use SpeechBrain's VAD for detection
        try:
            from speechbrain.pretrained import VAD
            vad_model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                savedir=os.path.expanduser("~/.cache/speechbrain/vad-crdnn-libriparty")
            )
            
            # Get speech/non-speech probabilities
            speech_prob = vad_model.get_speech_prob_chunk(waveform)
            
            # Get speech segments with specified threshold
            speech_segments = vad_model.get_speech_segments(
                speech_prob, 
                prob_threshold=threshold,
                close_th=0.25,  # Merge segments that are close to each other
                len_th=0.1,     # Minimum segment length in seconds
            )
            
            # Convert to list of (start, end) times
            vad_segments = [(start.item(), end.item()) for start, end in speech_segments]
            
            # Calculate total speech duration
            total_speech_duration = sum(end - start for start, end in vad_segments)
            
            # Check if there's enough speech
            speech_detected = total_speech_duration >= min_speech_seconds
            
            # Log the results
            if speech_detected:
                logger.info(f"Speech detected: {total_speech_duration:.2f} seconds of speech in {audio_duration:.2f} seconds audio")
            else:
                logger.warning(f"No significant speech detected: only {total_speech_duration:.2f} seconds in {audio_duration:.2f} seconds audio")
            
            return speech_detected, total_speech_duration
            
        except Exception as e:
            logger.warning(f"Error using VAD to detect speech: {e}")
            
            # Fallback: Use a simpler energy-based approach
            # Calculate audio energy (volume)
            energy = torch.mean(waveform ** 2)
            
            # Define a very basic threshold for energy
            basic_energy_threshold = 0.001
            speech_detected = energy > basic_energy_threshold
            
            logger.info(f"Fallback speech detection: {'speech detected' if speech_detected else 'no speech detected'} (energy: {energy:.6f})")
            
            # Return True by default for the fallback method
            return speech_detected, audio_duration if speech_detected else 0
            
    except Exception as e:
        logger.error(f"Error detecting speech presence: {e}")
        # Return True by default if detection fails
        return True, 0

def perform_pyannote_diarization(audio_file, pipeline, min_speech_seconds=1.0):
    """
    Perform diarization using pyannote.audio pipeline.
    Returns (segments, speech_detected)
    """
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_file)
        duration = waveform.shape[1] / sample_rate

        diarization = pipeline(audio_file)
        segments = []
        speaker_map = {}
        speaker_idx = 1

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Map speaker labels to "Person 1", "Person 2", etc.
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Person {speaker_idx}"
                speaker_idx += 1
            mapped_speaker = speaker_map[speaker]
            segments.append((turn.start, turn.end, mapped_speaker))

        # Filter out very short segments
        segments = [seg for seg in segments if (seg[1] - seg[0]) >= 0.5]

        total_speech = sum(seg[1] - seg[0] for seg in segments)
        speech_detected = total_speech >= min_speech_seconds

        if speech_detected:
            logger.info(f"pyannote: Detected {len(segments)} segments, {total_speech:.2f}s speech")
        else:
            logger.warning("pyannote: No significant speech detected")

        return segments, speech_detected
    except Exception as e:
        logger.error(f"Error in pyannote diarization: {e}")
        return [], False

def perform_diarization(audio_file, embedding_model=None, min_speakers=1, max_speakers=8, min_speech_seconds=1.0):
    """
    Perform speaker diarization on an audio file.
    Tries pyannote.audio if available, else falls back to SpeechBrain.
    """
    # Try pyannote.audio first
    try:
        from pyannote.audio import Pipeline
        pipeline = load_pyannote_pipeline(device="cpu")
        if pipeline is not None:
            return perform_pyannote_diarization(audio_file, pipeline, min_speech_seconds=min_speech_seconds)
    except ImportError:
        logger.info("pyannote.audio not available, falling back to SpeechBrain.")

    # Fallback: SpeechBrain
    if embedding_model is None:
        logger.error("No embedding model provided for SpeechBrain diarization.")
        return [], False

    try:
        import torchaudio
        import speechbrain as sb
        
        logger.info(f"Performing speaker diarization on {audio_file}")
        
        # First check if there's any speech in the file
        speech_detected, speech_duration = detect_speech_presence(audio_file, min_speech_seconds=min_speech_seconds)
        
        if not speech_detected:
            logger.warning(f"No significant speech detected in {audio_file}, skipping diarization")
            return [], False
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # VAD to find speech segments
        logger.info("Detecting speech segments...")
        
        try:
            # Try to use SpeechBrain's VAD
            from speechbrain.pretrained import VAD
            vad_model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                savedir=os.path.expanduser("~/.cache/speechbrain/vad-crdnn-libriparty")
            )
            
            # Get speech/non-speech probabilities
            speech_prob = vad_model.get_speech_prob_chunk(waveform)
            
            # Get speech segments with default threshold (0.5)
            speech_segments = vad_model.get_speech_segments(speech_prob)
            
            # Convert to list of (start, end) times
            vad_segments = [(start.item(), end.item()) for start, end in speech_segments]
            
            if not vad_segments:
                # If no segments detected, use the whole audio
                logger.warning("No speech detected by VAD, using the whole audio")
                vad_segments = [(0.0, waveform.shape[1] / sample_rate)]
                
            logger.info(f"VAD found {len(vad_segments)} speech segments")
            
        except Exception as e:
            logger.warning(f"Could not use SpeechBrain VAD, using fixed segmentation: {e}")
            # Use fixed segmentation as fallback
            segment_length_sec = 5.0
            overlap_sec = 1.0
            raw_segments = segment_audio(waveform, sample_rate, segment_length_sec, overlap_sec)
            vad_segments = [(start / sample_rate, end / sample_rate) for start, end in raw_segments]
        
        # Process each segment
        embeddings = []
        segment_times = []
        
        logger.info("Extracting speaker embeddings...")
        for i, (start_time, end_time) in enumerate(vad_segments):
            # Convert to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract segment
            segment = waveform[:, start_sample:end_sample]
            
            # Skip very short segments (less than 0.5 seconds)
            if segment.shape[1] < 0.5 * sample_rate:
                continue
            
            # Get embedding
            with torch.no_grad():
                try:
                    emb = embedding_model.encode_batch(segment)
                    embeddings.append(emb.squeeze(0))
                    segment_times.append((start_time, end_time))
                except Exception as e:
                    logger.warning(f"Error processing segment {i}: {e}")
        
        # Stack embeddings
        if not embeddings:
            logger.warning("No valid embeddings extracted")
            return [], True
            
        try:
            all_embeddings = torch.stack(embeddings)
        except RuntimeError as e:
            logger.error(f"Could not stack embeddings: {e}")
            # Try to handle different shaped embeddings by padding or trimming
            max_dim = max(emb.numel() for emb in embeddings)
            logger.info(f"Trying to reshape embeddings to consistent size: {max_dim}")
            
            reshaped_embeddings = []
            for emb in embeddings:
                # Flatten and pad or trim as needed
                flat_emb = emb.flatten()
                if flat_emb.numel() < max_dim:
                    # Pad
                    padded = torch.zeros(max_dim, device=flat_emb.device)
                    padded[:flat_emb.numel()] = flat_emb
                    reshaped_embeddings.append(padded)
                else:
                    # Trim
                    reshaped_embeddings.append(flat_emb[:max_dim])
            
            all_embeddings = torch.stack(reshaped_embeddings)
        
        # Cluster embeddings
        logger.info("Clustering speaker embeddings...")
        labels = cluster_embeddings(all_embeddings, min_speakers=min_speakers, max_speakers=max_speakers)
        
        # Create final segments
        final_segments = []
        for i, ((start_time, end_time), label) in enumerate(zip(segment_times, labels)):
            # Use "Person N" label for consistency with pyannote
            final_segments.append((start_time, end_time, f"Person {label+1}"))
            
        # Sort by start time
        final_segments.sort(key=lambda x: x[0])
        
        # Merge adjacent segments from same speaker
        merged_segments = []
        if final_segments:
            current_segment = list(final_segments[0])
            
            for start, end, speaker in final_segments[1:]:
                # If same speaker and close in time (< 1s gap)
                if speaker == current_segment[2] and start - current_segment[1] < 1.0:
                    # Extend current segment
                    current_segment[1] = end
                else:
                    # Save current segment and start a new one
                    merged_segments.append(tuple(current_segment))
                    current_segment = [start, end, speaker]
            
            # Add the last segment
            merged_segments.append(tuple(current_segment))
            
        # Before returning, verify we have valid segments
        if not merged_segments:
            logger.warning("No segments were created during diarization")
            return [], True  # Speech detected but no segments identified
            
        logger.info(f"Diarization complete. Found {len(merged_segments)} segments across {len(set([s[2] for s in merged_segments]))} speakers.")            
        return merged_segments, True
        
    except Exception as e:
        logger.error(f"Error during diarization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [], False

def merge_adjacent_segments(segments, max_gap_seconds=1.0):
    """
    Merge adjacent segments from the same speaker if they're close enough.
    
    Args:
        segments: List of (start, end, speaker_id) tuples
        max_gap_seconds: Maximum gap between segments to be merged
        
    Returns:
        List of merged segments as (start, end, speaker_id) tuples
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    # Initialize with the first segment
    merged_segments = [list(sorted_segments[0])]
    
    # Process the rest
    for start, end, speaker in sorted_segments[1:]:
        prev_start, prev_end, prev_speaker = merged_segments[-1]
        
        # If same speaker and close enough to previous segment, merge them
        if speaker == prev_speaker and start - prev_end <= max_gap_seconds:
            merged_segments[-1][1] = end  # Update end time of previous segment
        else:
            # Otherwise add as a new segment
            merged_segments.append([start, end, speaker])
    
    return [tuple(segment) for segment in merged_segments]

def extract_dialogue_segments(waveform, sample_rate, segments, output_dir, base_filename, save_speakers_separately=True):
    """
    Extract dialogue segments from waveform based on diarization results.
    
    Args:
        waveform: Audio waveform tensor [channels, samples]
        sample_rate: Sample rate of the audio
        segments: List of (start_time, end_time, speaker_id) tuples
        output_dir: Directory to save dialogue segments
        base_filename: Base filename for saved dialogues
        save_speakers_separately: If True, save each speaker's segments separately
                                  If False, save by dialogue blocks
                                  
    Returns:
        List of saved audio file paths
    """
    try:
        from .utils import ensure_dir_exists
        from .audio_io import save_audio_segment
        
        ensure_dir_exists(output_dir)
        saved_files = []
        
        # Check for empty segments list
        if not segments:
            logger.warning("No dialogue segments found to extract")
            return saved_files
            
        # Group segments by speaker if requested
        if save_speakers_separately:
            # Group by speaker ID
            speaker_segments = {}
            
            # Make sure each segment has all expected values before unpacking
            for segment in segments:
                # Check if this is a valid segment with 3 elements
                if not segment or len(segment) < 3:
                    logger.warning(f"Skipping invalid segment: {segment}")
                    continue
                    
                start, end, speaker = segment
                
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((start, end, speaker))
                
            # If no valid segments were found, return early
            if not speaker_segments:
                logger.warning("No valid dialogue segments to process")
                return saved_files
                
            # Merge adjacent segments for each speaker
            for speaker, spk_segments in speaker_segments.items():
                merged_segments = merge_adjacent_segments(spk_segments)
                
                # Create a separate file for each merged segment
                for i, (start, end, _) in enumerate(merged_segments):
                    # Convert timestamps to sample indices
                    start_sample = int(start * sample_rate)
                    end_sample = int(end * sample_rate)
                    
                    # Ensure indices are within bounds
                    if start_sample >= waveform.shape[1] or end_sample > waveform.shape[1]:
                        logger.warning(f"Segment {start}-{end} is out of bounds for waveform length {waveform.shape[1]/sample_rate}")
                        continue
                        
                    # Extract segment
                    segment_waveform = waveform[:, start_sample:end_sample]
                    
                    # Skip very short segments (less than 0.5 seconds)
                    if (end - start) < 0.5 or segment_waveform.shape[1] < sample_rate * 0.5:
                        continue
                        
                    # Generate output filename
                    output_filename = get_dialogue_filename(base_filename, i+1, speaker)
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save segment
                    file_path = save_audio_segment(segment_waveform, sample_rate, output_path, "mp3")
                    if file_path:
                        saved_files.append(file_path)
                        logger.info(f"Saved dialogue segment for speaker {speaker}: {file_path}")
        
        else:
            # Make sure we have valid segments before proceeding
            valid_segments = []
            for segment in segments:
                if segment and len(segment) >= 3:
                    valid_segments.append(segment)
                else:
                    logger.warning(f"Skipping invalid segment: {segment}")
            
            if not valid_segments:
                logger.warning("No valid dialogue segments to process")
                return saved_files
                
            # Merge adjacent segments regardless of speaker
            # First sort by start time
            sorted_segments = sorted(valid_segments, key=lambda x: x[0])
            
            # Group into dialogue blocks (separated by silences > 2 seconds)
            dialogue_blocks = []
            current_block = []
            
            for i, (start, end, speaker) in enumerate(sorted_segments):
                if i > 0 and start - sorted_segments[i-1][1] > 2.0:
                    # Gap > 2 seconds, start a new dialogue block
                    if current_block:
                        dialogue_blocks.append(current_block)
                    current_block = [(start, end, speaker)]
                else:
                    # Continue current dialogue block
                    current_block.append((start, end, speaker))
            
            # Add the last block if it exists
            if current_block:
                dialogue_blocks.append(current_block)
                
            # Process each dialogue block
            for i, block in enumerate(dialogue_blocks):
                if not block:
                    continue
                    
                # Extract start and end time for the whole block
                block_start = min(segment[0] for segment in block)
                block_end = max(segment[1] for segment in block)
                
                # Convert to sample indices
                start_sample = int(block_start * sample_rate)
                end_sample = int(block_end * sample_rate)
                
                # Ensure indices are within bounds
                if start_sample >= waveform.shape[1] or end_sample > waveform.shape[1]:
                    logger.warning(f"Block {block_start}-{block_end} is out of bounds")
                    continue
                
                # Extract segment
                segment_waveform = waveform[:, start_sample:end_sample]
                
                # Skip very short dialogues
                if (block_end - block_start) < 1.0 or segment_waveform.shape[1] < sample_rate * 1.0:
                    continue
                
                # Generate output filename
                output_filename = f"{base_filename}_dialogue_{i+1}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save segment
                file_path = save_audio_segment(segment_waveform, sample_rate, output_path, "mp3")
                if file_path:
                    saved_files.append(file_path)
                    logger.info(f"Saved dialogue block {i+1}: {file_path}")
        
        return saved_files
        
    except Exception as e:
        logger.error(f"Error extracting dialogue segments: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def get_dialogue_filename(base_filename, dialogue_num, speaker_id=None):
    """
    Generate filename for a dialogue segment.
    
    Args:
        base_filename: Base filename
        dialogue_num: Dialogue number
        speaker_id: Speaker ID (if None, don't include in filename)
    
    Returns:
        Generated filename without extension
    """
    if speaker_id is not None:
        # Include speaker ID in filename if provided
        return f"{base_filename}_dialogue_{dialogue_num}_speaker_{speaker_id}"
    else:
        # Otherwise just use dialogue number
        return f"{base_filename}_dialogue_{dialogue_num}"
