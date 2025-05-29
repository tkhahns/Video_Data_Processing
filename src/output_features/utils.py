"""
Utility functions for the output_features module.
"""
import os
import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)

def load_audio(file_path):
    """
    Load an audio file using librosa.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        logger.debug(f"Loaded audio file {file_path}: {len(audio_data)/sample_rate:.2f}s @ {sample_rate}Hz")
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise

def get_audio_duration(audio_data, sample_rate):
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
        
    Returns:
        float: Duration in seconds
    """
    return len(audio_data) / sample_rate

def frame_to_time(frame_idx, hop_length, sample_rate):
    """
    Convert frame index to time in seconds.
    
    Args:
        frame_idx (int): Frame index
        hop_length (int): Hop length in samples
        sample_rate (int): Sample rate
        
    Returns:
        float: Time in seconds
    """
    return frame_idx * hop_length / sample_rate

def time_to_frame(time_sec, hop_length, sample_rate):
    """
    Convert time in seconds to frame index.
    
    Args:
        time_sec (float): Time in seconds
        hop_length (int): Hop length in samples
        sample_rate (int): Sample rate
        
    Returns:
        int: Frame index
    """
    return int(time_sec * sample_rate / hop_length)

def normalize_audio(audio_data):
    """
    Normalize audio data to have values between -1 and 1.
    
    Args:
        audio_data (np.ndarray): Audio data
        
    Returns:
        np.ndarray: Normalized audio data
    """
    if np.max(np.abs(audio_data)) > 0:
        return audio_data / np.max(np.abs(audio_data))
    return audio_data

def segment_audio(audio_data, sample_rate, segment_length_sec=3.0, overlap_sec=0.5):
    """
    Segment audio into fixed-length segments with overlap.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
        segment_length_sec (float): Segment length in seconds
        overlap_sec (float): Overlap between segments in seconds
        
    Returns:
        list: List of audio segments
    """
    segment_length_samples = int(segment_length_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    hop_length = segment_length_samples - overlap_samples
    
    segments = []
    
    for start in range(0, len(audio_data), hop_length):
        end = start + segment_length_samples
        if end > len(audio_data):
            # Pad with zeros if needed
            segment = np.zeros(segment_length_samples)
            segment[:len(audio_data) - start] = audio_data[start:]
        else:
            segment = audio_data[start:end]
        
        segments.append(segment)
        
        if end >= len(audio_data):
            break
    
    return segments

def find_voice_activity(audio_data, sample_rate, threshold=0.01):
    """
    Find segments with voice activity.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
        threshold (float): Energy threshold for voice activity
        
    Returns:
        list: List of (start_time, end_time) tuples in seconds
    """
    # Calculate short-time energy
    hop_length = int(0.01 * sample_rate)  # 10ms hop
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    
    # Compute energy
    energy = np.array([
        np.sum(audio_data[i:i+frame_length]**2) / frame_length
        for i in range(0, len(audio_data) - frame_length, hop_length)
    ])
    
    # Find segments above threshold
    is_speech = energy > threshold
    
    # Convert to time ranges
    speech_segments = []
    in_speech = False
    start_time = 0
    
    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            in_speech = True
            start_time = i * hop_length / sample_rate
        elif not speech and in_speech:
            in_speech = False
            end_time = i * hop_length / sample_rate
            speech_segments.append((start_time, end_time))
    
    # Handle the case where audio ends during speech
    if in_speech:
        end_time = len(audio_data) / sample_rate
        speech_segments.append((start_time, end_time))
    
    return speech_segments

def resample_audio(audio_data, orig_sr, target_sr):
    """
    Resample audio to a different sample rate.
    
    Args:
        audio_data (np.ndarray): Audio data
        orig_sr (int): Original sample rate
        target_sr (int): Target sample rate
        
    Returns:
        np.ndarray: Resampled audio data
    """
    if orig_sr == target_sr:
        return audio_data
    
    return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)

def get_timestamps(feature_data, hop_length, sample_rate):
    """
    Get timestamps for frame-based feature data.
    
    Args:
        feature_data (np.ndarray): Feature data
        hop_length (int): Hop length in samples
        sample_rate (int): Sample rate
        
    Returns:
        np.ndarray: Timestamps in seconds
    """
    num_frames = feature_data.shape[0]
    timestamps = np.arange(num_frames) * hop_length / sample_rate
    return timestamps
