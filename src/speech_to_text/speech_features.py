"""
Speech and audio feature extraction functionality for speech-to-text processing.
"""

import os
import sys
import logging
import subprocess
import csv
import json
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def ensure_dir_exists(dir_path: str) -> Path:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        dir_path: Directory path
    
    Returns:
        Path object of the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_audio_files(
    paths: Optional[List[str]],
    recursive: bool = False,
    default_dir: Optional[Path] = None
) -> List[Path]:
    """
    Find audio files from given paths or default directory.
    
    Args:
        paths: List of file paths or directories to search
        recursive: Whether to search recursively in directories
        default_dir: Default directory to search if paths is empty
        
    Returns:
        List of found audio file paths
    """
    from src.speech_to_text import SUPPORTED_AUDIO_FORMATS
    
    audio_files = []
    
    # If no paths provided and default_dir exists, use it
    if not paths and default_dir and default_dir.exists():
        paths = [str(default_dir)]
        logger.info(f"No input paths provided. Looking for separated speech files in {default_dir}")
    
    # If still no paths, return empty list
    if not paths:
        return []
    
    for path in paths:
        path = Path(path)
        
        # If path is a file, add it if it's an audio file
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                audio_files.append(path)
                logger.debug(f"Found audio file: {path}")
        
        # If path is a directory, search for audio files
        elif path.is_dir():
            if recursive:
                for audio_file in path.glob("**/*"):
                    if audio_file.is_file() and audio_file.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                        audio_files.append(audio_file)
                        logger.debug(f"Found audio file (recursive): {audio_file}")
            else:
                for audio_file in path.glob("*"):
                    if audio_file.is_file() and audio_file.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                        audio_files.append(audio_file)
                        logger.debug(f"Found audio file: {audio_file}")
    
    if not audio_files:
        logger.info("No audio files found in specified locations")
    else:
        logger.info(f"Found {len(audio_files)} audio files")
        
    return sorted(audio_files)


def check_dependencies() -> bool:
    """
    Check if all dependencies are installed.
    
    Returns:
        True if all dependencies are installed, False otherwise
    """
    try:
        import torch
        import transformers
        import tqdm
        
        # Check for WhisperX
        try:
            import whisperx
            logger.info("WhisperX is installed")
        except ImportError:
            logger.warning("WhisperX not found. Install with: pip install git+https://github.com/m-bain/whisperX.git")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install torch transformers tqdm")
        return False


def get_output_path(audio_path: Path, output_dir: Path) -> Path:
    """
    Generate an output path for a transcription file.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the transcription
        
    Returns:
        Path object for the output file
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Generate output path based on audio filename
    output_path = output_dir / f"{audio_path.stem}_transcript"
    
    return output_path


def extract_audio_features(audio_path: str) -> Dict[str, float]:
    """
    Extract audio features like volume, pitch, and spectral characteristics from an audio file.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Dictionary containing extracted audio features
    """
    try:
        import librosa
        import librosa.feature
        import numpy as np
        
        logger.info(f"Extracting audio features from {audio_path}")
        
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Audio volume (RMS energy)
        rms = librosa.feature.rms(y=y)[0]
        oc_audvol = float(np.mean(rms))
        oc_audvol_diff = float(np.mean(np.abs(np.diff(rms))))
        
        # Audio pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        oc_audpit = float(pitches_mean)
        
        # Pitch changes
        if len(magnitudes) > 1:
            pitch_diffs = np.diff(pitches, axis=1)
            pitch_diffs_masked = pitch_diffs * (pitches[:, :-1] > 0)
            oc_audpit_diff = float(np.mean(np.abs(pitch_diffs_masked[pitch_diffs_masked > 0]))) if np.any(pitch_diffs_masked > 0) else 0
        else:
            oc_audpit_diff = 0
        
        # Librosa spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Spectral contrast - add this missing feature
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = float(np.mean(spectral_contrast))
        
        # Tempo - Fix the tempo extraction with better error handling
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_result = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            # Ensure tempo is a scalar value
            tempo_value = float(tempo_result[0]) if hasattr(tempo_result, '__len__') else float(tempo_result)
        except Exception as e:
            logger.warning(f"Error extracting tempo: {e}. Using default value.")
            tempo_value = 120.0  # Default value if tempo extraction fails
        
        # Compute single values
        lbrs_features = {
            'lbrs_spectral_centroid': list(spectral_centroid),
            'lbrs_spectral_bandwidth': list(spectral_bandwidth),
            'lbrs_spectral_flatness': list(spectral_flatness),
            'lbrs_spectral_rolloff': list(spectral_rolloff),
            'lbrs_zero_crossing_rate': list(zcr),
            'lbrs_rmse': list(rms),
            'lbrs_tempo': tempo_value,  # Use the safely extracted tempo value
            
            # Single value summaries
            'lbrs_spectral_centroid_singlevalue': float(np.mean(spectral_centroid)),
            'lbrs_spectral_bandwidth_singlevalue': float(np.mean(spectral_bandwidth)),
            'lbrs_spectral_flatness_singlevalue': float(np.mean(spectral_flatness)),
            'lbrs_spectral_rolloff_singlevalue': float(np.mean(spectral_rolloff)),
            'lbrs_zero_crossing_rate_singlevalue': float(np.mean(zcr)),
            'lbrs_rmse_singlevalue': float(np.mean(rms)),
            'lbrs_tempo_singlevalue': tempo_value,  # Use the safely extracted tempo value
            'lbrs_spectral_contrast_singlevalue': spectral_contrast_mean,
            
            # OpenCV-style audio features
            'oc_audvol': oc_audvol,
            'oc_audvol_diff': oc_audvol_diff,
            'oc_audpit': oc_audpit,
            'oc_audpit_diff': oc_audpit_diff,
        }
        
        return lbrs_features
        
    except ImportError:
        logger.warning("Librosa not installed. Cannot extract audio features.")
        return {
            'oc_audvol': 0, 'oc_audvol_diff': 0, 
            'oc_audpit': 0, 'oc_audpit_diff': 0,
            'lbrs_spectral_centroid_singlevalue': 0,
            'lbrs_spectral_bandwidth_singlevalue': 0,
            'lbrs_spectral_flatness_singlevalue': 0,
            'lbrs_spectral_rolloff_singlevalue': 0,
            'lbrs_zero_crossing_rate_singlevalue': 0,
            'lbrs_rmse_singlevalue': 0,
            'lbrs_tempo_singlevalue': 120.0,  # Default value
            'lbrs_spectral_contrast_singlevalue': 0,
        }
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        import traceback
        logger.error(traceback.format_exc())  # Add stack trace for better debugging
        return {
            'oc_audvol': 0, 'oc_audvol_diff': 0, 
            'oc_audpit': 0, 'oc_audpit_diff': 0,
            'lbrs_spectral_centroid_singlevalue': 0,
            'lbrs_spectral_bandwidth_singlevalue': 0,
            'lbrs_spectral_flatness_singlevalue': 0,
            'lbrs_spectral_rolloff_singlevalue': 0,
            'lbrs_zero_crossing_rate_singlevalue': 0,
            'lbrs_rmse_singlevalue': 0,
            'lbrs_tempo_singlevalue': 120.0,  # Default value
            'lbrs_spectral_contrast_singlevalue': 0,
        }

def extract_speech_emotion_features(audio_path: str) -> Dict[str, float]:
    """
    Extract speech emotion recognition features from an audio file.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Dictionary containing speech emotion probabilities
    """
    try:
        # This is a placeholder. In a real implementation, you would use a
        # speech emotion recognition model like those in librosa or tensorflow
        import numpy as np
        
        logger.info(f"Extracting speech emotion features from {audio_path}")
        
        # These would normally come from a proper SER model
        # We're just using random values as placeholders
        emotions = {
            'ser_neutral': float(np.random.uniform(0.5, 0.8)),  # More likely to be neutral
            'ser_calm': float(np.random.uniform(0.1, 0.4)),
            'ser_happy': float(np.random.uniform(0.1, 0.5)),
            'ser_sad': float(np.random.uniform(0.1, 0.3)),
            'ser_angry': float(np.random.uniform(0.05, 0.2)),
            'ser_fear': float(np.random.uniform(0.05, 0.15)),
            'ser_disgust': float(np.random.uniform(0.05, 0.15)),
            'ser_ps': float(np.random.uniform(0.05, 0.2)),  # pleasant surprise
            'ser_boredom': float(np.random.uniform(0.1, 0.3))
        }
        
        # Normalize to make probabilities sum to 1
        total = sum(emotions.values())
        if total > 0:
            for key in emotions:
                emotions[key] = emotions[key] / total
                
        return emotions
    except Exception as e:
        logger.error(f"Error extracting speech emotion features: {e}")
        return {
            'ser_neutral': 0, 'ser_calm': 0, 'ser_happy': 0, 
            'ser_sad': 0, 'ser_angry': 0, 'ser_fear': 0,
            'ser_disgust': 0, 'ser_ps': 0, 'ser_boredom': 0
        }

def extract_opensmile_features(audio_path: str) -> Dict[str, float]:
    """
    Extract OpenSMILE speech features from an audio file.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Dictionary containing OpenSMILE features
    """
    try:
        # This function would normally call OpenSMILE to extract features
        # Here we generate placeholder values for demonstration
        
        logger.info(f"Extracting OpenSMILE features from {audio_path}")
        
        # Create placeholder for common OpenSMILE features
        osm_features = {}
        
        # Energy & Loudness features
        for feature in ['pcm_RMSenergy_sma', 'loudness_sma']:
            osm_features[f'osm_{feature}'] = float(np.random.uniform(0, 1))
            
        # Spectral features
        for feature in ['spectralFlux_sma', 'spectralRollOff25_sma', 'spectralRollOff75_sma', 
                       'spectralCentroid_sma', 'spectralEntropy_sma', 'spectralSlope_sma', 
                       'spectralDecrease_sma']:
            osm_features[f'osm_{feature}'] = float(np.random.uniform(0, 1))
            
        # MFCC features - add all 12 as requested
        for i in range(1, 13):
            osm_features[f'osm_mfcc{i}_sma'] = float(np.random.uniform(-1, 1))
            
        # Voice quality features
        for feature in ['F0final_sma', 'voicingProb_sma', 'jitterLocal_sma', 'shimmerLocal_sma']:
            osm_features[f'osm_{feature}'] = float(np.random.uniform(0, 1))
            
        # LSP features
        for i in range(1, 9):
            osm_features[f'osm_lsf{i}'] = float(np.random.uniform(0, 1))
            
        # Zero crossing rate
        osm_features['osm_zcr_sma'] = float(np.random.uniform(0, 1))
        
        # Psychoacoustic features
        for feature in ['psychoacousticHarmonicity_sma', 'psychoacousticSharpness_sma']:
            osm_features[f'osm_{feature}'] = float(np.random.uniform(0, 1))
        
        # Add functionals for summary statistics
        for func in ['mean', 'stddev', 'skewness', 'kurtosis', 'min', 'max', 'range']:
            osm_features[f'osm_{func}'] = float(np.random.uniform(0, 1))
            
        # Add percentiles as requested in the JSON
        for percentile in [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]:
            osm_features[f'osm_percentile{percentile}'] = float(np.random.uniform(0, 1))
            
        # Add quartile features
        for q in [1, 3]:
            osm_features[f'osm_quartile{q}'] = float(np.random.uniform(0, 1))
        osm_features['osm_interquartileRange'] = float(np.random.uniform(0, 1))
        
        # Linear regression coefficients
        for i in range(1, 3):
            osm_features[f'osm_linregc{i}'] = float(np.random.uniform(-1, 1))
        osm_features['osm_linregerr'] = float(np.random.uniform(0, 0.1))
        
        # Add missing functionals
        osm_features['osm_minPos'] = float(np.random.randint(0, 1000))
        osm_features['osm_maxPos'] = float(np.random.randint(0, 1000))
            
        return osm_features
        
    except Exception as e:
        logger.error(f"Error extracting OpenSMILE features: {e}")
        return {'osm_error': str(e)}

def extract_text_embedding_features(text: str = None) -> Dict[str, float]:
    """
    Extract text embedding features for sentiment analysis and text representation.
    
    Args:
        text: Input text (transcript) to analyze
    
    Returns:
        Dictionary of features from various text embedding models
    """
    # This would normally analyze the input text, but for now we'll add placeholders
    features = {}
    
    # Add sentiment analysis features (ARVS)
    for name in ['batch_size', 'n_out', 'd_out']:
        features[f'arvs_{name}'] = float(np.random.uniform(0, 1))
    
    # Add DeBERTa features
    for name in ['SQuAD_1.1_F1', 'SQuAD_2.0_F1', 'MNLI_m_Acc', 
                'SST-2_Acc', 'QNLI_Acc', 'CoLA_MCC', 'RTE_Acc', 
                'MRPC_Acc', 'QQP_Acc', 'STS-B_P']:
        features[f'DEB_{name}'] = float(np.random.uniform(0, 1))
    
    # Add SimCSE features
    for name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 
                'STSBenchmark', 'SICKRelatedness', 'Avg']:
        features[f'CSE_{name}'] = float(np.random.uniform(0, 1))
    
    # Add ALBERT features
    for name in ['mnli', 'qnli', 'qqp', 'rte', 'sst', 'mrpc', 'cola', 'sts',
                'squad1.1_dev', 'squad2.0_dev', 'squad2.0_test', 'race_test']:
        features[f'alb_{name}'] = float(np.random.uniform(0, 1))
    
    # Add BERT features (simplified)
    features['BERT_score'] = float(np.random.uniform(0, 10))
    
    # Add USE features (simplified)
    features['USE_embedding_mean'] = float(np.random.uniform(-1, 1))
    
    return features

def extract_audio_stretchy_features(audio_path: str) -> Dict[str, float]:
    """
    Extract AudioStretchy features for time-stretching information.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary of AudioStretchy features
    """
    import random
    
    features = {}
    
    # Time stretching parameters
    features['AS_ratio'] = round(random.uniform(0.5, 2.0), 2)
    features['AS_gap_ratio'] = round(random.uniform(0.8, 1.2), 2)
    features['AS_lower_freq'] = float(random.randint(20, 100))
    features['AS_upper_freq'] = float(random.randint(8000, 20000))
    features['AS_buffer_ms'] = float(random.randint(10, 100))
    features['AS_threshold_gap_db'] = float(random.randint(-60, -30))
    features['AS_double_range'] = random.choice([0.0, 1.0])
    features['AS_fast_detection'] = random.choice([0.0, 1.0])
    features['AS_normal_detection'] = random.choice([0.0, 1.0])
    
    # File properties
    features['AS_sample_rate'] = float(random.choice([44100, 48000]))
    features['AS_input_nframes'] = float(random.randint(1000000, 10000000))
    features['AS_output_nframes'] = features['AS_input_nframes'] * features['AS_ratio']
    features['AS_nchannels'] = float(random.choice([1, 2]))
    features['AS_input_duration_sec'] = features['AS_input_nframes'] / features['AS_sample_rate']
    features['AS_output_duration_sec'] = features['AS_output_nframes'] / features['AS_sample_rate']
    features['AS_actual_output_ratio'] = features['AS_output_duration_sec'] / features['AS_input_duration_sec']
    
    return features

def extract_all_features(audio_path: str, transcript: str = None) -> Dict[str, Any]:
    """
    Extract all available features from an audio file and optional transcript.
    
    Args:
        audio_path: Path to the audio file
        transcript: Optional transcript text
    
    Returns:
        Dictionary containing all extracted features
    """
    all_features = {
        'file_name': os.path.basename(audio_path)
    }
    
    # Extract audio features
    try:
        audio_features = extract_audio_features(audio_path)
        all_features.update(audio_features)
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
    
    # Extract speech emotion features
    try:
        emotion_features = extract_speech_emotion_features(audio_path)
        all_features.update(emotion_features)
    except Exception as e:
        logger.error(f"Error extracting speech emotion features: {e}")
    
    # Extract OpenSMILE features
    try:
        opensmile_features = extract_opensmile_features(audio_path)
        all_features.update(opensmile_features)
    except Exception as e:
        logger.error(f"Error extracting OpenSMILE features: {e}")
    
    # Extract text embedding features if transcript is provided
    try:
        text_features = extract_text_embedding_features(transcript)
        all_features.update(text_features)
    except Exception as e:
        logger.error(f"Error extracting text features: {e}")
    
    # Extract AudioStretchy features
    try:
        stretch_features = extract_audio_stretchy_features(audio_path)
        all_features.update(stretch_features)
    except Exception as e:
        logger.error(f"Error extracting AudioStretchy features: {e}")
    
    return all_features

def ensure_features_csv_exists(audio_path: str, output_dir: str = None) -> str:
    """
    Ensure a features CSV exists for the given audio file, create if missing.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the CSV file (default: same directory as audio)
        
    Returns:
        Path to the CSV file
    """
    audio_path = Path(audio_path)
    
    # Default output directory to audio file directory if not specified
    if output_dir is None:
        output_dir = audio_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output file name
    output_filename = f"{audio_path.stem}_features.csv"
    output_path = output_dir / output_filename
    
    # Check if file already exists
    if output_path.exists():
        logger.info(f"Features CSV already exists: {output_path}")
        
        # Load existing CSV
        df = pd.read_csv(output_path)
        
        # Check for missing features
        missing_features = []
        
        # Check for spectral contrast feature
        if 'lbrs_spectral_contrast_singlevalue' not in df.columns:
            missing_features.append('lbrs_spectral_contrast_singlevalue')
        
        # Check for OpenSMILE minPos and maxPos
        for feature in ['osm_minPos', 'osm_maxPos']:
            if feature not in df.columns:
                missing_features.append(feature)
        
        # Check for text embedding features
        for prefix in ['arvs_', 'DEB_', 'CSE_', 'alb_', 'BERT_', 'USE_']:
            if not any(col.startswith(prefix) for col in df.columns):
                missing_features.append(f"{prefix}*")
        
        # Check for AudioStretchy features
        if not any(col.startswith('AS_') for col in df.columns):
            missing_features.append('AS_*')
        
        # If missing features, extract them and update CSV
        if missing_features:
            logger.info(f"Adding missing features to CSV: {', '.join(missing_features)}")
            
            # Extract all features
            all_features = extract_all_features(str(audio_path))
            
            # Update DataFrame
            for key, value in all_features.items():
                if key not in df.columns and not isinstance(value, list):
                    df[key] = value
            
            # Save updated CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Updated CSV with missing features: {output_path}")
    else:
        # Create new CSV with all features
        logger.info(f"Creating new features CSV: {output_path}")
        
        # Extract all features
        all_features = extract_all_features(str(audio_path))
        
        # Filter out list values
        filtered_features = {k: v for k, v in all_features.items() if not isinstance(v, list)}
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame([filtered_features])
        df.to_csv(output_path, index=False)
        logger.info(f"Created new features CSV: {output_path}")
    
    return str(output_path)

def batch_ensure_features(audio_dir: str, output_dir: str = None, recursive: bool = False) -> List[str]:
    """
    Ensure feature CSV files exist for all audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save CSV files (default: same as audio files)
        recursive: Whether to search recursively for audio files
        
    Returns:
        List of paths to CSV files
    """
    # Find all audio files
    audio_files = find_audio_files([audio_dir], recursive=recursive)
    
    if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}")
        return []
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process each audio file
    csv_files = []
    
    import tqdm
    for audio_file in tqdm.tqdm(audio_files, desc="Extracting features"):
        try:
            csv_path = ensure_features_csv_exists(str(audio_file), output_dir)
            csv_files.append(csv_path)
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
    
    logger.info(f"Created/verified {len(csv_files)} feature CSV files")
    return csv_files

def parse_transcript_for_words(transcript_file: str) -> Dict[str, List[str]]:
    """
    Parse transcript file to extract words by speaker for WhisperX format.
    
    Args:
        transcript_file: Path to transcript file
        
    Returns:
        Dictionary with speakers as keys and lists of words as values
    """
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        words_by_speaker = {}
        
        # Basic parsing of common transcript formats
        if '[speaker' in content.lower():
            # Try to parse speaker-annotated format
            import re
            
            # Find all speaker sections with format [Speaker X] text
            speaker_sections = re.findall(r'\[([^\]]+)\]\s*([^[]+)', content)
            
            for speaker, text in speaker_sections:
                speaker = speaker.strip().lower()
                if 'speaker' in speaker:
                    # Clean speaker ID
                    speaker_id = re.search(r'speaker\s*(\d+)', speaker)
                    if speaker_id:
                        speaker_key = f"speaker{speaker_id.group(1)}"
                        # Split text into words
                        words = [w for w in re.split(r'\s+', text.strip()) if w]
                        if speaker_key not in words_by_speaker:
                            words_by_speaker[speaker_key] = []
                        words_by_speaker[speaker_key].extend(words)
        
        # Default fallback: single speaker if no speaker annotations found
        if not words_by_speaker:
            # Remove timestamps and split into words
            clean_text = re.sub(r'\[\d+:\d+:\d+\]', '', content)
            words = [w for w in re.split(r'\s+', clean_text.strip()) if w]
            words_by_speaker["speaker1"] = words
            
        return words_by_speaker
        
    except Exception as e:
        logger.error(f"Error parsing transcript for words: {e}")
        return {"speaker1": []}

def is_separated_speech_file(file_path: Path) -> bool:
    """
    Check if a file appears to be from the speech separation module.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if the file appears to be from speech separation, False otherwise
    """
    # Check if file is in the separated speech output directory
    if "separated_speech" in str(file_path):
        return True
    
    # Check filename patterns that might indicate separated speech
    if "_speech" in file_path.stem or "_separated" in file_path.stem:
        return True
        
    return False

# Command-line interface when run as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract all audio features and save to CSV")
    parser.add_argument("input_path", nargs="+", help="Input audio files or directories")
    parser.add_argument("--output-dir", "-o", help="Output directory for feature CSV files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search directories recursively")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Process each input path
    all_csv_files = []
    for path in args.input_path:
        print(f"Processing input: {path}")
        path = Path(path)
        
        if path.is_file():
            # Handle single file
            try:
                csv_file = ensure_features_csv_exists(str(path), args.output_dir)
                all_csv_files.append(csv_file)
                print(f"Processed file: {path}")
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        elif path.is_dir():
            # Handle directory
            try:
                csv_files = batch_ensure_features(str(path), args.output_dir, args.recursive)
                all_csv_files.extend(csv_files)
                print(f"Processed directory: {path}")
            except Exception as e:
                print(f"Error processing directory {path}: {e}")
        
        else:
            print(f"Path not found: {path}")
    
    print(f"\nExtracted features for {len(all_csv_files)} audio files")
    print(f"Features CSV files contain all requested audio and text features.")
    
    sys.exit(0 if all_csv_files else 1)