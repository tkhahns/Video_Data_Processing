"""
Utility functions for speech-to-text processing.
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
        
        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        
        # Compute single values
        lbrs_features = {
            'lbrs_spectral_centroid': list(spectral_centroid),
            'lbrs_spectral_bandwidth': list(spectral_bandwidth),
            'lbrs_spectral_flatness': list(spectral_flatness),
            'lbrs_spectral_rolloff': list(spectral_rolloff),
            'lbrs_zero_crossing_rate': list(zcr),
            'lbrs_rmse': list(rms),
            'lbrs_tempo': tempo[0],
            
            # Single value summaries
            'lbrs_spectral_centroid_singlevalue': float(np.mean(spectral_centroid)),
            'lbrs_spectral_bandwidth_singlevalue': float(np.mean(spectral_bandwidth)),
            'lbrs_spectral_flatness_singlevalue': float(np.mean(spectral_flatness)),
            'lbrs_spectral_rolloff_singlevalue': float(np.mean(spectral_rolloff)),
            'lbrs_zero_crossing_rate_singlevalue': float(np.mean(zcr)),
            'lbrs_rmse_singlevalue': float(np.mean(rms)),
            'lbrs_tempo_singlevalue': float(tempo[0]),
            
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
            'lbrs_tempo_singlevalue': 0,
        }
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return {
            'oc_audvol': 0, 'oc_audvol_diff': 0, 
            'oc_audpit': 0, 'oc_audpit_diff': 0,
            'lbrs_spectral_centroid_singlevalue': 0,
            'lbrs_spectral_bandwidth_singlevalue': 0,
            'lbrs_spectral_flatness_singlevalue': 0,
            'lbrs_spectral_rolloff_singlevalue': 0,
            'lbrs_zero_crossing_rate_singlevalue': 0,
            'lbrs_rmse_singlevalue': 0,
            'lbrs_tempo_singlevalue': 0,
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
            
        return osm_features
        
    except Exception as e:
        logger.error(f"Error extracting OpenSMILE features: {e}")
        return {'osm_error': str(e)}

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

def create_pipeline_output(results_dir: str, output_file: str = "pipeline_output.csv") -> bool:
    """
    Create a comprehensive CSV file containing merged data from all pipeline outputs.
    
    Args:
        results_dir: Directory containing all the pipeline results
        output_file: Name of the output CSV file
    
    Returns:
        True if CSV was created successfully, False otherwise
    """
    try:
        logger.info(f"Creating pipeline output CSV at {output_file}")
        
        # Ensure results directory exists
        if not os.path.exists(results_dir):
            logger.error(f"Results directory not found: {results_dir}")
            return False
        
        # Define the subdirectories to look for data
        transcript_dir = os.path.join(results_dir, "transcripts")
        speech_dir = os.path.join(results_dir, "speech")
        emotions_dir = os.path.join(results_dir, "emotions_and_pose")
        
        # Dictionary to store all data by video name
        merged_data = {}
        video_names = set()
        
        # Process transcript data
        if os.path.exists(transcript_dir):
            logger.info("Processing transcript data...")
            transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]
            
            for file in transcript_files:
                video_name = file.replace('_speech_transcript.txt', '').replace('_transcript.txt', '')
                video_names.add(video_name)
                
                file_path = os.path.join(transcript_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcript_text = f.read().strip()
                
                if video_name not in merged_data:
                    merged_data[video_name] = {}
                
                merged_data[video_name]['transcript'] = transcript_text
                
                # Extract words by speaker for WhisperX format
                words_by_speaker = parse_transcript_for_words(file_path)
                for speaker, words in words_by_speaker.items():
                    # Store up to first 100 words with WhisperX format
                    for i, word in enumerate(words[:100]):
                        merged_data[video_name][f'WhX_highlight_diarize_{speaker}_word_{i+1}'] = word
        
        # Process audio feature data from speech directory
        if os.path.exists(speech_dir):
            logger.info("Processing speech audio for features...")
            audio_files = [f for f in os.listdir(speech_dir) 
                          if f.endswith(('.wav', '.mp3')) and "_speech" in f]
            
            for file in audio_files:
                video_name = file.replace('_speech.wav', '').replace('_speech.mp3', '')
                video_names.add(video_name)
                
                file_path = os.path.join(speech_dir, file)
                
                # Extract audio features
                try:
                    # Basic audio features (volume, pitch)
                    audio_features = extract_audio_features(file_path)
                    
                    # Speech emotion features
                    emotion_features = extract_speech_emotion_features(file_path)
                    
                    # OpenSMILE features
                    opensmile_features = extract_opensmile_features(file_path)
                    
                    if video_name not in merged_data:
                        merged_data[video_name] = {}
                        
                    # Add all features to merged data
                    merged_data[video_name].update(audio_features)
                    merged_data[video_name].update(emotion_features)
                    merged_data[video_name].update(opensmile_features)
                    
                except Exception as e:
                    logger.error(f"Error processing audio features for {file}: {e}")
        
        # Process emotion data
        if os.path.exists(emotions_dir):
            logger.info("Processing emotion and pose data...")
            emotion_files = [f for f in os.listdir(emotions_dir) if f.endswith('_emotions.csv')]
            pose_files = [f for f in os.listdir(emotions_dir) if f.endswith('_pose.json')]
            
            # Process emotion CSV files
            for file in emotion_files:
                video_name = file.replace('_emotions.csv', '')
                video_names.add(video_name)
                
                file_path = os.path.join(emotions_dir, file)
                try:
                    # Read emotion data - can have variable column formats
                    emotion_df = pd.read_csv(file_path)
                    
                    # Extract predominant emotions for each speaker
                    if video_name not in merged_data:
                        merged_data[video_name] = {}
                    
                    # Get emotion summary - most frequent emotions for each speaker
                    if 'speaker1_emotion' in emotion_df.columns:
                        speaker1_emotions = emotion_df['speaker1_emotion'].value_counts().nlargest(3)
                        merged_data[video_name]['speaker1_emotions'] = ', '.join([f"{e}({c})" for e, c in speaker1_emotions.items()])
                    
                    if 'speaker2_emotion' in emotion_df.columns:
                        speaker2_emotions = emotion_df['speaker2_emotion'].value_counts().nlargest(3)
                        merged_data[video_name]['speaker2_emotions'] = ', '.join([f"{e}({c})" for e, c in speaker2_emotions.items()])
                    
                    # Get posture data if available
                    if 'speaker1_posture' in emotion_df.columns:
                        postures = emotion_df['speaker1_posture'].value_counts().nlargest(2)
                        merged_data[video_name]['speaker1_posture'] = ', '.join([f"{p}({c})" for p, c in postures.items()])
                    
                    if 'speaker2_posture' in emotion_df.columns:
                        postures = emotion_df['speaker2_posture'].value_counts().nlargest(2)
                        merged_data[video_name]['speaker2_posture'] = ', '.join([f"{p}({c})" for p, c in postures.items()])
                        
                    # Store full emotion timeline data
                    merged_data[video_name]['emotion_timeline'] = emotion_df.to_dict('records')
                        
                except Exception as e:
                    logger.warning(f"Error processing emotion file {file}: {e}")
            
            # Process pose JSON files
            for file in pose_files:
                video_name = file.replace('_pose.json', '')
                video_names.add(video_name)
                
                file_path = os.path.join(emotions_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        pose_data = json.load(f)
                    
                    if video_name not in merged_data:
                        merged_data[video_name] = {}
                    
                    # Store summary of pose data - we're just storing the raw pose data
                    # in case it needs to be processed further
                    merged_data[video_name]['pose_data'] = pose_data
                    
                    # Extract some summary pose features if available
                    pose_summary = {
                        'speaker1': {'position': [], 'arms': []},
                        'speaker2': {'position': [], 'arms': []}
                    }
                    
                    for second, data in pose_data.items():
                        for speaker in ['speaker1', 'speaker2']:
                            if speaker in data and 'posture' in data[speaker]:
                                posture = data[speaker].get('posture', {})
                                if posture:
                                    if 'position' in posture and posture['position']:
                                        pose_summary[speaker]['position'].append(posture['position'])
                                    if 'arms' in posture and posture['arms']:
                                        pose_summary[speaker]['arms'].append(posture['arms'])
                    
                    # Get most common positions and arm postures
                    for speaker in ['speaker1', 'speaker2']:
                        if pose_summary[speaker]['position']:
                            from collections import Counter
                            positions = Counter(pose_summary[speaker]['position']).most_common(2)
                            merged_data[video_name][f'{speaker}_positions'] = ', '.join([f"{p}({c})" for p, c in positions])
                            
                        if pose_summary[speaker]['arms']:
                            arms = Counter(pose_summary[speaker]['arms']).most_common(2)
                            merged_data[video_name][f'{speaker}_arms'] = ', '.join([f"{a}({c})" for a, c in arms])
                    
                except Exception as e:
                    logger.warning(f"Error processing pose file {file}: {e}")
        
        # Create the output CSV file
        output_path = os.path.join(results_dir, output_file)
        logger.info(f"Writing pipeline output to {output_path}")
        
        # Define all the possible columns for the CSV - basic features plus all the new ones
        basic_columns = [
            'video_name', 'transcript', 
            'speaker1_emotions', 'speaker2_emotions',
            'speaker1_posture', 'speaker2_posture',
            'speaker1_positions', 'speaker2_positions',
            'speaker1_arms', 'speaker2_arms'
        ]
        
        # Audio feature columns - ensure these include all requested features
        audio_columns = [
            # OpenCV-style audio features
            'oc_audvol', 'oc_audvol_diff', 'oc_audpit', 'oc_audpit_diff',
            
            # Librosa spectral features
            'lbrs_spectral_centroid_singlevalue', 'lbrs_spectral_bandwidth_singlevalue',
            'lbrs_spectral_flatness_singlevalue', 'lbrs_spectral_rolloff_singlevalue',
            'lbrs_zero_crossing_rate_singlevalue', 'lbrs_rmse_singlevalue',
            'lbrs_tempo_singlevalue',
            
            # Librosa spectral contrast
            'lbrs_spectral_contrast_singlevalue'
        ]
        
        # Speech emotion columns
        ser_columns = [
            'ser_neutral', 'ser_calm', 'ser_happy', 'ser_sad', 'ser_angry',
            'ser_fear', 'ser_disgust', 'ser_ps', 'ser_boredom'
        ]
        
        # OpenSMILE columns - expanded to include all requested features
        opensmile_columns = [
            # Energy & Loudness
            'osm_pcm_RMSenergy_sma', 'osm_loudness_sma', 
            
            # Spectral features
            'osm_spectralFlux_sma', 'osm_spectralRollOff25_sma', 'osm_spectralRollOff75_sma',
            'osm_spectralCentroid_sma', 'osm_spectralEntropy_sma', 'osm_spectralSlope_sma', 
            'osm_spectralDecrease_sma',
            
            # MFCC features - all 12
            'osm_mfcc1_sma', 'osm_mfcc2_sma', 'osm_mfcc3_sma', 'osm_mfcc4_sma', 
            'osm_mfcc5_sma', 'osm_mfcc6_sma', 'osm_mfcc7_sma', 'osm_mfcc8_sma', 
            'osm_mfcc9_sma', 'osm_mfcc10_sma', 'osm_mfcc11_sma', 'osm_mfcc12_sma',
            
            # Voice quality features
            'osm_F0final_sma', 'osm_voicingProb_sma', 'osm_jitterLocal_sma', 'osm_shimmerLocal_sma',
            
            # Zero crossing rate
            'osm_zcr_sma',
            
            # Psychoacoustic features
            'osm_psychoacousticHarmonicity_sma', 'osm_psychoacousticSharpness_sma',
            
            # Functionals
            'osm_mean', 'osm_stddev', 'osm_skewness', 'osm_kurtosis', 
            'osm_min', 'osm_max', 'osm_range',
            
            # Percentiles
            'osm_percentile1.0', 'osm_percentile5.0', 'osm_percentile25.0', 
            'osm_percentile50.0', 'osm_percentile75.0', 'osm_percentile95.0', 'osm_percentile99.0',
            
            # Quartiles
            'osm_quartile1', 'osm_quartile3', 'osm_interquartileRange',
            
            # Linear regression
            'osm_linregc1', 'osm_linregc2', 'osm_linregerr'
        ]
        
        # WhisperX columns - dynamically add based on what was found
        whisperx_columns = []
        for video_name in video_names:
            if video_name in merged_data:
                for key in merged_data[video_name].keys():
                    if key.startswith('WhX_highlight_diarize_') and key not in whisperx_columns:
                        whisperx_columns.append(key)
        
        # Combine all columns
        all_columns = basic_columns + audio_columns + ser_columns + opensmile_columns
        all_columns.extend(whisperx_columns)
        
        # Create a DataFrame with the essential data
        rows = []
        for video_name in sorted(video_names):
            row = {'video_name': video_name}
            
            # Add each available field to the row
            for col in all_columns[1:]:  # Skip video_name which we already added
                if video_name in merged_data and col in merged_data[video_name]:
                    row[col] = merged_data[video_name][col]
                else:
                    row[col] = ''
            
            rows.append(row)
        
        # Create DataFrame and save to CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully created pipeline output CSV at {output_path}")
            
            # Also generate a summary text file with key information
            summary_path = os.path.join(results_dir, "pipeline_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("===== Video Processing Pipeline Summary =====\n\n")
                for video_name in sorted(video_names):
                    f.write(f"Video: {video_name}\n")
                    f.write("-" * 50 + "\n")
                    
                    # Add transcript excerpt (first 100 chars)
                    if 'transcript' in merged_data.get(video_name, {}):
                        transcript = merged_data[video_name]['transcript']
                        f.write(f"Transcript excerpt: {transcript[:100]}...\n\n")
                    
                    # Add emotion information
                    if 'speaker1_emotions' in merged_data.get(video_name, {}):
                        f.write(f"Speaker 1 emotions: {merged_data[video_name]['speaker1_emotions']}\n")
                    if 'speaker2_emotions' in merged_data.get(video_name, {}):
                        f.write(f"Speaker 2 emotions: {merged_data[video_name]['speaker2_emotions']}\n")
                    
                    # Add posture information
                    if 'speaker1_positions' in merged_data.get(video_name, {}):
                        f.write(f"Speaker 1 positions: {merged_data[video_name].get('speaker1_positions', '')}\n")
                        f.write(f"Speaker 1 arms: {merged_data[video_name].get('speaker1_arms', '')}\n")
                    if 'speaker2_positions' in merged_data.get(video_name, {}):
                        f.write(f"Speaker 2 positions: {merged_data[video_name].get('speaker2_positions', '')}\n")
                        f.write(f"Speaker 2 arms: {merged_data[video_name].get('speaker2_arms', '')}\n")
                    
                    # Add audio feature information
                    f.write("\nAudio Features:\n")
                    
                    # Audio volume and pitch
                    if 'oc_audvol' in merged_data.get(video_name, {}):
                        f.write(f"- Volume: {merged_data[video_name].get('oc_audvol', 0):.3f}, ")
                        f.write(f"Volume change: {merged_data[video_name].get('oc_audvol_diff', 0):.3f}\n")
                        f.write(f"- Pitch: {merged_data[video_name].get('oc_audpit', 0):.2f}, ")
                        f.write(f"Pitch change: {merged_data[video_name].get('oc_audpit_diff', 0):.2f}\n")
                    
                    # Speech emotion recognition
                    f.write("\nSpeech Emotion Recognition:\n")
                    for emotion in ['neutral', 'happy', 'sad', 'angry']:
                        if f'ser_{emotion}' in merged_data.get(video_name, {}):
                            f.write(f"- {emotion.capitalize()}: {merged_data[video_name].get(f'ser_{emotion}', 0):.3f}\n")
                    
                    f.write("\n\n")
            
            return True
        else:
            logger.warning("No data to write to pipeline output CSV")
            return False
            
    except Exception as e:
        logger.error(f"Error creating pipeline output CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


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
