"""
Main entry point for the output features extraction module.
Processes audio files to extract various features and compile them into CSV output.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import glob
import json

# Add parent directory to sys.path for imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import feature extractors
from src.output_features.extractors import (
    audio_basic,
    speech_emotion,
    speech_transcription,
    spectral_features
)
from src.output_features import utils

# Set up logging
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

def process_audio_file(audio_path, extractors, output_dir):
    """
    Process a single audio file with all configured extractors.
    
    Args:
        audio_path (str): Path to the audio file
        extractors (list): List of feature extractor instances
        output_dir (str): Directory to save individual extractor outputs
        
    Returns:
        dict: Dictionary of extracted features
    """
    logger.info(f"Processing {os.path.basename(audio_path)}")
    
    # Create a features dictionary
    features = {
        'file_name': os.path.basename(audio_path),
        'file_path': audio_path,
    }
    
    # Load the audio once
    try:
        audio_data, sample_rate = utils.load_audio(audio_path)
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        return None
    
    # Apply each extractor
    for extractor in extractors:
        try:
            logger.info(f"Extracting {extractor.name} features")
            extractor_features = extractor.extract(audio_data, sample_rate)
            
            # Only merge non-empty results
            if extractor_features:
                features.update(extractor_features)
                
                # Save individual extractor results if needed
                if output_dir and hasattr(extractor, 'save_output'):
                    file_base = os.path.splitext(os.path.basename(audio_path))[0]
                    extractor_output_dir = os.path.join(output_dir, extractor.name.lower().replace(' ', '_'))
                    os.makedirs(extractor_output_dir, exist_ok=True)
                    extractor.save_output(audio_data, sample_rate, extractor_features, 
                                          os.path.join(extractor_output_dir, file_base))
        except Exception as e:
            logger.error(f"Error extracting {extractor.name} features: {e}")
    
    return features

def main():
    """Main entry point for feature extraction."""
    parser = argparse.ArgumentParser(description='Extract features from audio files')
    parser.add_argument('--input-dir', required=True, 
                        help='Directory containing input audio files (wav files from separate_speech)')
    parser.add_argument('--output-dir', default='output/features', 
                        help='Directory to save output files')
    parser.add_argument('--output-csv', default='all_features.csv',
                        help='Filename for the combined CSV output')
    parser.add_argument('--feature-config', default=None,
                        help='Path to JSON file with feature configuration')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of WAV files
    if os.path.isdir(args.input_dir):
        audio_files = glob.glob(os.path.join(args.input_dir, '*.wav'))
        if not audio_files:
            logger.error(f"No WAV files found in {args.input_dir}")
            return 1
    else:
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
        
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Initialize feature extractors
    extractors = []
    
    # Basic audio features
    extractors.append(audio_basic.AudioBasicExtractor())
    
    # Speech emotion recognition
    extractors.append(speech_emotion.SpeechEmotionExtractor())
    
    # Spectral features
    extractors.append(spectral_features.SpectralFeaturesExtractor())
    
    # Speech transcription (if available)
    try:
        extractors.append(speech_transcription.SpeechTranscriptionExtractor())
    except Exception as e:
        logger.warning(f"Could not initialize speech transcription: {e}")
    
    # Process each file
    all_features = []
    for audio_file in audio_files:
        features = process_audio_file(audio_file, extractors, args.output_dir)
        if features:
            all_features.append(features)
    
    # Convert to DataFrame
    if all_features:
        df = pd.DataFrame(all_features)
        
        # Save to CSV
        csv_path = os.path.join(args.output_dir, args.output_csv)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved combined features to {csv_path}")
        
        # Also save as Excel for easier viewing
        excel_path = os.path.join(args.output_dir, args.output_csv.replace('.csv', '.xlsx'))
        df.to_excel(excel_path, index=False)
        logger.info(f"Saved features to Excel: {excel_path}")
    else:
        logger.warning("No features extracted")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
