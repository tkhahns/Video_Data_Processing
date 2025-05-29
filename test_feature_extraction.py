#!/usr/bin/env python3

"""
Test script for the output_features module.
This will generate a sample audio file and extract features from it.
"""

import os
import numpy as np
import librosa
from src.output_features.main import process_audio_file
from src.output_features.extractors.audio_basic import AudioBasicExtractor
from src.output_features.extractors.speech_emotion import SpeechEmotionExtractor
from src.output_features.extractors.spectral_features import SpectralFeaturesExtractor
import pandas as pd

def generate_test_audio(output_path, duration=5.0, sample_rate=16000):
    """Generate a test audio file with a simple tone."""
    # Generate time array
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Generate a tone with varying frequency
    freq = 220 + 220 * np.sin(2 * np.pi * 0.2 * t)
    signal = np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    noise = np.random.normal(0, 0.01, signal.shape)
    signal = signal + noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Save the file
    librosa.output.write_wav(output_path, signal, sample_rate)
    print(f"Generated test audio file: {output_path}")
    
    return signal, sample_rate

def main():
    """Run a test of the feature extraction pipeline."""
    # Create output directory
    output_dir = "output/test_features"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate test audio file
    test_audio_path = os.path.join(output_dir, "test_tone.wav")
    audio_data, sample_rate = generate_test_audio(test_audio_path)
    
    # Initialize extractors
    extractors = [
        AudioBasicExtractor(),
        SpeechEmotionExtractor(),
        SpectralFeaturesExtractor()
    ]
    
    # Process the file
    features = process_audio_file(test_audio_path, extractors, output_dir)
    
    # Print the extracted features
    print("\nExtracted features:")
    for key, value in features.items():
        print(f"{key}: {value}")
    
    # Save features to CSV
    df = pd.DataFrame([features])
    csv_path = os.path.join(output_dir, "test_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFeatures saved to: {csv_path}")
    
if __name__ == "__main__":
    main()
