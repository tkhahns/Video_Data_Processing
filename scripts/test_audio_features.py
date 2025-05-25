
#!/usr/bin/env python3

import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath("."))

from src.speech_to_text.utils import extract_audio_features, extract_speech_emotion_features, extract_opensmile_features

def test_audio_file(audio_path):
    """Test all audio feature extraction functions on a single file."""
    print(f"Testing audio features for: {audio_path}")
    
    print("\n=== Basic Audio Features ===")
    audio_features = extract_audio_features(audio_path)
    for key, value in audio_features.items():
        if not isinstance(value, list):  # Only print scalar values
            print(f"{key}: {value}")
    
    print("\n=== Speech Emotion Features ===")
    emotion_features = extract_speech_emotion_features(audio_path)
    for key, value in emotion_features.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== OpenSMILE Features (sample) ===")
    smile_features = extract_opensmile_features(audio_path)
    # Print just a few example features from each category
    categories = {
        "Energy": ['osm_pcm_RMSenergy_sma', 'osm_loudness_sma'],
        "Spectral": ['osm_spectralCentroid_sma', 'osm_spectralEntropy_sma'],
        "MFCC": ['osm_mfcc1_sma', 'osm_mfcc2_sma'],
        "Voice": ['osm_F0final_sma', 'osm_voicingProb_sma'],
        "Stats": ['osm_mean', 'osm_stddev']
    }
    
    for category, features in categories.items():
        print(f"  {category} features:")
        for key in features:
            if key in smile_features:
                print(f"    {key}: {smile_features[key]}")
    
    print(f"\nTotal OpenSMILE features: {len(smile_features)}")
    
    return {**audio_features, **emotion_features, **smile_features}

def write_features_csv(audio_path, output_csv):
    """Extract all features and write to CSV for inspection."""
    print(f"Extracting all features from {audio_path} to {output_csv}")
    
    all_features = test_audio_file(audio_path)
    
    # Convert to DataFrame and save
    import pandas as pd
    df = pd.DataFrame([all_features])
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(all_features)} features to {output_csv}")
    
    return all_features

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_audio_features.py <audio_file_path> [output_csv]")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        # Write all features to CSV if output path provided
        output_csv = sys.argv[2]
        write_features_csv(audio_path, output_csv)
    else:
        # Just print sample features
        test_audio_file(audio_path)
