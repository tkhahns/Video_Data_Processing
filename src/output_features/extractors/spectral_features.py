"""
Spectral features extractor using librosa.
"""
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import logging
from ..extractors import BaseExtractor
from .. import utils

logger = logging.getLogger(__name__)

class SpectralFeaturesExtractor(BaseExtractor):
    """Extractor for spectral features, pitch, and rhythm."""
    
    def __init__(self):
        super().__init__("Spectral Features")
        
    def extract(self, audio_data, sample_rate):
        """
        Extract spectral features using librosa.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            
        Returns:
            dict: Dictionary with extracted features
        """
        features = {}
        
        # Define the hop length for frame-wise analysis
        hop_length = 512
        
        # --- Spectral Centroid ---
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate, hop_length=hop_length
        )[0]
        features["lbrs_spectral_centroid"] = float(np.mean(spectral_centroid))
        
        # --- Spectral Bandwidth ---
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_data, sr=sample_rate, hop_length=hop_length
        )[0]
        features["lbrs_spectral_bandwidth"] = float(np.mean(spectral_bandwidth))
        
        # --- Spectral Flatness ---
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio_data, hop_length=hop_length
        )[0]
        features["lbrs_spectral_flatness"] = float(np.mean(spectral_flatness))
        features["lbrs_spectral_flatness_singlevalue"] = float(np.mean(spectral_flatness))
        
        # --- Spectral Rolloff ---
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate, hop_length=hop_length
        )[0]
        features["lbrs_spectral_rolloff"] = float(np.mean(spectral_rolloff))
        
        # --- Zero Crossing Rate ---
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio_data, hop_length=hop_length
        )[0]
        features["lbrs_zero_crossing_rate"] = float(np.mean(zero_crossing_rate))
        features["lbrs_zero_crossing_rate_singlevalue"] = float(np.mean(zero_crossing_rate))
        
        # --- Root Mean Square Energy ---
        rmse = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        features["lbrs_rmse"] = float(np.mean(rmse))
        features["lbrs_rmse_singlevalue"] = float(np.mean(rmse))
        
        # --- Tempo Estimation ---
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]
        features["lbrs_tempo"] = float(tempo)
        features["lbrs_tempo_singlevalue"] = float(tempo)
        
        # --- Spectral Contrast ---
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio_data, sr=sample_rate, hop_length=hop_length
        )
        features["lbrs_spectral_contrast_singlevalue"] = float(np.mean(spectral_contrast))
        
        # Store for visualization
        self.spectral_centroid = spectral_centroid
        self.spectral_bandwidth = spectral_bandwidth
        self.spectral_flatness = spectral_flatness
        self.spectral_rolloff = spectral_rolloff
        self.zero_crossing_rate = zero_crossing_rate
        self.rmse = rmse
        self.tempo = tempo
        self.timestamps = utils.get_timestamps(spectral_centroid, hop_length, sample_rate)
        
        return features
        
    def save_output(self, audio_data, sample_rate, features, output_prefix):
        """
        Save visualizations of spectral features.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            features (dict): Extracted features
            output_prefix (str): Prefix for output files
        """
        try:
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
            
            # Plot spectral centroid
            axes[0].plot(self.timestamps, self.spectral_centroid)
            axes[0].set_ylabel('Frequency (Hz)')
            axes[0].set_title('Spectral Centroid')
            axes[0].grid(True)
            
            # Plot spectral bandwidth
            axes[1].plot(self.timestamps, self.spectral_bandwidth)
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].set_title('Spectral Bandwidth')
            axes[1].grid(True)
            
            # Plot spectral flatness
            axes[2].plot(self.timestamps, self.spectral_flatness)
            axes[2].set_ylabel('Flatness')
            axes[2].set_title('Spectral Flatness')
            axes[2].grid(True)
            
            # Plot spectral rolloff
            axes[3].plot(self.timestamps, self.spectral_rolloff)
            axes[3].set_ylabel('Frequency (Hz)')
            axes[3].set_title('Spectral Rolloff')
            axes[3].grid(True)
            
            # Plot zero crossing rate
            axes[4].plot(self.timestamps, self.zero_crossing_rate)
            axes[4].set_ylabel('Rate')
            axes[4].set_xlabel('Time (s)')
            axes[4].set_title('Zero Crossing Rate')
            axes[4].grid(True)
            
            # Add text annotation with overall stats
            text = (f"Mean Centroid: {features['lbrs_spectral_centroid']:.1f} Hz\n"
                   f"Mean Bandwidth: {features['lbrs_spectral_bandwidth']:.1f} Hz\n"
                   f"Mean Flatness: {features['lbrs_spectral_flatness']:.4f}\n"
                   f"Mean Rolloff: {features['lbrs_spectral_rolloff']:.1f} Hz\n"
                   f"Mean Zero Crossing Rate: {features['lbrs_zero_crossing_rate']:.4f}\n"
                   f"Mean RMSE: {features['lbrs_rmse']:.4f}\n"
                   f"Tempo: {features['lbrs_tempo']:.1f} BPM")
            plt.figtext(0.02, 0.02, text, fontsize=9, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_spectral.png")
            plt.close()
            
            # Save raw data as CSV for further analysis
            df = pd.DataFrame({
                'time': self.timestamps,
                'spectral_centroid': self.spectral_centroid,
                'spectral_bandwidth': self.spectral_bandwidth,
                'spectral_flatness': self.spectral_flatness,
                'spectral_rolloff': self.spectral_rolloff,
                'zero_crossing_rate': self.zero_crossing_rate,
                'rmse': self.rmse
            })
            df.to_csv(f"{output_prefix}_spectral.csv", index=False)
            
        except Exception as e:
            logger.error(f"Error saving spectral feature visualizations: {e}")
