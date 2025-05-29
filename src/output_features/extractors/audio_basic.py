"""
Basic audio feature extractor for volume, pitch, and their changes.
"""
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import logging
from ..extractors import BaseExtractor
from .. import utils

logger = logging.getLogger(__name__)

class AudioBasicExtractor(BaseExtractor):
    """Extractor for basic audio features like volume and pitch."""
    
    def __init__(self):
        super().__init__("Audio Basic")
        
    def extract(self, audio_data, sample_rate):
        """
        Extract basic audio features.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            
        Returns:
            dict: Dictionary with extracted features
        """
        features = {}
        
        # Define the hop length for frame-wise analysis (e.g., 10ms)
        hop_length = int(0.01 * sample_rate)
        
        # --- Volume (RMS Energy) ---
        # Calculate frame-wise RMS energy
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        
        # Calculate overall average volume
        features["oc_audvol"] = float(np.mean(rms))
        
        # Calculate frame-to-frame volume difference
        rms_diff = np.diff(rms)
        features["oc_audvol_diff"] = float(np.mean(np.abs(rms_diff)))
        
        # Store the full RMS curve for plotting
        self.rms = rms
        self.rms_diff = rms_diff
        
        # --- Pitch (Fundamental Frequency) ---
        # Calculate pitch using PYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            hop_length=hop_length
        )
        
        # Filter out unvoiced frames
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            # Calculate mean pitch (ignoring unvoiced frames)
            features["oc_audpit"] = float(np.nanmean(f0_voiced))
            
            # Calculate frame-to-frame pitch difference (only for voiced frames)
            f0_diff = np.diff(f0_voiced)
            if len(f0_diff) > 0:
                features["oc_audpit_diff"] = float(np.nanmean(np.abs(f0_diff)))
            else:
                features["oc_audpit_diff"] = 0.0
        else:
            # Handle case with no voiced frames
            features["oc_audpit"] = 0.0
            features["oc_audpit_diff"] = 0.0
        
        # Store the full pitch curve for plotting
        self.f0 = f0
        self.voiced_flag = voiced_flag
        
        # Store timestamps for plotting
        self.timestamps_rms = utils.get_timestamps(rms, hop_length, sample_rate)
        self.timestamps_f0 = utils.get_timestamps(f0, hop_length, sample_rate)
        
        return features
        
    def save_output(self, audio_data, sample_rate, features, output_prefix):
        """
        Save visualizations of the extracted features.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            features (dict): Extracted features
            output_prefix (str): Prefix for output files
        """
        try:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot volume (RMS energy)
            ax1.plot(self.timestamps_rms, self.rms)
            ax1.set_ylabel('RMS Energy')
            ax1.set_title('Audio Volume')
            ax1.grid(True)
            
            # Plot pitch (f0)
            times = self.timestamps_f0
            ax2.scatter(times[self.voiced_flag], self.f0[self.voiced_flag], 
                       alpha=0.5, marker='.', color='blue', label='Voiced')
            ax2.scatter(times[~self.voiced_flag], self.f0[~self.voiced_flag], 
                       alpha=0.2, marker='.', color='gray', label='Unvoiced')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_xlabel('Time (s)')
            ax2.set_title('Audio Pitch')
            ax2.legend()
            ax2.grid(True)
            
            # Add text annotation with overall stats
            text = (f"Mean Volume: {features['oc_audvol']:.4f}\n"
                   f"Volume Change: {features['oc_audvol_diff']:.4f}\n"
                   f"Mean Pitch: {features['oc_audpit']:.1f} Hz\n"
                   f"Pitch Change: {features['oc_audpit_diff']:.1f} Hz")
            plt.figtext(0.02, 0.02, text, fontsize=9, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_basic_audio.png")
            plt.close()
            
            # Save raw data as CSV for further analysis
            df = pd.DataFrame({
                'time': self.timestamps_rms,
                'rms': self.rms
            })
            df.to_csv(f"{output_prefix}_volume.csv", index=False)
            
            df_pitch = pd.DataFrame({
                'time': self.timestamps_f0,
                'f0': self.f0,
                'voiced': self.voiced_flag
            })
            df_pitch.to_csv(f"{output_prefix}_pitch.csv", index=False)
            
        except Exception as e:
            logger.error(f"Error saving audio basic visualizations: {e}")
