"""
Feature extractors for audio and video.
Each extractor processes input data and returns a dictionary of features.
"""

class BaseExtractor:
    """Base class for all feature extractors."""
    
    def __init__(self, name):
        self.name = name
        
    def extract(self, audio_data, sample_rate):
        """
        Extract features from audio data.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            
        Returns:
            dict: Dictionary of extracted features
        """
        raise NotImplementedError("Subclasses must implement extract()")
    
    def save_output(self, audio_data, sample_rate, features, output_prefix):
        """
        Save extractor-specific output files (optional).
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            features (dict): Extracted features
            output_prefix (str): Prefix for output files
        """
        pass
