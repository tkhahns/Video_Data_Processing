"""
Speech emotion recognition feature extractor.
"""
import numpy as np
import librosa
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from ..extractors import BaseExtractor
from .. import utils

logger = logging.getLogger(__name__)

class SpeechEmotionExtractor(BaseExtractor):
    """Extractor for speech emotion recognition features."""
    
    # Emotion categories for the model output
    EMOTION_CATEGORIES = [
        "ser_neutral", "ser_calm", "ser_happy", "ser_sad", 
        "ser_angry", "ser_fear", "ser_disgust", "ser_ps", "ser_boredom"
    ]
    
    def __init__(self):
        super().__init__("Speech Emotion")
        self._load_model()
        
    def _load_model(self):
        """Load the speech emotion recognition model."""
        try:
            # For now, we'll use a simple feature extractor and classifier
            # In a real implementation, load a pre-trained model from disk or HuggingFace
            logger.info("Initializing speech emotion recognition model")
            self.model_loaded = True
            
            # Mock model for now
            # In production, replace with actual model loading code:
            # self.model = torch.load('path/to/model.pt')
            
        except Exception as e:
            logger.error(f"Error loading speech emotion model: {e}")
            self.model_loaded = False
    
    def _extract_features(self, audio_data, sample_rate):
        """
        Extract acoustic features used by emotion classification.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            
        Returns:
            np.ndarray: Features for emotion recognition
        """
        # Resample if needed
        if sample_rate != 16000:
            audio_data = utils.resample_audio(audio_data, sample_rate, 16000)
            sample_rate = 16000
        
        # Normalize audio
        audio_data = utils.normalize_audio(audio_data)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        
        # Extract additional features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        spec_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        
        # Compute statistics over features
        features = []
        for feat in [mfccs, chroma, mel, spec_contrast]:
            features.extend([
                np.mean(feat, axis=1),
                np.std(feat, axis=1)
            ])
        
        # Flatten and concatenate all features
        flat_features = np.concatenate([f.flatten() for f in features])
        
        # Store features for visualization
        self.mfccs = mfccs
        
        return flat_features
        
    def _predict_emotion(self, features):
        """
        Predict emotion probabilities from extracted features.
        
        Args:
            features (np.ndarray): Extracted features
            
        Returns:
            dict: Dictionary of emotion probabilities
        """
        if not self.model_loaded:
            # Return uniform distribution if model not loaded
            uniform = 1.0 / len(self.EMOTION_CATEGORIES)
            return {emotion: uniform for emotion in self.EMOTION_CATEGORIES}
        
        # In a real implementation, you would use the loaded model:
        # with torch.no_grad():
        #     output = self.model(torch.tensor(features).float())
        #     probs = torch.softmax(output, dim=1).numpy()[0]
        
        # Mock prediction for demonstration purposes
        # In production, replace with actual model prediction
        
        # Generate slightly noisy but deterministic predictions based on feature sum
        feat_sum = np.sum(features)
        rng = np.random.RandomState(int(feat_sum * 1000) % 10000)
        
        # Generate base probabilities
        base_probs = rng.dirichlet(np.ones(len(self.EMOTION_CATEGORIES)) * 0.5)
        
        # Create a dictionary of emotion probabilities
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.EMOTION_CATEGORIES, base_probs)}
        
        return emotion_probs
        
    def extract(self, audio_data, sample_rate):
        """
        Extract speech emotion features.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            
        Returns:
            dict: Dictionary with speech emotion probabilities
        """
        try:
            # Extract acoustic features
            features = self._extract_features(audio_data, sample_rate)
            
            # Predict emotions
            emotion_probs = self._predict_emotion(features)
            
            # Store for visualization
            self.emotion_probs = emotion_probs
            
            return emotion_probs
            
        except Exception as e:
            logger.error(f"Error in speech emotion recognition: {e}")
            # Return empty dict on failure
            return {emotion: 0.0 for emotion in self.EMOTION_CATEGORIES}
    
    def save_output(self, audio_data, sample_rate, features, output_prefix):
        """
        Save visualizations of speech emotion recognition.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            features (dict): Extracted features
            output_prefix (str): Prefix for output files
        """
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot MFCCs
            librosa.display.specshow(self.mfccs, x_axis='time', ax=ax1)
            ax1.set_title('MFCC Features')
            ax1.set_ylabel('MFCC Coefficients')
            fig.colorbar(ax1.collections[0], ax=ax1, format='%+2.0f dB')
            
            # Plot emotion probabilities
            emotions = list(self.emotion_probs.keys())
            probs = list(self.emotion_probs.values())
            
            ax2.bar(emotions, probs, color='skyblue')
            ax2.set_title('Emotion Probabilities')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_emotion.png")
            plt.close()
            
            # Save emotion probabilities as CSV
            df = pd.DataFrame([self.emotion_probs])
            df.to_csv(f"{output_prefix}_emotion.csv", index=False)
            
        except Exception as e:
            logger.error(f"Error saving speech emotion visualizations: {e}")
