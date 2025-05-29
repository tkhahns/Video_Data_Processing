"""
Speech transcription feature extractor using WhisperX.
"""
import os
import logging
import json
import numpy as np
import pandas as pd
import librosa
import tempfile
import matplotlib.pyplot as plt
from ..extractors import BaseExtractor
from .. import utils

logger = logging.getLogger(__name__)

class SpeechTranscriptionExtractor(BaseExtractor):
    """Extractor for speech transcription using WhisperX."""
    
    def __init__(self):
        super().__init__("Speech Transcription")
        self._load_model()
        
    def _load_model(self):
        """Load the WhisperX model."""
        try:
            logger.info("Initializing WhisperX model")
            
            # Check if whisperx is installed
            try:
                import whisperx
                self.model_loaded = True
                # In a real implementation, load the model:
                # self.model = whisperx.load_model("base", device="cpu")
            except ImportError:
                logger.warning("WhisperX not installed. Speech transcription will be mocked.")
                self.model_loaded = False
                
        except Exception as e:
            logger.error(f"Error loading WhisperX model: {e}")
            self.model_loaded = False
    
    def extract(self, audio_data, sample_rate):
        """
        Extract speech transcription features.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            
        Returns:
            dict: Dictionary with transcription features
        """
        features = {}
        
        try:
            if not self.model_loaded:
                # Mock transcription for demonstration
                logger.warning("Using mock transcription as WhisperX is not loaded")
                mock_text = "This is a mock transcription as WhisperX is not loaded."
                features["transcription"] = mock_text
                
                # Mock word-level timestamps
                duration = len(audio_data) / sample_rate
                words = mock_text.split()
                word_timestamps = []
                
                for i, word in enumerate(words):
                    start_time = i * (duration / len(words))
                    end_time = (i + 1) * (duration / len(words))
                    word_timestamps.append({
                        "word": word,
                        "start": start_time,
                        "end": end_time,
                        "speaker": "speaker1" if i % 2 == 0 else "speaker2",
                        "confidence": 0.8
                    })
                
                self.word_timestamps = word_timestamps
                return features
            
            # In a real implementation, use the model:
            # Save audio temporarily for WhisperX
            # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            #     temp_path = temp_file.name
            #     librosa.output.write_wav(temp_path, audio_data, sample_rate)
            #
            # # Run WhisperX
            # result = self.model.transcribe(temp_path, language="en")
            # features["transcription"] = result["text"]
            #
            # # Process word-level timestamps
            # self.word_timestamps = result["segments"]
            #
            # # Clean up
            # os.remove(temp_path)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in speech transcription: {e}")
            features["transcription"] = ""
            self.word_timestamps = []
            return features
    
    def save_output(self, audio_data, sample_rate, features, output_prefix):
        """
        Save speech transcription and visualizations.
        
        Args:
            audio_data (np.ndarray): Audio data
            sample_rate (int): Sample rate
            features (dict): Extracted features
            output_prefix (str): Prefix for output files
        """
        try:
            if "transcription" not in features or not features["transcription"]:
                return
                
            # Save transcription as text
            with open(f"{output_prefix}_transcript.txt", "w") as f:
                f.write(features["transcription"])
            
            # Save word timestamps as JSON
            with open(f"{output_prefix}_words.json", "w") as f:
                json.dump(self.word_timestamps, f, indent=2)
                
            # Generate a simple visualization of word timings
            if self.word_timestamps:
                plt.figure(figsize=(12, 6))
                
                # Plot word timings
                words = [w["word"] for w in self.word_timestamps]
                start_times = [w["start"] for w in self.word_timestamps]
                speakers = [w["speaker"] for w in self.word_timestamps]
                
                # Group by speaker
                speaker_colors = {"speaker1": "blue", "speaker2": "red"}
                
                for i, (word, start, speaker) in enumerate(zip(words, start_times, speakers)):
                    color = speaker_colors.get(speaker, "gray")
                    plt.text(start, i % 5, word, color=color)
                    plt.axvline(x=start, color=color, alpha=0.3, linestyle="--")
                
                # Plot waveform at the bottom
                plt.subplot(212)
                plt.plot(np.linspace(0, len(audio_data)/sample_rate, len(audio_data)), audio_data)
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.title("Audio Waveform")
                
                plt.tight_layout()
                plt.savefig(f"{output_prefix}_transcript_visualization.png")
                plt.close()
                
                # Save as CSV
                df = pd.DataFrame(self.word_timestamps)
                df.to_csv(f"{output_prefix}_words.csv", index=False)
                
        except Exception as e:
            logger.error(f"Error saving speech transcription: {e}")
