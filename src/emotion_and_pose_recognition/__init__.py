"""
Facial emotion recognition and body pose analysis module for video processing.

This module provides functionality to detect faces in videos and recognize emotions 
using deep learning models from the DeepFace library, as well as analyze body poses.
It also supports advanced video feature extraction for pose estimation and facial expression analysis.
"""

__version__ = "0.1.0"

# Make video feature extraction available at the module level
from .video_features import extract_video_features, VideoFeatureExtractor
