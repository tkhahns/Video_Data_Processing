"""
Video processing for emotion recognition.
"""
import os
import cv2
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import tqdm

# Import from the same package
from .detector import EmotionDetector

# Try importing utility functions
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS.ms format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

class EmotionProcessor:
    """
    Process videos for emotion recognition.
    """
    
    def __init__(self, frame_interval=1.0, confidence_threshold=0.5, device="cpu"):
        """
        Initialize the emotion processor.
        
        Args:
            frame_interval: Interval between frames to process (in seconds)
            confidence_threshold: Minimum confidence for emotion detection
            device: Device to run models on ("cpu" or "cuda")
        """
        self.frame_interval = frame_interval
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Initialize the emotion detector
        self.detector = EmotionDetector(device=device)
    
    def process_video(self, video_path: str, output_dir: str) -> str:
        """
        Process a video file for emotion recognition.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the output
            
        Returns:
            Path to the output file
        """
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file path
        output_file = os.path.join(output_dir, f"{video_name}_emotions.txt")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        logger.info(f"Processing video: {video_filename} ({format_timestamp(duration)})")
        
        # Calculate the frame step based on the interval
        frame_step = int(fps * self.frame_interval)
        frame_step = max(1, frame_step)  # Ensure at least 1
        
        # Initialize variables for tracking emotions
        emotions_timeline = []
        current_frame = 0
        
        # Process frames
        with tqdm.tqdm(total=frame_count // frame_step, desc=f"Processing {video_filename}", unit="frame") as pbar:
            while True:
                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get current timestamp
                timestamp = current_frame / fps
                
                # Process the frame for emotions
                emotions = self.detector.analyze_emotions(frame)
                
                # Get dominant emotion
                emotion_name, confidence = self.detector.get_dominant_emotion(emotions)
                
                # If confidence meets threshold or no faces detected
                if confidence >= self.confidence_threshold or emotion_name == "unknown":
                    emotions_timeline.append({
                        'timestamp': timestamp,
                        'frame': current_frame,
                        'emotion': emotion_name,
                        'confidence': confidence
                    })
                
                # Move to the next frame to process
                current_frame += frame_step
                if current_frame >= frame_count:
                    break
                
                pbar.update(1)
        
        # Release resources
        cap.release()
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write(f"Emotion Analysis for {video_filename}\n")
            f.write(f"Processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Frame interval: {self.frame_interval} seconds\n")
            f.write(f"Confidence threshold: {self.confidence_threshold}\n")
            f.write("-" * 50 + "\n\n")
            f.write("TIMESTAMP | EMOTION | CONFIDENCE\n")
            f.write("-" * 50 + "\n")
            
            for entry in emotions_timeline:
                f.write(f"{format_timestamp(entry['timestamp'])} | {entry['emotion']} | {entry['confidence']:.4f}\n")
        
        logger.info(f"Saved emotion analysis to {output_file}")
        return output_file
    
    def process_directory(self, input_dir: str, output_dir: str, file_extensions=('.mp4', '.avi', '.mov', '.mkv')):
        """
        Process all video files in a directory.
        
        Args:
            input_dir: Directory containing video files
            output_dir: Directory to save output files
            file_extensions: Tuple of video file extensions to process
        
        Returns:
            List of processed output files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all video files in the directory
        video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                      if os.path.isfile(os.path.join(input_dir, f)) and 
                      f.lower().endswith(file_extensions)]
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Process each video
        output_files = []
        for video_path in video_files:
            output_file = self.process_video(video_path, output_dir)
            if output_file:
                output_files.append(output_file)
        
        return output_files
    
    def release(self):
        """Release resources."""
        if self.detector:
            self.detector.release()
