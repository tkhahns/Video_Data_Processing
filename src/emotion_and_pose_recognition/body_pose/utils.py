"""
Utility functions for body pose estimation.
"""
import cv2
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def draw_pose_with_confidence(frame: np.ndarray, 
                              pose_data: Dict, 
                              threshold: float = 0.7) -> np.ndarray:
    """
    Draw pose landmarks on a frame with colors based on confidence.
    
    Args:
        frame: Input BGR frame
        pose_data: Pose data dictionary from PoseEstimator
        threshold: Confidence threshold for displaying landmarks
        
    Returns:
        Frame with pose landmarks drawn on it
    """
    landmarks = pose_data.get("landmarks", {})
    result_frame = frame.copy()
    
    for name, point in landmarks.items():
        if point["visibility"] > threshold:
            # High confidence points in green
            color = (0, 255, 0)  # Green in BGR
            size = 5
        elif point["visibility"] > threshold / 2:
            # Medium confidence in yellow
            color = (0, 255, 255)  # Yellow in BGR
            size = 4
        else:
            # Low confidence in red and smaller
            color = (0, 0, 255)  # Red in BGR
            size = 3
            
        cv2.circle(result_frame, (point["x"], point["y"]), size, color, -1)
        
    return result_frame

def overlay_pose_text(frame: np.ndarray, pose_data: Dict) -> np.ndarray:
    """
    Overlay pose information text on a frame.
    
    Args:
        frame: Input BGR frame
        pose_data: Pose data dictionary from PoseEstimator
        
    Returns:
        Frame with pose information overlaid
    """
    result_frame = frame.copy()
    
    # Get basic posture info
    posture = pose_data.get("posture", {})
    position = posture.get("position", "unknown")
    arms = posture.get("arms", "unknown")
    
    # Add overlay text at the top-left
    text_lines = [
        f"Position: {position.capitalize()}",
        f"Arms: {arms.replace('_', ' ').capitalize()}"
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(result_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 255), 2)
        y_offset += 30
    
    return result_frame

def save_pose_data(pose_data: Dict, output_path: str):
    """
    Save pose data to a JSON file.
    
    Args:
        pose_data: Pose data dictionary from PoseEstimator
        output_path: Path to save the JSON file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert float32 values to standard floats for JSON serialization
        serializable_data = convert_to_serializable(pose_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
            
        logger.info(f"Pose data saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving pose data: {e}")

def convert_to_serializable(data):
    """Convert data to JSON serializable format."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return data
