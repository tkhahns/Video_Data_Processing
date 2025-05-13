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
    
    if isinstance(landmarks, list):
        # Handle list format (indices)
        for i, landmark in enumerate(landmarks):
            if landmark is None:
                continue
            
            # Default to high confidence for list format
            color = (0, 255, 0)  # Green in BGR
            size = 5
                
            try:
                cv2.circle(result_frame, (int(landmark[0]), int(landmark[1])), size, color, -1)
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Error drawing landmark {i}: {e}")
    else:
        # Handle dictionary format (named landmarks)
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

def draw_pose_skeleton(frame: np.ndarray,
                       pose_data: Union[Dict, List],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2,
                       connections: List[Tuple[int, int]] = None) -> np.ndarray:
    """
    Draw a complete pose skeleton on the frame based on pose landmarks.
    Works with both dictionary and list landmark formats.
    
    Args:
        frame: Input BGR frame
        pose_data: Either a dictionary with landmarks or a list of landmark coordinates
        color: BGR color tuple for the skeleton lines
        thickness: Line thickness
        connections: List of landmark index pairs to connect
        
    Returns:
        Frame with skeleton drawn on it
    """
    result_frame = frame.copy()
    
    # Define default connections if none provided
    if connections is None:
        connections = [
            # Face connections
            (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
            # Shoulders to ears
            (9, 10), (11, 12), (11, 13), (12, 14),
            # Arms
            (13, 15), (15, 17), (15, 19), (15, 21),
            (14, 16), (16, 18), (16, 20), (16, 22),
            # Body core
            (11, 23), (12, 24), (23, 24),
            # Legs
            (23, 25), (25, 27), (27, 29), (29, 31),
            (24, 26), (26, 28), (28, 30), (30, 32)
        ]
    
    # Check if we have a list of all None values
    if isinstance(pose_data, list) and all(x is None for x in pose_data):
        logger.warning("Received list of all None landmarks - cannot draw skeleton")
        return result_frame
        
    # Convert the pose_data to a standardized format
    points = {}
    
    # Handle list format directly (most common case from processor.py)
    if isinstance(pose_data, list):
        for i, landmark in enumerate(pose_data):
            if landmark is not None:
                try:
                    # Cast to float first to handle various numeric types
                    x, y = float(landmark[0]), float(landmark[1])
                    points[i] = (int(x), int(y))
                except (ValueError, TypeError, IndexError) as e:
                    # Skip invalid landmarks without crashing
                    logger.debug(f"Skipped invalid landmark {i}: {e}")
    
    # Log if we found points or not
    if not points:
        logger.warning("NO VALID POINTS FOR SKELETON DRAWING - Check landmark format!")
        return result_frame
    
    # Draw the connections between points
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx in points and end_idx in points:
            try:
                cv2.line(result_frame, points[start_idx], points[end_idx], color, thickness)
            except Exception as e:
                logger.debug(f"Error drawing line: {e}")
    
    # Draw the landmark points
    for idx, point in points.items():
        # Make important points bigger
        radius = 5 if idx in [11, 12, 23, 24] else 3
        cv2.circle(result_frame, point, radius, color, -1)
    
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

def create_fallback_pose(face_x: int, face_y: int, face_w: int, face_h: int) -> List:
    """
    Create fallback pose landmarks when detection fails, based on face position.
    
    Args:
        face_x: x coordinate of face
        face_y: y coordinate of face
        face_w: width of face
        face_h: height of face
        
    Returns:
        List of estimated landmark coordinates
    """
    # Create a list of None values as a base
    landmarks = [None] * 33
    
    # Head landmarks
    landmarks[0] = (face_x + face_w//2, face_y + face_h//2)  # Nose at face center
    
    # Use face width to estimate body proportions - assume face is about 1/6 of body height
    body_scale = face_h * 6
    shoulder_width = face_w * 1.5
    
    # Eyes
    eye_y = face_y + face_h//3
    landmarks[2] = (face_x + face_w//4, eye_y)      # Left eye
    landmarks[5] = (face_x + 3*face_w//4, eye_y)    # Right eye
    
    # Ears
    landmarks[7] = (face_x, face_y + face_h//2)     # Left ear
    landmarks[8] = (face_x + face_w, face_y + face_h//2)  # Right ear
    
    # Shoulders
    shoulder_y = face_y + face_h + face_h//2
    landmarks[11] = (face_x + face_w//2 - shoulder_width//2, shoulder_y)  # Left shoulder
    landmarks[12] = (face_x + face_w//2 + shoulder_width//2, shoulder_y)  # Right shoulder
    
    # Elbows - halfway between shoulder and wrist
    elbow_y = shoulder_y + body_scale//5
    landmarks[13] = (face_x + face_w//2 - shoulder_width*0.8, elbow_y)  # Left elbow
    landmarks[14] = (face_x + face_w//2 + shoulder_width*0.8, elbow_y)  # Right elbow
    
    # Wrists
    wrist_y = elbow_y + body_scale//5
    landmarks[15] = (face_x + face_w//2 - shoulder_width*0.9, wrist_y)  # Left wrist
    landmarks[16] = (face_x + face_w//2 + shoulder_width*0.9, wrist_y)  # Right wrist
    
    # Hip points
    hip_y = shoulder_y + body_scale//3
    hip_width = shoulder_width * 0.8
    landmarks[23] = (face_x + face_w//2 - hip_width//2, hip_y)  # Left hip
    landmarks[24] = (face_x + face_w//2 + hip_width//2, hip_y)  # Right hip
    
    # Knees
    knee_y = hip_y + body_scale//3
    landmarks[25] = (face_x + face_w//2 - hip_width//2, knee_y)  # Left knee
    landmarks[26] = (face_x + face_w//2 + hip_width//2, knee_y)  # Right knee
    
    # Ankles
    ankle_y = knee_y + body_scale//3
    landmarks[27] = (face_x + face_w//2 - hip_width//2, ankle_y)  # Left ankle
    landmarks[28] = (face_x + face_w//2 + hip_width//2, ankle_y)  # Right ankle
    
    return landmarks

def create_landmark_dict_from_list(landmarks: List) -> Dict:
    """
    Convert a list of landmarks to a dictionary format for posture analysis.
    
    Args:
        landmarks: List of 33 landmarks (x,y) coordinates or None values
        
    Returns:
        Dictionary mapping landmark names to position data
    """
    landmark_names = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR",
        "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]
    
    result = {}
    for i, landmark in enumerate(landmarks):
        if i >= len(landmark_names):
            break
            
        if landmark is not None:
            result[landmark_names[i]] = {
                "x": landmark[0],
                "y": landmark[1],
                "z": 0.0,  # We don't have Z data from 2D landmarks
                "visibility": 1.0  # Assume full visibility for artificial landmarks
            }
    
    return result
