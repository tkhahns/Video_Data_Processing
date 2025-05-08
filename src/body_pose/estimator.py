"""
Core functionality for body pose estimation in videos.
"""
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

class PoseEstimator:
    """MediaPipe-based pose estimator for videos."""
    
    def __init__(self, 
                 static_image_mode=False, 
                 model_complexity=1, 
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the pose estimator with MediaPipe.
        
        Args:
            static_image_mode: Whether to treat input as static images (vs video)
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
            smooth_landmarks: Whether to apply smoothing to landmarks
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info(f"Initialized PoseEstimator with model complexity {model_complexity}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame to detect pose.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (annotated frame, pose data dictionary)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        pose_data = {}
        
        if results.pose_landmarks:
            # Draw pose landmarks on the frame
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Extract pose data
            pose_data = self._extract_pose_data(results.pose_landmarks, frame.shape)
            
            # Add basic posture analysis
            pose_data["posture"] = self._analyze_posture(pose_data)
            
        return annotated_frame, pose_data
    
    def _extract_pose_data(self, landmarks, frame_shape) -> Dict:
        """
        Extract relevant pose data from landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Dictionary with pose landmark positions and confidence scores
        """
        height, width, _ = frame_shape
        data = {}
        
        # Convert landmarks to pixel coordinates and confidence
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            # Get the landmark name from MediaPipe's POSE_LANDMARKS enum
            landmark_name = self.mp_pose.PoseLandmark(idx).name
            
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # Depth (relative to hip)
            visibility = landmark.visibility
            
            points[landmark_name] = {
                "x": x,
                "y": y,
                "z": z,
                "visibility": float(visibility)
            }
        
        data["landmarks"] = points
        
        # Calculate key angles and distances
        data["angles"] = self._calculate_key_angles(points)
        data["distances"] = self._calculate_key_distances(points, width, height)
        
        return data
    
    def _calculate_key_angles(self, points) -> Dict:
        """Calculate key body angles from pose landmarks."""
        angles = {}
        
        # Only calculate angles if necessary landmarks are detected
        required_landmarks = {
            "elbow_angle_right": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
            "elbow_angle_left": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
            "knee_angle_right": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
            "knee_angle_left": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
            "hip_angle_right": ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
            "hip_angle_left": ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"]
        }
        
        for angle_name, landmarks in required_landmarks.items():
            if all(lm in points for lm in landmarks):
                p1 = points[landmarks[0]]
                p2 = points[landmarks[1]]
                p3 = points[landmarks[2]]
                
                # Only calculate if all points are reasonably visible
                if (p1["visibility"] > 0.7 and p2["visibility"] > 0.7 and p3["visibility"] > 0.7):
                    angle = self._calculate_angle(
                        (p1["x"], p1["y"]),
                        (p2["x"], p2["y"]),
                        (p3["x"], p3["y"])
                    )
                    angles[angle_name] = angle
        
        return angles
    
    def _calculate_key_distances(self, points, frame_width, frame_height) -> Dict:
        """Calculate key distances between body parts, normalized by frame size."""
        distances = {}
        
        # Define pairs of landmarks to measure distances between
        pairs = [
            ("wrist_distance", "LEFT_WRIST", "RIGHT_WRIST"),
            ("ankle_distance", "LEFT_ANKLE", "RIGHT_ANKLE"),
            ("shoulder_distance", "LEFT_SHOULDER", "RIGHT_SHOULDER"),
            ("hip_distance", "LEFT_HIP", "RIGHT_HIP")
        ]
        
        diagonal = np.sqrt(frame_width**2 + frame_height**2)
        
        for name, lm1, lm2 in pairs:
            if lm1 in points and lm2 in points:
                p1 = points[lm1]
                p2 = points[lm2]
                
                # Check visibility
                if p1["visibility"] > 0.7 and p2["visibility"] > 0.7:
                    dist = np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
                    # Normalize by diagonal of frame
                    distances[name] = dist / diagonal
        
        return distances
    
    def _calculate_angle(self, p1, p2, p3) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            p1, p2, p3: Three points as (x, y) tuples. p2 is the vertex.
            
        Returns:
            Angle in degrees
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Prevent numerical errors
        
        # Calculate angle in degrees
        angle = np.arccos(cosine_angle) * 180 / np.pi
        
        return angle
    
    def _analyze_posture(self, pose_data) -> Dict:
        """
        Analyze the posture based on pose landmarks and angles.
        
        Args:
            pose_data: Dictionary containing pose landmarks and angles
            
        Returns:
            Dictionary with posture analysis results
        """
        posture = {}
        landmarks = pose_data.get("landmarks", {})
        angles = pose_data.get("angles", {})
        
        # Check if person is standing or sitting
        if "LEFT_KNEE" in landmarks and "RIGHT_KNEE" in landmarks and "LEFT_ANKLE" in landmarks and "RIGHT_ANKLE" in landmarks:
            left_knee = landmarks["LEFT_KNEE"]
            right_knee = landmarks["RIGHT_KNEE"]
            left_ankle = landmarks["LEFT_ANKLE"]
            right_ankle = landmarks["RIGHT_ANKLE"]
            
            # If knees are significantly higher than ankles, person is likely sitting
            knee_y_avg = (left_knee["y"] + right_knee["y"]) / 2
            ankle_y_avg = (left_ankle["y"] + right_ankle["y"]) / 2
            
            if knee_y_avg < ankle_y_avg - 50:  # Threshold for sitting detection
                posture["position"] = "sitting"
            else:
                posture["position"] = "standing"
        else:
            posture["position"] = "unknown"
        
        # Check if arms are raised
        if "LEFT_SHOULDER" in landmarks and "LEFT_WRIST" in landmarks and "RIGHT_SHOULDER" in landmarks and "RIGHT_WRIST" in landmarks:
            left_shoulder = landmarks["LEFT_SHOULDER"]
            left_wrist = landmarks["LEFT_WRIST"]
            right_shoulder = landmarks["RIGHT_SHOULDER"]
            right_wrist = landmarks["RIGHT_WRIST"]
            
            left_arms_raised = left_wrist["y"] < left_shoulder["y"]
            right_arms_raised = right_wrist["y"] < right_shoulder["y"]
            
            if left_arms_raised and right_arms_raised:
                posture["arms"] = "both_raised"
            elif left_arms_raised:
                posture["arms"] = "left_raised"
            elif right_arms_raised:
                posture["arms"] = "right_raised"
            else:
                posture["arms"] = "lowered"
        else:
            posture["arms"] = "unknown"
        
        return posture
    
    def close(self):
        """Release resources used by the pose estimator."""
        self.pose.close()
        logger.info("Pose estimator resources released")
