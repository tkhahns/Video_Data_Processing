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
                 min_tracking_confidence=0.5,
                 enable_segmentation=False):  # CHANGED: Default to False to avoid segmentation errors
        """
        Initialize the pose estimator with MediaPipe.
        
        Args:
            static_image_mode: Whether to treat input as static images (vs video)
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
            smooth_landmarks: Whether to apply smoothing to landmarks
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_segmentation: Enable segmentation mask for better person isolation
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize main pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,  # Use the passed parameter
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Store if segmentation is enabled for error handling
        self.enable_segmentation = enable_segmentation
        
        # For multi-person detection, we'll also use a holistic model
        # which sometimes gives better results for multiple people
        try:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                enable_segmentation=enable_segmentation
            )
            self.has_holistic = True
        except Exception as e:
            logger.warning(f"Could not initialize holistic model: {e}")
            self.has_holistic = False
            
        # Initialize multi-person Detector (if available in MediaPipe)
        # This uses a trick with the detector-only mode
        try:
            # Use lower confidence for detection to catch more people
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,  # Always use static mode for detector
                model_complexity=model_complexity,
                min_detection_confidence=0.3,  # Lower threshold to detect more people
                enable_segmentation=True
            )
            self.has_detector = True
        except Exception as e:
            logger.warning(f"Could not initialize multi-person detector: {e}")
            self.has_detector = False
        
        self.previous_poses = {}  # Store previous pose data for tracking
        self.person_count = 0  # Counter for detected people
        
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
        height, width, _ = frame.shape
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Initialize pose data dictionary
        pose_data = {
            "landmarks": [],
            "multi_person_landmarks": []
        }
        
        # First process with main pose detector
        try:
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract main pose data
                landmarks = self._extract_pose_data(results.pose_landmarks, frame.shape)
                pose_data["landmarks"] = landmarks
                
                # Analyze posture
                pose_data["posture"] = self._analyze_posture(landmarks)
        except Exception as e:
            logger.warning(f"Error in main pose detection: {e}")
        
        # Try multiple detection methods to capture all people in the frame
        multi_person_data = []
        
        # Method 1: Try to find people in different regions - BUT DISABLE SEGMENTATION
        # Divide the image into regions and process each
        if self.has_detector:
            # Create a separate detector with segmentation disabled to avoid dimension errors
            try:
                temp_detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.3,
                    enable_segmentation=False  # Disable segmentation to avoid errors
                )
                
                # Divide frame into left and right halves
                left_half = rgb_frame[:, :width//2, :]
                right_half = rgb_frame[:, width//2:, :]
                
                # Process each half separately
                for i, half_frame in enumerate([left_half, right_half]):
                    try:
                        half_results = temp_detector.process(half_frame)
                        if half_results and half_results.pose_landmarks:
                            # Extract pose data - adjust x coordinates
                            half_landmarks = half_results.pose_landmarks
                            if i == 1:  # Right half - adjust x coordinates
                                for lm in half_landmarks.landmark:
                                    lm.x = (lm.x * 0.5) + 0.5  # Convert to global coordinates
                            
                            person_data = self._extract_pose_data(half_landmarks, frame.shape)
                            multi_person_data.append(person_data)
                    except Exception as e:
                        logger.debug(f"Error processing half frame {i}: {e}")
                
                # Clean up temporary detector
                temp_detector.close()
                
            except Exception as e:
                logger.warning(f"Error creating temporary detector: {e}")
        
        # Method 2: Try holistic model which sometimes catches different people
        if self.has_holistic:
            try:
                holistic_results = self.holistic.process(rgb_frame)
                if holistic_results and holistic_results.pose_landmarks:
                    holistic_data = self._extract_pose_data(holistic_results.pose_landmarks, frame.shape)
                    
                    # Only add if this looks like a different person
                    if not self._is_duplicate_pose(holistic_data, pose_data["landmarks"]) and \
                       not any(self._is_duplicate_pose(holistic_data, p) for p in multi_person_data):
                        multi_person_data.append(holistic_data)
            except Exception as e:
                logger.debug(f"Error processing with holistic model: {e}")
        
        # Method 3: Process different subregions to catch more poses - USING NON-SEGMENTATION APPROACH
        try:
            # Create a separate detector with segmentation OFF for region processing
            region_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.3,
                enable_segmentation=False  # Disable segmentation for region processing
            )
            
            regions = [
                # Skip full frame which we already processed with the main estimator
                (0, 0, width//2, height),  # Left half
                (width//2, 0, width//2, height),  # Right half
            ]
            
            for i, (x, y, w, h) in enumerate(regions):
                try:
                    # Extract region
                    region = rgb_frame[y:y+h, x:x+w, :]
                    if region.size == 0:
                        continue
                        
                    # Process region with non-segmentation detector
                    region_results = region_detector.process(region)
                    if region_results and region_results.pose_landmarks:
                        # Convert local coordinates to global
                        for lm in region_results.pose_landmarks.landmark:
                            lm.x = lm.x * (w / width) + (x / width)
                            lm.y = lm.y * (h / height) + (y / height)
                        
                        region_data = self._extract_pose_data(region_results.pose_landmarks, frame.shape)
                        
                        # Check if this is a new person
                        if not self._is_duplicate_pose(region_data, pose_data["landmarks"]) and \
                           not any(self._is_duplicate_pose(region_data, p) for p in multi_person_data):
                            multi_person_data.append(region_data)
                except Exception as e:
                    logger.debug(f"Error processing region {i}: {e}")
            
            # Clean up temporary detector
            region_detector.close()
            
        except Exception as e:
            logger.debug(f"Error in region processing: {e}")
        
        # Add multi-person landmarks to pose_data
        pose_data["multi_person_landmarks"] = multi_person_data
        
        # Update the number of people detected
        pose_data["person_count"] = 1 + len(multi_person_data)
        
        # Store for future tracking
        self.previous_poses = pose_data
        
        return annotated_frame, pose_data
    
    def _is_duplicate_pose(self, pose1, pose2, threshold=0.3):
        """Check if two poses are likely the same person."""
        # If either pose is empty, they're not duplicates
        if not pose1 or not pose2:
            return False
            
        # Extract key points if they exist
        key_landmarks = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        
        # Count matching points
        matching_points = 0
        total_points = 0
        
        for lm in key_landmarks:
            if lm in pose1 and lm in pose2:
                total_points += 1
                # Calculate distance
                p1 = pose1[lm]
                p2 = pose2[lm]
                
                # Simple distance check
                if isinstance(p1, dict) and isinstance(p2, dict) and "x" in p1 and "x" in p2:
                    dist = np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
                    # If points are close, consider them the same person
                    if dist < threshold * 100:  # Using pixel threshold
                        matching_points += 1
        
        # If we have enough matching points, consider them the same
        return matching_points > 0 and matching_points / max(total_points, 1) > 0.5
    
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
        for idx, landmark in enumerate(landmarks.landmark):
            # Get the landmark name from MediaPipe's POSE_LANDMARKS enum
            landmark_name = self.mp_pose.PoseLandmark(idx).name
            
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # Depth (relative to hip)
            visibility = landmark.visibility
            
            data[landmark_name] = {
                "x": x,
                "y": y,
                "z": z,
                "visibility": float(visibility)
            }
        
        # Calculate key angles and distances
        data["angles"] = self._calculate_key_angles(data)
        data["distances"] = self._calculate_key_distances(data, width, height)
        
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
        landmarks = pose_data
        angles = pose_data.get("angles", {})
        
        # Initialize with unknown posture
        posture["position"] = "unknown"
        posture["confidence"] = 0.0
        
        # Collect evidence for sitting/standing classification
        evidence = {
            "sitting": 0.0,
            "standing": 0.0
        }
        
        # First check: Do we have enough landmarks to make a determination?
        has_upper_body = all(lm in landmarks for lm in ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER"])
        has_hips = all(lm in landmarks for lm in ["LEFT_HIP", "RIGHT_HIP"])
        has_knees = all(lm in landmarks for lm in ["LEFT_KNEE", "RIGHT_KNEE"])
        has_ankles = all(lm in landmarks for lm in ["LEFT_ANKLE", "RIGHT_ANKLE"])
        
        # If we don't have the basic upper body landmarks, we can't reliably determine posture
        if not has_upper_body:
            return {"position": "unknown", "confidence": 0.0, "arms": "unknown"}
        
        # Calculate relevant landmark positions
        shoulder_y = 0
        if "LEFT_SHOULDER" in landmarks and "RIGHT_SHOULDER" in landmarks:
            left_shoulder = landmarks["LEFT_SHOULDER"]
            right_shoulder = landmarks["RIGHT_SHOULDER"]
            shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
            
        hip_y = 0
        if has_hips:
            left_hip = landmarks["LEFT_HIP"]
            right_hip = landmarks["RIGHT_HIP"]
            hip_y = (left_hip["y"] + right_hip["y"]) / 2
        
        knee_y = 0
        if has_knees:
            left_knee = landmarks["LEFT_KNEE"]
            right_knee = landmarks["RIGHT_KNEE"]
            knee_y = (left_knee["y"] + right_knee["y"]) / 2
            
        ankle_y = 0
        if has_ankles:
            left_ankle = landmarks["LEFT_ANKLE"]
            right_ankle = landmarks["RIGHT_ANKLE"]
            ankle_y = (left_ankle["y"] + right_ankle["y"]) / 2
            
        # Evidence 1: Knee angle (bent knees suggest sitting)
        knee_angle_left = angles.get("knee_angle_left", 180)
        knee_angle_right = angles.get("knee_angle_right", 180)
        avg_knee_angle = (knee_angle_left + knee_angle_right) / 2 if has_knees else 180
        
        # Bent knees (60-130 degrees) strongly suggest sitting
        if has_knees:
            if 60 <= avg_knee_angle <= 130:
                evidence["sitting"] += 0.5
            else:
                evidence["standing"] += 0.3
            
        # Evidence 2: Relative positions of body parts
        if has_hips and has_knees and has_ankles:
            knee_hip_diff = abs(knee_y - hip_y)
            knee_ankle_diff = abs(knee_y - ankle_y)
            
            # If knees are closer to hips than ankles, likely sitting
            if knee_hip_diff < knee_ankle_diff * 0.5:
                evidence["sitting"] += 0.4
            else:
                evidence["standing"] += 0.3
                
            # If knees are higher than ankles, likely sitting
            if knee_y < ankle_y - 10:
                evidence["sitting"] += 0.3
            else:
                evidence["standing"] += 0.2
                
        # Evidence 3: Hip visibility and position
        # In many videos, lower body is not visible when person is standing
        if has_upper_body and not has_knees and not has_ankles and has_hips:
            # If we see shoulders and hips but not legs, probably standing
            evidence["standing"] += 0.4
        
        # Evidence 4: Vertical distribution of visible landmarks
        visible_landmarks_y = [landmarks[lm]["y"] for lm in landmarks if 
                              isinstance(landmarks[lm], dict) and "y" in landmarks[lm]]
        
        if visible_landmarks_y:
            min_y = min(visible_landmarks_y)
            max_y = max(visible_landmarks_y)
            range_y = max_y - min_y
            
            # If landmarks are spread out vertically, likely standing
            if range_y > 300:  # Significant vertical spread
                evidence["standing"] += 0.3
            else:
                evidence["sitting"] += 0.2
                
        # For artificial poses or cases with limited landmarks, use shoulder-hip ratio
        if has_upper_body and has_hips:
            shoulder_hip_distance = abs(hip_y - shoulder_y)
            
            # Compare to heuristics based on typical body proportions:
            # - Standing: longer vertical distance from shoulders to hips
            # - Sitting: often compressed, shorter distance from shoulders to hips
            if shoulder_hip_distance > 100:
                evidence["standing"] += 0.3
            else:
                evidence["sitting"] += 0.2
        
        # Determine final posture based on evidence
        total_evidence = evidence["sitting"] + evidence["standing"]
        if total_evidence > 0:
            if evidence["sitting"] > evidence["standing"]:
                posture["position"] = "sitting"
                posture["confidence"] = evidence["sitting"] / total_evidence
            else:
                posture["position"] = "standing" 
                posture["confidence"] = evidence["standing"] / total_evidence
        
        # Add detailed evidence for debugging
        posture["evidence"] = evidence
        posture["angles"] = {
            "knee_angle_left": knee_angle_left,
            "knee_angle_right": knee_angle_right,
            "avg_knee_angle": avg_knee_angle
        }
        posture["positions"] = {
            "shoulder_y": shoulder_y,
            "hip_y": hip_y,
            "knee_y": knee_y,
            "ankle_y": ankle_y
        }
        
        # Determine arm position (similar to original code)
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
        try:
            # Use a safer closure of the pose estimator
            if hasattr(self, 'pose'):
                try:
                    self.pose.close()
                except RuntimeError as e:
                    # We know the segmentation smoother can fail, so log and continue
                    if "SegmentationSmoothingCalculator" in str(e):
                        logger.warning("Ignoring known segmentation smoother error during pose close")
                    else:
                        # If it's some other error, we should still log it
                        logger.warning(f"Error closing main pose estimator: {e}")
            
            # Handle holistic closer
            if hasattr(self, 'holistic') and self.has_holistic:
                try:
                    self.holistic.close()
                except Exception as e:
                    logger.warning(f"Error closing holistic estimator: {e}")
            
            # Handle pose detector closer
            if hasattr(self, 'pose_detector') and self.has_detector:
                try:
                    self.pose_detector.close()
                except Exception as e:
                    logger.warning(f"Error closing pose detector: {e}")
                    
        except Exception as e:
            # Catch any other exceptions during close
            logger.warning(f"Unexpected error during pose estimator cleanup: {e}")
            
        logger.info("Pose estimator resources released")
