"""
Face detection and emotion recognition functionality.
"""
import os
import logging
import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Union

# Try importing utility functions
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

class EmotionDetector:
    """
    Class for detecting faces and recognizing emotions in images/video frames.
    Uses Py-Feat for emotion recognition.
    """
    
    def __init__(self, device="cpu"):
        """
        Initialize the emotion detector.
        
        Args:
            device: Device to run models on ("cpu" or "cuda")
        """
        self.device = device
        self.face_detector = None
        self.emotion_model = None
        self.use_pyfeat = True
        
        # Initialize components
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize face detection and emotion recognition models."""
        logger.info("Initializing face detection model...")
        
        try:
            # Try to import feat (Py-Feat)
            import feat
            from feat.detector import Detector
            
            # Log the initialization
            logger.info("Loading Py-Feat detector...")
            
            try:
                # Try to create detector with fewer components to reduce potential issues
                self.detector = Detector(
                    face_model="retinaface",
                    landmark_model="mobilenet",
                    au_model="jaanet",
                    emotion_model="resmasknet",
                    facepose_model=None  # Skip facepose to simplify
                )
                logger.info("Successfully loaded Py-Feat detector")
            except Exception as e:
                logger.error(f"Error initializing Py-Feat detector: {e}")
                logger.info("Attempting to initialize with minimal components...")
                
                # Try with minimal components
                try:
                    self.detector = Detector(
                        face_model="retinaface",
                        landmark_model=None,
                        au_model=None,
                        emotion_model="resmasknet",
                        facepose_model=None
                    )
                    logger.info("Successfully loaded Py-Feat detector with minimal components")
                except Exception as e2:
                    logger.error(f"Failed to initialize Py-Feat with minimal components: {e2}")
                    self.use_pyfeat = False
            
        except ImportError:
            logger.error("Failed to import Py-Feat. Please install with: pip install py-feat")
            logger.error("If there are version conflicts, try: pip install py-feat==0.5.1")
            self.use_pyfeat = False
            
        # Initialize OpenCV face detector regardless, as a fallback
        logger.info("Loading OpenCV face detector as fallback...")
        try:
            # Initialize OpenCV face detector using Haar Cascades
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            logger.debug(f"Loading Haar cascade from: {haar_cascade_path}")
            
            self.face_detector = cv2.CascadeClassifier(haar_cascade_path)
            if self.face_detector.empty():
                logger.error("Failed to load OpenCV face detector")
                
                # Try DNN-based detector as an additional fallback
                logger.info("Attempting to initialize OpenCV DNN face detector...")
                try:
                    self._initialize_dnn_face_detector()
                except Exception as e:
                    logger.error(f"Failed to initialize DNN face detector: {e}")
            else:
                logger.info("Loaded OpenCV Haar Cascade face detector")
        except Exception as e:
            logger.error(f"Error setting up OpenCV face detector: {e}")
            self.face_detector = None
    
    def _initialize_dnn_face_detector(self):
        """Initialize DNN-based face detector as a fallback."""
        # Check if model files exist
        model_file = os.path.join(os.path.dirname(__file__), "models", "opencv_face_detector_uint8.pb")
        config_file = os.path.join(os.path.dirname(__file__), "models", "opencv_face_detector.pbtxt")
        
        if not os.path.exists(model_file) or not os.path.exists(config_file):
            # Create models directory if it doesn't exist
            os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
            
            logger.warning("OpenCV DNN face detector model not found. Using Haar cascade instead.")
            return
        
        # Load DNN face detector
        self.dnn_face_detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        logger.info("Loaded OpenCV DNN face detector")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using OpenCV (fallback method).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of face rectangles as (x, y, w, h)
        """
        if self.face_detector is None:
            return []
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using multiple scales for better results
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces detected, try with more relaxed parameters
        if len(faces) == 0:
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                logger.debug(f"Detected {len(faces)} faces with relaxed parameters")
        
        return list(faces)
    
    def analyze_emotions(self, frame: np.ndarray) -> Dict:
        """
        Analyze emotions in a frame using Py-Feat.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with emotion analysis results
        """
        # Verify frame is valid
        if frame is None or frame.size == 0:
            logger.warning("Empty or invalid frame provided to emotion analyzer")
            return {'success': False, 'error': 'Invalid frame'}
        
        # Check if Py-Feat detector is available and we want to use it
        if hasattr(self, 'detector') and self.use_pyfeat:
            try:
                # Convert BGR to RGB for Py-Feat
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                logger.debug(f"Frame shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
                
                # Analyze the frame
                result = self.detector.detect_image(rgb_frame)
                
                # Check if faces were detected
                if not hasattr(result, 'faces') or result.faces is None or len(result.faces) == 0:
                    logger.debug("No faces detected by Py-Feat")
                    
                    # Try OpenCV fallback
                    cv_faces = self.detect_faces(frame)
                    if len(cv_faces) > 0:
                        logger.debug(f"OpenCV detected {len(cv_faces)} faces, but Py-Feat found none")
                
                # Extract emotion predictions
                emotions = {}
                if hasattr(result, 'emotions') and result.emotions is not None and not result.emotions.empty:
                    emotions = result.emotions.to_dict('records')
                    logger.debug(f"Detected emotions: {emotions}")
                else:
                    logger.debug("No emotions detected in frame")
                
                return {
                    'success': True,
                    'faces': result.faces if hasattr(result, 'faces') else [],
                    'emotions': emotions,
                    'raw_result': result
                }
                
            except Exception as e:
                logger.error(f"Error analyzing emotions with Py-Feat: {e}")
                # Fall back to OpenCV face detection
        
        # Fall back to basic face detection
        faces = self.detect_faces(frame)
        logger.debug(f"Fallback face detection found {len(faces)} faces")
        
        return {
            'success': bool(len(faces) > 0),
            'faces': faces,
            'emotions': None
        }
    
    def get_dominant_emotion(self, emotions: Dict) -> Tuple[str, float]:
        """
        Get the dominant emotion and its score from an emotion dictionary.
        
        Args:
            emotions: Dictionary of emotions with scores
            
        Returns:
            Tuple of (dominant_emotion, score)
        """
        if not emotions or 'emotions' not in emotions or not emotions['emotions']:
            if emotions and 'faces' in emotions and len(emotions['faces']) > 0:
                # If faces were detected but no emotions classified
                return ("neutral", 0.5)  # Assume neutral with medium confidence
            return ("unknown", 0.0)
        
        try:
            # Get the first face's emotions
            face_emotions = emotions['emotions'][0]
            
            # Log the emotion values for debugging
            logger.debug(f"Raw emotion values from detector: {face_emotions}")
            
            # Standard emotion keys we want to use
            standard_keys = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
            
            # Define mappings from various model outputs to our standard keys
            emotion_mappings = {
                # Standard keys (lowercase)
                'anger': 'anger',
                'disgust': 'disgust', 
                'fear': 'fear',
                'happiness': 'happiness',
                'happy': 'happiness',
                'joy': 'happiness',
                'sadness': 'sadness',
                'sad': 'sadness',
                'surprise': 'surprise',
                'surprised': 'surprise',
                'neutral': 'neutral',
                
                # Capitalized variants
                'Anger': 'anger',
                'Disgust': 'disgust',
                'Fear': 'fear',
                'Happiness': 'happiness',
                'Happy': 'happiness',
                'Joy': 'happiness',
                'Sadness': 'sadness',
                'Sad': 'sadness',
                'Surprise': 'surprise',
                'Surprised': 'surprise',
                'Neutral': 'neutral',
                
                # All caps variants
                'ANGER': 'anger',
                'DISGUST': 'disgust',
                'FEAR': 'fear',
                'HAPPINESS': 'happiness',
                'HAPPY': 'happiness',
                'JOY': 'happiness',
                'SADNESS': 'sadness',
                'SAD': 'sadness',
                'SURPRISE': 'surprise',
                'SURPRISED': 'surprise',
                'NEUTRAL': 'neutral',
            }
            
            # Create a normalized emotion scores dictionary
            emotion_scores = {}
            
            # First, try direct matching with our standard keys
            for key in standard_keys:
                if key in face_emotions:
                    emotion_scores[key] = face_emotions[key]
            
            # If no matches found, try using the mapping
            if not emotion_scores:
                for model_key, value in face_emotions.items():
                    if model_key in emotion_mappings:
                        standard_key = emotion_mappings[model_key]
                        emotion_scores[standard_key] = value
                        logger.debug(f"Mapped '{model_key}' to standard emotion '{standard_key}'")
            
            # If still no matches, try partial matching with the keys
            if not emotion_scores:
                logger.debug("No exact emotion matches found, trying partial matches")
                for model_key, value in face_emotions.items():
                    if isinstance(value, (int, float)):  # Only consider numeric values
                        for std_key in standard_keys:
                            if std_key.lower() in model_key.lower():
                                emotion_scores[std_key] = value
                                logger.debug(f"Partial match: '{model_key}' -> '{std_key}'")
            
            # Final fallback - use any numeric values that aren't clearly non-emotion fields
            if not emotion_scores:
                logger.debug("Using raw values as no standard matches found")
                # Skip common non-emotion keys
                non_emotion_keys = ['index', 'frame', 'face', 'time', 'timestamp', 'confidence']
                for key, value in face_emotions.items():
                    if (isinstance(value, (int, float)) and 
                        not any(non_key in key.lower() for non_key in non_emotion_keys)):
                        emotion_scores[key] = value
                        logger.debug(f"Using raw emotion: {key} = {value}")
            
            # If we have emotion scores, find the dominant one
            if emotion_scores:
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                logger.info(f"Detected dominant emotion: {dominant_emotion[0]} with score {dominant_emotion[1]:.4f}")
                return dominant_emotion
            else:
                logger.warning("No matching emotion keys found in model output")
                logger.debug(f"Available keys in model output: {list(face_emotions.keys())}")
                return ("neutral", 0.5)  # Default to neutral if no emotions found
            
        except Exception as e:
            logger.error(f"Error getting dominant emotion: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return ("unknown", 0.0)
    
    def release(self):
        """Release resources."""
        # Clean up any resources if needed
        pass
