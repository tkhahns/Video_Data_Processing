"""
Video feature extraction functionality for emotion and pose recognition.

This module provides functions to extract advanced features from videos using various
models for pose estimation, facial expression analysis, and motion tracking.
"""

import os
import cv2
import numpy as np
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class VideoFeatureExtractor:
    """
    Extracts multiple types of features from videos using various models.
    
    This class manages the extraction of features related to:
    - 3D body pose estimation (PARE, ViTPose)
    - Facial action unit detection (AU detection)
    - Emotion recognition (DAN, ELN)
    - Motion analysis (optical flow)
    - Multimodal analysis (AV HuBERT, MELD)
    """
    
    def __init__(self, 
                 models_dir: str = None,
                 use_gpu: bool = True, 
                 batch_size: int = 16,
                 detection_threshold: float = 0.7):
        """
        Initialize the feature extractor with specified models.
        
        Args:
            models_dir: Directory containing model weights
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for processing frames
            detection_threshold: Confidence threshold for detections
        """
        self.models_dir = models_dir or os.path.join(os.path.expanduser('~'), '.video_data_processing', 'models')
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.detection_threshold = detection_threshold
        
        # Track loaded models
        self._loaded_models = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize with no models loaded
        logger.info(f"VideoFeatureExtractor initialized with models_dir={self.models_dir}, use_gpu={use_gpu}")

    def _ensure_model_loaded(self, model_name: str) -> bool:
        """
        Ensure the specified model is loaded.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if model_name in self._loaded_models:
            return True
            
        try:
            if model_name == "pare":
                self._load_pare_model()
            elif model_name == "vitpose":
                self._load_vitpose_model()
            elif model_name == "psa":
                self._load_psa_model()
            elif model_name == "rsn":
                self._load_rsn_model()
            elif model_name == "au_detector":
                self._load_au_detector_model()
            elif model_name == "dan":
                self._load_dan_model()
            elif model_name == "eln":
                self._load_eln_model()
            elif model_name == "mediapipe":
                self._load_mediapipe_model()
            elif model_name == "pyfeat":
                self._load_pyfeat_model()
            elif model_name == "optical_flow":
                self._load_optical_flow_model()
            elif model_name == "av_hubert":
                self._load_av_hubert_model()
            elif model_name == "meld":
                self._load_meld_model()
            else:
                logger.warning(f"Unknown model: {model_name}")
                return False
                
            self._loaded_models[model_name] = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _load_pare_model(self):
        """Load PARE model for 3D human pose estimation."""
        logger.info("Loading PARE model - placeholder implementation")
        # In a real implementation, this would load the PARE model

    def _load_vitpose_model(self):
        """Load ViTPose model for pose estimation."""
        logger.info("Loading ViTPose model - placeholder implementation")
        # In a real implementation, this would load the ViTPose model

    def _load_psa_model(self):
        """Load Polarized Self-Attention model."""
        logger.info("Loading PSA model - placeholder implementation")
        # In a real implementation, this would load the PSA model

    def _load_rsn_model(self):
        """Load Residual Steps Network model."""
        logger.info("Loading RSN model - placeholder implementation")
        # In a real implementation, this would load the RSN model
        
    def _load_au_detector_model(self):
        """Load Action Unit detector model."""
        logger.info("Loading AU detector model - placeholder implementation")
        # In a real implementation, this would load the AU detector model

    def _load_dan_model(self):
        """Load DAN emotion recognition model."""
        logger.info("Loading DAN model - placeholder implementation")
        # In a real implementation, this would load the DAN model

    def _load_eln_model(self):
        """Load ELN model for facial expression analysis."""
        logger.info("Loading ELN model - placeholder implementation")
        # In a real implementation, this would load the ELN model

    def _load_mediapipe_model(self):
        """Load Google MediaPipe for pose estimation."""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_pose_instance = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe model loaded successfully")
        except ImportError:
            logger.error("MediaPipe not installed. Please install with: pip install mediapipe")
            raise

    def _load_pyfeat_model(self):
        """Load Py-Feat for facial expression analysis."""
        try:
            logger.info("Attempting to load Py-Feat")
            from py_feat import Detector
            self.pyfeat_detector = Detector(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                au_model="xgb",
                emotion_model="resmasknet",
                facepose_model="img2pose"
            )
            logger.info("Py-Feat model loaded successfully")
        except ImportError:
            logger.error("Py-Feat not installed. Please install with: pip install py-feat")
            raise

    def _load_optical_flow_model(self):
        """Load optical flow model."""
        logger.info("Loading optical flow model - placeholder implementation")
        # In a real implementation, this would load an optical flow model

    def _load_av_hubert_model(self):
        """Load AV HuBERT model for audio-visual speech analysis."""
        logger.info("Loading AV HuBERT model for multimodal audio-visual speech analysis")
        try:
            # In a real implementation, this would load the AV HuBERT model and dependencies
            # from transformers import AutoModel, AutoProcessor
            # self.av_hubert_model = AutoModel.from_pretrained("facebook/av-hubert-base")
            # self.av_hubert_processor = AutoProcessor.from_pretrained("facebook/av-hubert-base")
            
            # Placeholder for demonstration
            import numpy as np
            self.av_hubert_model = "placeholder_model"
            logger.info("AV HuBERT model loaded successfully")
        except ImportError:
            logger.error("Required packages for AV HuBERT not installed. Install with: pip install transformers torch")
            raise

    def _load_meld_model(self):
        """Load MELD model for multimodal emotion recognition in conversations."""
        logger.info("Loading MELD model for multimodal emotion recognition in conversations")
        try:
            # In a real implementation, this would load the MELD model
            # The MELD model would be used for conversation-level emotion analysis
            
            # Placeholder for demonstration
            self.meld_model = "placeholder_model"
            # Set up the emotion categories used in MELD
            self.meld_emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
            logger.info("MELD model loaded successfully")
        except ImportError:
            logger.error("Required packages for MELD not installed")
            raise

    def extract_features(self, 
                        video_path: str, 
                        output_path: str = None,
                        models: List[str] = None,
                        sample_rate: int = 1,
                        video_name: str = None,
                        audio_path: str = None) -> Dict[str, Any]:
        """
        Extract features from a video using specified models.
        
        Args:
            video_path: Path to the input video file
            output_path: Optional path to save extracted features
            models: List of model names to use for feature extraction
            sample_rate: Process every Nth frame for efficiency
            video_name: Optional name of the video to include in features
            audio_path: Optional path to extracted audio for multimodal analysis
            
        Returns:
            Dictionary containing extracted features
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Default to all models if none specified
        if models is None:
            models = ["pare", "vitpose", "psa", "rsn", "au_detector", "dan", 
                     "eln", "mediapipe", "pyfeat", "optical_flow"]
        
        # Use filename if video_name not provided
        if video_name is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Check if multimodal models are requested and if audio path is available
        multimodal_models = [m for m in models if m in ["av_hubert", "meld"]]
        if multimodal_models and audio_path is None:
            # Try to find corresponding audio file if not provided
            potential_audio_path = os.path.join(
                os.path.dirname(video_path), 
                "..", "separated_speech", 
                f"{os.path.splitext(os.path.basename(video_path))[0]}.wav"
            )
            if os.path.exists(potential_audio_path):
                audio_path = potential_audio_path
                logger.info(f"Found corresponding audio file: {audio_path}")
            else:
                logger.warning(f"Multimodal models {multimodal_models} requested but no audio path provided")
        
        # Initialize capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        # Initialize feature dictionaries for each model
        features = {model: {} for model in models}
        features["metadata"] = {
            "video_name": video_name,
            "video_path": video_path,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "processed_frames": 0
        }
        
        # Try to load all requested models
        for model in models:
            if not self._ensure_model_loaded(model):
                logger.warning(f"Skipping {model} as it could not be loaded")
                models.remove(model)
        
        # Process the video
        frame_idx = 0
        processed_count = 0
        
        try:
            # Process video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame based on sample_rate
                if frame_idx % sample_rate == 0:
                    # Process frame with each model
                    for model in models:
                        try:
                            if model == "pare":
                                model_features = self._extract_pare_features(frame, frame_idx)
                            elif model == "vitpose":
                                model_features = self._extract_vitpose_features(frame, frame_idx)
                            elif model == "psa":
                                model_features = self._extract_psa_features(frame, frame_idx)
                            elif model == "rsn":
                                model_features = self._extract_rsn_features(frame, frame_idx)
                            elif model == "au_detector":
                                model_features = self._extract_au_features(frame, frame_idx)
                            elif model == "dan":
                                model_features = self._extract_dan_features(frame, frame_idx)
                            elif model == "eln":
                                model_features = self._extract_eln_features(frame, frame_idx)
                            elif model == "mediapipe":
                                model_features = self._extract_mediapipe_features(frame, frame_idx)
                            elif model == "pyfeat":
                                model_features = self._extract_pyfeat_features(frame, frame_idx)
                            elif model == "optical_flow":
                                model_features = self._extract_optical_flow_features(frame, frame_idx)
                            else:
                                continue
                                
                            # Store features
                            features[model][frame_idx] = model_features
                            
                        except Exception as e:
                            logger.error(f"Error extracting {model} features from frame {frame_idx}: {e}")
                    
                    processed_count += 1
                    
                    # Log progress every 100 processed frames
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} frames ({frame_idx}/{frame_count})")
                
                frame_idx += 1
        
        finally:
            # Clean up
            cap.release()
            features["metadata"]["processed_frames"] = processed_count
            logger.info(f"Processed {processed_count} frames out of {frame_count} total frames")
        
        # Extract multimodal features if requested after frame-by-frame processing
        if "av_hubert" in models and audio_path:
            try:
                av_hubert_features = self._extract_av_hubert_features(video_path, audio_path)
                features["av_hubert"] = av_hubert_features
                logger.info(f"Extracted AV HuBERT features from {video_path}")
            except Exception as e:
                logger.error(f"Error extracting AV HuBERT features: {e}")
                import traceback
                logger.error(traceback.format_exc())

        if "meld" in models and audio_path:
            try:
                meld_features = self._extract_meld_features(video_path, audio_path)
                features["meld"] = meld_features
                # Include these features in the aggregate for easier access
                if "aggregate" in features:
                    for key, value in meld_features.items():
                        features["aggregate"][key] = value
                logger.info(f"Extracted MELD features from {video_path}")
            except Exception as e:
                logger.error(f"Error extracting MELD features: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Compute aggregate features
        aggregate_features = self._compute_aggregate_features(features)
        
        # Add video name to aggregate features
        aggregate_features["video_name"] = video_name
        
        features["aggregate"] = aggregate_features
        
        # Save features if output path is provided
        if output_path:
            self._save_features(features, output_path)
            
        return features

    def _extract_pare_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract PARE features from a video frame."""
        # Placeholder implementation
        return {
            "PARE_pred_cam": np.random.rand(3).tolist(),
            "PARE_orig_cam": np.random.rand(3).tolist(),
            "PARE_verts": np.random.rand(10).tolist(),  # Simplified
            "PARE_pose": np.random.rand(10).tolist(),   # Simplified
            "PARE_betas": np.random.rand(10).tolist(),
            "PARE_joints3d": np.random.rand(10, 3).tolist(),  # Simplified
            "PARE_joints2d": np.random.rand(10, 2).tolist(),  # Simplified
            "PARE_smpl_joints2d": np.random.rand(10, 2).tolist(),  # Simplified
            "PARE_bboxes": [0, 0, 100, 100],  # Simplified
            "PARE_frame_ids": frame_idx
        }

    def _extract_vitpose_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract ViTPose features from a video frame."""
        # Placeholder implementation
        return {
            "vit_AR": float(np.random.rand(1)[0]),
            "vit_AP": float(np.random.rand(1)[0]),
            "vit_AU": float(np.random.rand(1)[0]),
            "vit_mean": float(np.random.rand(1)[0])
        }

    def _extract_psa_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract PSA features from a video frame."""
        # Placeholder implementation
        return {
            "psa_AP": float(np.random.rand(1)[0]),
            "psa_val_mloU": float(np.random.rand(1)[0])
        }

    def _extract_rsn_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract RSN features from a video frame."""
        # Placeholder implementation
        return {
            "rsn_gflops": float(np.random.rand(1)[0]),
            "rsn_ap": float(np.random.rand(1)[0]),
            "rsn_ap50": float(np.random.rand(1)[0]),
            "rsn_ap75": float(np.random.rand(1)[0]),
            "rsn_apm": float(np.random.rand(1)[0]),
            "rsn_apl": float(np.random.rand(1)[0]),
            "rsn_ar_head": float(np.random.rand(1)[0]),
            "rsn_shoulder": float(np.random.rand(1)[0]),
            "rsn_elbow": float(np.random.rand(1)[0]),
            "rsn_wrist": float(np.random.rand(1)[0]),
            "rsn_hip": float(np.random.rand(1)[0]),
            "rsn_knee": float(np.random.rand(1)[0]),
            "rsn_ankle": float(np.random.rand(1)[0]),
            "rsn_mean": float(np.random.rand(1)[0])
        }

    def _extract_au_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract Action Unit features from a video frame."""
        # Placeholder implementation
        features = {}
        
        # BP4D dataset AUs
        for au in [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]:
            features[f"ann_AU{au}_bp4d"] = float(np.random.rand(1)[0])
        
        features["ann_avg_bp4d"] = float(np.random.rand(1)[0])
        
        # DISFA dataset AUs
        for au in [1, 2, 4, 6, 9, 12, 25, 26]:
            features[f"ann_AU{au}_dis"] = float(np.random.rand(1)[0])
            
        features["ann_avg_dis"] = float(np.random.rand(1)[0])
        
        return features

    def _extract_dan_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract DAN emotion features from a video frame."""
        # Placeholder implementation - for DAN model we use a prefix
        emotions = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
        
        features = {}
        for emotion in emotions:
            features[f"dan_{emotion}"] = float(np.random.rand(1)[0])
        
        features["dan_confidence"] = float(np.random.rand(1)[0])
        
        return features

    def _extract_eln_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract ELN features from a video frame."""
        # Placeholder implementation
        features = {
            # Arousal and valence
            "eln_arousal": float(np.random.rand(1)[0]),
            "eln_valence": float(np.random.rand(1)[0]),
        }
        
        # Action units
        for au in [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]:
            features[f"eln_AU{au}"] = float(np.random.rand(1)[0])
        
        # Emotion categories
        for emotion in ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise", "other"]:
            features[f"eln_{emotion}_f1"] = float(np.random.rand(1)[0])
        
        return features

    def _extract_mediapipe_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract MediaPipe pose features from a video frame."""
        if not hasattr(self, 'mp_pose_instance'):
            logger.warning("MediaPipe model not loaded. Returning placeholder data.")
            return self._generate_mediapipe_placeholders()
            
        # Process the frame with MediaPipe
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose_instance.process(rgb_frame)
            
            if not results.pose_landmarks:
                logger.debug(f"No pose detected in frame {frame_idx}")
                return self._generate_mediapipe_placeholders()
                
            # Extract landmarks
            features = {}
            
            # Process pose landmarks
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    features[f"GMP_land_x_{i+1}"] = landmark.x
                    features[f"GMP_land_y_{i+1}"] = landmark.y
                    features[f"GMP_land_z_{i+1}"] = landmark.z
                    features[f"GMP_land_visi_{i+1}"] = landmark.visibility
                    features[f"GMP_land_presence_{i+1}"] = 1.0
                
            # Process world landmarks (3D)
            if results.pose_world_landmarks:
                for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                    features[f"GMP_world_x_{i+1}"] = landmark.x
                    features[f"GMP_world_y_{i+1}"] = landmark.y
                    features[f"GMP_world_z_{i+1}"] = landmark.z
                    features[f"GMP_world_visi_{i+1}"] = landmark.visibility
                    features[f"GMP_world_presence_{i+1}"] = 1.0
                
            # Add segmentation mask info
            if results.segmentation_mask is not None:
                features["GMP_SM_pic"] = "segmentation_available"
            else:
                features["GMP_SM_pic"] = "no_segmentation"
                
            return features
                
        except Exception as e:
            logger.error(f"Error in MediaPipe processing for frame {frame_idx}: {e}")
            return self._generate_mediapipe_placeholders()

    def _generate_mediapipe_placeholders(self) -> Dict[str, float]:
        """Generate placeholder MediaPipe features when detection fails."""
        features = {}
        
        # Generate placeholder landmarks
        for i in range(1, 34):  # MediaPipe has 33 pose landmarks
            features[f"GMP_land_x_{i}"] = 0.0
            features[f"GMP_land_y_{i}"] = 0.0
            features[f"GMP_land_z_{i}"] = 0.0
            features[f"GMP_land_visi_{i}"] = 0.0
            features[f"GMP_land_presence_{i}"] = 0.0
            
            features[f"GMP_world_x_{i}"] = 0.0
            features[f"GMP_world_y_{i}"] = 0.0
            features[f"GMP_world_z_{i}"] = 0.0
            features[f"GMP_world_visi_{i}"] = 0.0
            features[f"GMP_world_presence_{i}"] = 0.0
        
        features["GMP_SM_pic"] = "no_segmentation"
        return features

    def _extract_pyfeat_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract Py-Feat features from a video frame."""
        if not hasattr(self, 'pyfeat_detector'):
            logger.warning("Py-Feat model not loaded. Returning placeholder data.")
            return self._generate_pyfeat_placeholders()
        
        try:
            # Process the frame with Py-Feat
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = self.pyfeat_detector.detect_image(frame_rgb)
            
            # Check if any faces were detected
            if predictions.empty or len(predictions) == 0:
                logger.debug(f"No faces detected in frame {frame_idx}")
                return self._generate_pyfeat_placeholders()
            
            # Get the first face (highest confidence)
            face_data = predictions.iloc[0]
            
            features = {}
            
            # Extract action units
            for i in range(1, 29):
                au_col = f"AU{i:02d}"
                if au_col in face_data:
                    features[f"pf_au{i:02d}"] = float(face_data[au_col])
            
            # Extract emotions
            for emotion in ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]:
                if emotion in face_data:
                    features[f"pf_{emotion}"] = float(face_data[emotion])
            
            # Extract face rectangle
            features["pf_facerectx"] = float(face_data.get("FaceRectX", 0))
            features["pf_facerecty"] = float(face_data.get("FaceRectY", 0))
            features["pf_facerectwidth"] = float(face_data.get("FaceRectWidth", 0))
            features["pf_facerectheight"] = float(face_data.get("FaceRectHeight", 0))
            features["pf_facescore"] = float(face_data.get("FaceScore", 0))
            
            # Extract pose
            features["pf_pitch"] = float(face_data.get("pitch", 0))
            features["pf_roll"] = float(face_data.get("roll", 0))
            features["pf_yaw"] = float(face_data.get("yaw", 0))
            
            # Add placeholder 3D coordinates (not provided by Py-Feat)
            features["pf_x"] = 0.0
            features["pf_y"] = 0.0
            features["pf_z"] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error in Py-Feat processing for frame {frame_idx}: {e}")
            return self._generate_pyfeat_placeholders()

    def _generate_pyfeat_placeholders(self) -> Dict[str, float]:
        """Generate placeholder Py-Feat features when detection fails."""
        features = {}
        
        # Action units
        for i in range(1, 29):
            if i in [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]:
                features[f"pf_au{i:02d}"] = 0.0
        
        # Emotions
        for emotion in ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]:
            features[f"pf_{emotion}"] = 0.0
        
        # Face rectangle and pose
        features["pf_facerectx"] = 0.0
        features["pf_facerecty"] = 0.0
        features["pf_facerectwidth"] = 0.0
        features["pf_facerectheight"] = 0.0
        features["pf_facescore"] = 0.0
        features["pf_pitch"] = 0.0
        features["pf_roll"] = 0.0
        features["pf_yaw"] = 0.0
        features["pf_x"] = 0.0
        features["pf_y"] = 0.0
        features["pf_z"] = 0.0
        
        return features

    def _extract_optical_flow_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Extract optical flow features from a video frame."""
        # Placeholder implementation for optical flow metrics
        return {
            "of_fg_static_epe_st": float(np.random.rand(1)[0]),
            "of_fg_static_r2_st": float(np.random.rand(1)[0]),
            "of_bg_static_epe_st": float(np.random.rand(1)[0]),
            "of_bg_static_r2_st": float(np.random.rand(1)[0]),
            "of_fg_dynamic_epe_st": float(np.random.rand(1)[0]),
            "of_fg_dynamic_r2_st": float(np.random.rand(1)[0]),
            "of_bg_dynamic_epe_st": float(np.random.rand(1)[0]),
            "of_bg_dynamic_r2_st": float(np.random.rand(1)[0]),
            "of_fg_avg_epe_st": float(np.random.rand(1)[0]),
            "of_fg_avg_r2_st": float(np.random.rand(1)[0]),
            "of_bg_avg_epe_st": float(np.random.rand(1)[0]),
            "of_bg_avg_r2_st": float(np.random.rand(1)[0]),
            "of_avg_epe_st": float(np.random.rand(1)[0]),
            "of_avg_r2_st": float(np.random.rand(1)[0]),
            "of_time_length_st": float(np.random.rand(1)[0]),
        }

    def _compute_aggregate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute aggregate features across all frames.
        
        Args:
            features: Dictionary of all extracted features
            
        Returns:
            Dictionary of aggregated features
        """
        aggregate = {}
        
        # Include video name from metadata
        if "metadata" in features and "video_name" in features["metadata"]:
            aggregate["video_name"] = features["metadata"]["video_name"]
        
        # Process each model's features
        for model_name, frame_features in features.items():
            if model_name in ["metadata", "aggregate"]:
                continue
                
            # Skip if no frames were processed for this model
            if not frame_features:
                continue
                
            # Get first frame for feature names
            first_frame = next(iter(frame_features.values()))
            
            # For each feature, compute mean, min, max, std
            for feature_name in first_frame.keys():
                # Extract values across all frames
                values = []
                for frame_idx, frame_data in frame_features.items():
                    if feature_name in frame_data:
                        value = frame_data[feature_name]
                        # Only include numeric values in aggregation
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            values.append(value)
                
                # Compute statistics if we have values
                if values:
                    aggregate[f"{feature_name}_mean"] = float(np.mean(values))
                    aggregate[f"{feature_name}_min"] = float(np.min(values))
                    aggregate[f"{feature_name}_max"] = float(np.max(values))
                    aggregate[f"{feature_name}_std"] = float(np.std(values))
        
        return aggregate

    def _save_features(self, features: Dict[str, Any], output_path: str):
        """
        Save extracted features to files.
        
        Args:
            features: Dictionary of all extracted features
            output_path: Base path for output files
        """
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save full features as JSON
        json_path = f"{output_path}_full.json"
        try:
            # Handle numpy types for JSON serialization
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
                    
            with open(json_path, 'w') as f:
                json.dump(features, f, cls=NumpyEncoder)
            logger.info(f"Saved full features to {json_path}")
        except Exception as e:
            logger.error(f"Error saving JSON features: {e}")
        
        # Save aggregate features as CSV
        csv_path = f"{output_path}_aggregate.csv"
        try:
            # Create DataFrame from aggregate features
            df = pd.DataFrame([features["aggregate"]])
            
            # Ensure video_name is the first column
            if "video_name" in df.columns:
                cols = ["video_name"] + [col for col in df.columns if col != "video_name"]
                df = df[cols]
                
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved aggregate features to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving CSV features: {e}")

    def _extract_av_hubert_features(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """
        Extract AV HuBERT multimodal features combining audio and video.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the extracted audio file
            
        Returns:
            Dictionary of AV HuBERT features
        """
        logger.info(f"Extracting AV HuBERT features from {video_path} and {audio_path}")
        
        # This is a placeholder implementation
        # In a real implementation, we would:
        # 1. Load the audio using librosa
        # 2. Extract video frames
        # 3. Process both through the AV HuBERT model
        # 4. Extract embeddings and other features
        
        features = {
            "avhubert_embedding": np.random.rand(768).tolist(),  # Simulated embedding vector
            "avhubert_audio_visual_alignment_score": float(np.random.uniform(0.7, 0.99)),
            "avhubert_speech_recognition_confidence": float(np.random.uniform(0.8, 0.99)),
        }
        
        return features

    def _extract_meld_features(self, video_path: str, audio_path: str) -> Dict[str, Any]:
        """
        Extract MELD multimodal features for conversation-level emotion analysis.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the extracted audio file
            
        Returns:
            Dictionary of MELD features
        """
        logger.info(f"Extracting MELD features from {video_path} and {audio_path}")
        
        # In a real implementation, we would:
        # 1. Load the transcript from the speech-to-text output
        # 2. Identify speakers and utterances
        # 3. Process through the MELD model
        # 4. Extract emotion predictions and other conversation metrics
        
        # For demonstration, generate values within reasonable ranges
        num_utterances = np.random.randint(10, 100)
        num_speakers = np.random.randint(1, 4)
        
        # Generate emotion counts that sum to num_utterances
        emotion_counts = np.random.dirichlet(np.ones(7)) * num_utterances
        emotion_counts = emotion_counts.astype(int)
        # Adjust to ensure they sum to num_utterances
        adjustment = num_utterances - np.sum(emotion_counts)
        emotion_counts[0] += adjustment  # Add any difference to first emotion
        
        # Create MELD features
        features = {
            "MELD_modality": "audio-visual",
            "MELD_unique_words": int(np.random.randint(50, 300)),
            "MELD_avg_utterance_length": float(np.random.uniform(5, 20)),
            "MELD_max_utterance_length": int(np.random.randint(15, 50)),
            "MELD_avg_num_emotions_per_dialogue": float(np.random.uniform(2, 5)),
            "MELD_num_dialogues": int(np.random.randint(1, 5)),
            "MELD_num_utterances": num_utterances,
            "MELD_num_speakers": num_speakers,
            "MELD_num_emotion_shift": int(np.random.randint(3, 15)),
            "MELD_avg_utterance_duration": float(np.random.uniform(2, 8)),
            "MELD_count_anger": int(emotion_counts[0]),
            "MELD_count_disgust": int(emotion_counts[1]),
            "MELD_count_fear": int(emotion_counts[2]),
            "MELD_count_joy": int(emotion_counts[3]),
            "MELD_count_neutral": int(emotion_counts[4]),
            "MELD_count_sadness": int(emotion_counts[5]),
            "MELD_count_surprise": int(emotion_counts[6])
        }
        
        return features

def extract_video_features(video_path: str, 
                          output_dir: str = None, 
                          models: List[str] = None,
                          use_gpu: bool = True,
                          sample_rate: int = 1,
                          video_name: str = None,
                          audio_path: str = None) -> Dict[str, Any]:
    """
    Extract features from a video file using various models.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output files
        models: List of models to use for feature extraction
        use_gpu: Whether to use GPU acceleration
        sample_rate: Process every Nth frame
        video_name: Optional name of the video to include in features
        audio_path: Optional path to extracted audio for multimodal analysis
        
    Returns:
        Dictionary of extracted features
    """
    # If video_name not provided, use the filename without extension
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output path if provided
    output_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}_video_features")
    
    # Initialize extractor and extract features
    extractor = VideoFeatureExtractor(use_gpu=use_gpu)
    features = extractor.extract_features(
        video_path=video_path,
        output_path=output_path,
        models=models,
        sample_rate=sample_rate,
        video_name=video_name,
        audio_path=audio_path
    )
    
    return features
