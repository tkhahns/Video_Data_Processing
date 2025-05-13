"""
Core functionality for detecting faces and recognizing emotions in videos.
"""
import os
import cv2
import logging
import numpy as np
import torch
import json
from deepface import DeepFace
from pathlib import Path
from . import utils
from copy import deepcopy

# Import the body pose estimator
try:
    from .body_pose.estimator import PoseEstimator
    from .body_pose.utils import overlay_pose_text, save_pose_data
    POSE_AVAILABLE = True
except ImportError:
    # Fall back to standard logging if not already defined
    if 'logger' not in locals():
        logger = logging.getLogger(__name__)
    logger.warning("Body pose estimation module not available. Install with: pip install mediapipe")
    POSE_AVAILABLE = False

# Try importing from utils package
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

# Update the pose connections to match MediaPipe's actual pose model connections
# These pairs represent the indices of landmarks that should be connected
POSE_CONNECTIONS = [
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

def draw_pose_skeleton(frame, landmarks, color):
    """
    Draw skeleton connections between landmarks to visualize body pose.
    
    Args:
        frame: The frame to draw on
        landmarks: List of landmark coordinates
        color: BGR color tuple for the skeleton lines
    """
    if not landmarks:
        return  # No landmarks to draw
    
    # Log the number of valid landmarks
    valid_count = sum(1 for l in landmarks if l is not None)
    logger.debug(f"Drawing skeleton with {valid_count}/{len(landmarks)} valid landmarks")
    
    # Convert landmarks to a format usable by OpenCV
    points = {}  # Use a dict to store only available landmarks
    for i, landmark in enumerate(landmarks):
        if landmark is not None:
            try:
                x, y = int(landmark[0]), int(landmark[1])
                points[i] = (x, y)
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Could not convert landmark {i}: {e}")
    
    # Log the number of valid points
    logger.debug(f"Converted {len(points)} valid points for skeleton drawing")
    
    # Draw the connections for the skeleton
    connections_drawn = 0
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        
        # Only draw if both landmarks exist in our points dict
        if start_idx in points and end_idx in points:
            try:
                cv2.line(frame, points[start_idx], points[end_idx], color, 2)
                connections_drawn += 1
            except Exception as e:
                logger.debug(f"Error drawing line between points {start_idx} and {end_idx}: {e}")
    
    logger.debug(f"Drew {connections_drawn}/{len(POSE_CONNECTIONS)} skeleton connections")
    
    # Manual drawing of key segments for better visibility
    # Draw body center line if possible (vertical line from nose to mid-hip)
    if 0 in points and 23 in points and 24 in points:
        # Calculate mid-hip point
        mid_hip_x = (points[23][0] + points[24][0]) // 2
        mid_hip_y = (points[23][1] + points[24][1]) // 2
        
        # Draw a thicker line for the body center
        cv2.line(frame, points[0], (mid_hip_x, mid_hip_y), color, 3)
    
    # Always draw the points for better visibility
    for point_idx, point in points.items():
        # Use larger circles for key points (shoulders, hips, etc)
        radius = 5 if point_idx in [11, 12, 23, 24] else 3
        cv2.circle(frame, point, radius, color, -1)

def process_person_landmarks(person_landmarks, width, height):
    """
    Process landmarks for a single person from multi-person detection.
    
    Args:
        person_landmarks: Landmarks for a single person
        width: Frame width for coordinate normalization
        height: Frame height for coordinate normalization
        
    Returns:
        list: List of 33 landmarks with None for invalid landmarks
    """
    safe_landmarks = [None] * 33  # Initialize with None values
    
    # Handle different landmark formats
    if isinstance(person_landmarks, dict):
        # Dictionary format with named landmarks
        mapping = {
            'NOSE': 0, 
            'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
            'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
            'LEFT_EAR': 7, 'RIGHT_EAR': 8,
            'MOUTH_LEFT': 9, 'MOUTH_RIGHT': 10,
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
            'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
            'LEFT_INDEX': 19, 'RIGHT_INDEX': 20,
            'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24,
            'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
            'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
        }
        
        # Map each named landmark to the correct index
        for name, landmark_data in person_landmarks.items():
            if name in mapping and isinstance(landmark_data, dict):
                idx = mapping[name]
                if 'x' in landmark_data and 'y' in landmark_data:
                    x_coord = landmark_data['x']
                    y_coord = landmark_data['y']
                    
                    # Scale normalized coordinates if needed
                    if isinstance(x_coord, float) and 0 <= x_coord <= 1.0:
                        x_coord *= width
                    if isinstance(y_coord, float) and 0 <= y_coord <= 1.0:
                        y_coord *= height
                        
                    safe_landmarks[idx] = (x_coord, y_coord)
    
    elif isinstance(person_landmarks, (list, tuple)):
        # List format with index-based landmarks
        for i, landmark in enumerate(person_landmarks):
            if i >= 33:  # Only use up to 33 landmarks
                break
                
            if landmark is not None:
                try:
                    if isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                        # Enhanced handling for non-numeric values
                        # First check for string values that can't be converted to float
                        if isinstance(landmark[0], str):
                            # Try to convert if it's a numeric string
                            try:
                                landmark[0] = float(landmark[0]) 
                            except ValueError:
                                # If it contains any non-numeric characters, skip this landmark
                                logger.debug(f"Frame {i}: Invalid x-coordinate string: '{landmark[0]}'")
                                safe_landmarks.append(None)
                                continue
                        
                        if isinstance(landmark[1], str):
                            # Try to convert if it's a numeric string
                            try:
                                landmark[1] = float(landmark[1])
                            except ValueError:
                                # If it contains any non-numeric characters, skip this landmark
                                logger.debug(f"Frame {i}: Invalid y-coordinate string: '{landmark[1]}'")
                                safe_landmarks.append(None)
                                continue
                        
                        # Handle common directional indicators or special values
                        for j in range(2):
                            if isinstance(landmark[j], str) and landmark[j].upper() in ['N', 'NONE', 'R', 'L', 'RIGHT', 'LEFT', 'M', 'MIDDLE', 'D', 'DOWN', 'U', 'UP']:
                                safe_landmarks.append(None)
                                continue
                            
                        # Ensure landmark coordinates are numeric and adjust to original frame
                        x_coord = float(landmark[0])
                        y_coord = float(landmark[1])
                        
                        # Skip invalid coordinates
                        if not (0 <= x_coord < width and 0 <= y_coord < height):
                            safe_landmarks.append(None)
                            continue
                            
                        safe_landmarks[i] = (x_coord, y_coord)
                    elif isinstance(landmark, dict) and 'x' in landmark and 'y' in landmark:
                        x_coord = landmark['x']
                        y_coord = landmark['y']
                        
                        # Scale normalized coordinates if needed
                        if isinstance(x_coord, float) and 0 <= x_coord <= 1.0:
                            x_coord *= width
                        if isinstance(y_coord, float) and 0 <= y_coord <= 1.0:
                            y_coord *= height
                            
                        safe_landmarks[i] = (x_coord, y_coord)
                except (TypeError, ValueError, IndexError):
                    safe_landmarks[i] = None
    
    return safe_landmarks

def process_video(input_path, output_path=None, log_path=None, show_preview=False, 
                  skip_frames=0, backend="opencv", model_name="emotion", log_only=False,
                  with_pose=True, pose_log_path=None):
    """
    Process an input video to detect faces and recognize emotions frame by frame.
    Optionally saves annotated video and logs emotions every 1 second.

    Args:
        input_path (str): Path to the input mp4 video file.
        output_path (str, optional): Path to save the annotated output video. If None, shows live preview.
        log_path (str, optional): Path to save a txt log of emotions (one entry per second).
        show_preview (bool): Whether to show a live preview window.
        skip_frames (int): Number of frames to skip between processing (0 = process every frame).
        backend (str): Face detection backend ("opencv", "ssd", "mtcnn", etc.).
        model_name (str): Emotion recognition model to use.
        log_only (bool): If True, only generate the log file, skip video output entirely.
        with_pose (bool): If True, also perform body pose estimation.
        pose_log_path (str, optional): Path to save pose data log (JSON).
    
    Returns:
        bool: True if processing completed successfully, False otherwise.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input video file not found: {input_path}")
        return False
    
    # Create default output paths if not specified
    if not log_only and output_path is None:
        input_basename = os.path.basename(input_path)
        input_name = os.path.splitext(input_basename)[0]
        default_output_dir = os.path.join(os.path.dirname(input_path), "emotions_output")
        output_path = os.path.join(default_output_dir, f"{input_name}_emotions.mp4")
        logger.info(f"No output path specified. Will use: {output_path}")
    
    if log_path is None:
        log_dir = os.path.dirname(output_path) if output_path else os.path.join(os.path.dirname(input_path), "emotions_output")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        log_path = os.path.join(log_dir, f"{input_name}_emotions.csv")
        logger.info(f"No log path specified. Will use: {log_path}")
    
    # Create default pose log path if pose estimation is enabled
    if with_pose and pose_log_path is None:
        log_dir = os.path.dirname(log_path)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        pose_log_path = os.path.join(log_dir, f"{input_name}_pose.json")
        logger.info(f"No pose log path specified. Will use: {pose_log_path}")
    
    # Set output_path to None if log_only mode is enabled
    if log_only:
        output_path = None
        show_preview = False
        logger.info("Log-only mode enabled: skipping video output for faster processing")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing video: {input_path}")
    logger.info(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")

    # Initialize speaker tracking
    previous_speakers = None
    speaker_colors = utils.get_speaker_colors()
    
    # Initialize pose estimator dict to track different speakers
    pose_estimators = {}
    # Add a global pose estimator as fallback
    global_pose_estimator = None
    pose_data_by_second = {}
    speaker_pose_data_by_second = {}  # Track pose data separately for each speaker
    
    if with_pose and POSE_AVAILABLE:
        logger.info("Multi-speaker pose estimation enabled")
        # Initialize global pose estimator as fallback
        logger.info("Initializing global pose estimator as fallback")
        global_pose_estimator = PoseEstimator(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False  # Disable segmentation to avoid errors
        )
    elif with_pose:
        logger.warning("Body pose estimation requested but mediapipe is not available. Install with: pip install mediapipe")
        with_pose = False

    # Only initialize video writer if not in log_only mode
    if output_path and not log_only:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Will save output to: {output_path}")
    else:
        if log_only and output_path:
            logger.info("Running in log-only mode, video output will be skipped")

    # Prepare log file
    log_file = None
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = open(log_path, 'w')
        
        # Write header with separate columns for each speaker
        header = 'time_seconds,frame_number,speaker1_emotion,speaker2_emotion'
        if with_pose:
            header += ',speaker1_posture,speaker2_posture'
        log_file.write(header + '\n')
        
        logger.info(f"Will save emotion log to: {log_path}")

    frame_count = 0
    last_logged_second = -1
    skip_count = 0
    
    # Use tqdm if available for progress display
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_frames, desc="Processing")
    except ImportError:
        progress_bar = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            if progress_bar:
                progress_bar.update(1)
                
            # Skip frames if requested (for faster processing)
            if skip_frames > 0 and skip_count < skip_frames:
                skip_count += 1
                frame_count += 1
                continue
            else:
                skip_count = 0

            # Make a copy of frame for each pose estimator
            original_frame = frame.copy()

            # Analyze emotions in the current frame
            faces = []
            try:
                results = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend=backend
                )
                faces = results if isinstance(results, list) else [results]
            except Exception as e:
                logger.warning(f"Frame {frame_count}: Error analyzing emotions: {e}")
                faces = []
            
            # Assign consistent speaker IDs to detected faces
            speakers = utils.assign_speaker_ids(faces, previous_speakers)
            previous_speakers = speakers
            
            # Process body pose for each speaker if enabled
            speaker_pose_data = {}
            skeletons_drawn = 0
            if with_pose and POSE_AVAILABLE:
                # First try speaker-specific pose estimation
                for speaker_id, face_data in list(speakers.items()):  # Use list() to create a copy of keys
                    # Create pose estimator for this speaker if not exists
                    if speaker_id not in pose_estimators:
                        pose_estimators[speaker_id] = PoseEstimator(
                            static_image_mode=False,
                            model_complexity=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            enable_segmentation=False  # Disable segmentation to avoid errors
                        )
                    
                    # Create a region of interest around the face for pose estimation
                    region = face_data['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    # Convert region values to integers to prevent type issues
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    
                    # Expand ROI to capture more of the body - MUCH LARGER TO INCLUDE LEGS
                    # Convert all calculations to integers to avoid slice index errors
                    roi_x = max(0, int(x - w*1.5))                   # Wider ROI to capture arms
                    roi_y = max(0, int(y - h))                       # Start higher above the face
                    roi_w = min(width - roi_x, int(w * 4))           # Make wider for better arm detection
                    roi_h = min(height - roi_y, int(height - roi_y)) # Extend to bottom of frame to capture legs
                    
                    # Error fix: Ensure roi_h is calculating correctly (previous logic was redundant)
                    roi_h = height - roi_y  # Simply extend to the bottom of the frame
                    
                    # Ensure all ROI values are integers
                    roi_x, roi_y, roi_w, roi_h = int(roi_x), int(roi_y), int(roi_w), int(roi_h)
                    
                    # Check if ROI is too small or too large (use full frame for very large faces)
                    if roi_w < 100 or roi_h < 200 or w > width/3:
                        logger.debug(f"Frame {frame_count}: ROI inappropriate for {speaker_id}, using full frame")
                        roi = original_frame.copy()
                        roi_x, roi_y = 0, 0  # Reset ROI offset
                    else:
                        # Extract region of interest
                        roi = original_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                        if roi.size == 0:  # Skip if ROI is empty
                            logger.warning(f"Empty ROI for speaker {speaker_id}, skipping pose detection")
                            continue
                    
                    # Process pose in the ROI
                    try:
                        # Process the ROI for pose estimation
                        _, pose_data = pose_estimators[speaker_id].process_frame(roi)
                        
                        # Ensure pose_data is a dictionary
                        if not isinstance(pose_data, dict):
                            logger.warning(f"Frame {frame_count}: Invalid pose data type for {speaker_id}")
                            continue
                        
                        # Adjust landmark coordinates to global frame coordinates
                        if 'landmarks' in pose_data:
                            # Debug landmark structure before processing
                            logger.debug(f"Frame {frame_count}: Speaker {speaker_id} has {len(pose_data['landmarks'])} landmarks")
                            
                            safe_landmarks = []
                            # Make a copy of landmarks to avoid modifying during iteration
                            landmarks_copy = list(pose_data['landmarks'])
                            for i, landmark in enumerate(landmarks_copy):
                                if landmark is not None:  # Some landmarks might be None
                                    try:
                                        # Enhanced handling for non-numeric values
                                        # First check for string values that can't be converted to float
                                        if isinstance(landmark[0], str):
                                            # Try to convert if it's a numeric string
                                            try:
                                                landmark[0] = float(landmark[0]) 
                                            except ValueError:
                                                # If it contains any non-numeric characters, skip this landmark
                                                logger.debug(f"Frame {frame_count}: Invalid x-coordinate string: '{landmark[0]}'")
                                                safe_landmarks.append(None)
                                                continue
                                        
                                        if isinstance(landmark[1], str):
                                            # Try to convert if it's a numeric string
                                            try:
                                                landmark[1] = float(landmark[1])
                                            except ValueError:
                                                # If it contains any non-numeric characters, skip this landmark
                                                logger.debug(f"Frame {frame_count}: Invalid y-coordinate string: '{landmark[1]}'")
                                                safe_landmarks.append(None)
                                                continue
                                        
                                        # Handle common directional indicators or special values
                                        for j in range(2):
                                            if isinstance(landmark[j], str) and landmark[j].upper() in ['N', 'NONE', 'R', 'L', 'RIGHT', 'LEFT', 'M', 'MIDDLE', 'D', 'DOWN', 'U', 'UP']:
                                                safe_landmarks.append(None)
                                                continue
                                            
                                        # Ensure landmark coordinates are numeric and adjust to original frame
                                        x_coord = float(landmark[0]) + roi_x
                                        y_coord = float(landmark[1]) + roi_y
                                        
                                        # Skip invalid coordinates
                                        if not (0 <= x_coord < width and 0 <= y_coord < height):
                                            safe_landmarks.append(None)
                                            continue
                                            
                                        safe_landmarks.append((x_coord, y_coord))
                                    except (TypeError, ValueError, IndexError) as e:
                                        # More detailed error logging
                                        if isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                                            logger.warning(f"Frame {frame_count}: Invalid landmark format: {e}, values: {landmark[0]}, {landmark[1]}")
                                        else:
                                            logger.warning(f"Frame {frame_count}: Invalid landmark structure: {type(landmark)}")
                                        # Add None instead of skipping to maintain proper indexing
                                        safe_landmarks.append(None)
                                else:
                                    safe_landmarks.append(None)
                            
                            # Fill any remaining indices to maintain full landmark array
                            # MediaPipe pose has 33 landmarks
                            while len(safe_landmarks) < 33:
                                safe_landmarks.append(None)
                                
                            # Check if we have at least some minimal valid landmarks
                            valid_count = sum(1 for lm in safe_landmarks if lm is not None)
                            if valid_count < 5:  # Need at least a few points to draw anything meaningful
                                logger.debug(f"Not enough valid landmarks ({valid_count}) for {speaker_id}")
                                continue
                                
                            # Replace landmarks with safe version
                            pose_data['landmarks'] = safe_landmarks
                        
                        # Store pose data for this speaker
                        speaker_pose_data[speaker_id] = pose_data
                        
                        # Draw the pose skeleton and landmarks
                        if 'landmarks' in pose_data and pose_data['landmarks'] and any(pose_data['landmarks']):
                            color = speaker_colors.get(speaker_id, (255, 255, 255))
                            
                            # Force debug output for first few frames
                            if frame_count < 10:
                                logger.info(f"Frame {frame_count}: Drawing pose for {speaker_id} with {len(pose_data['landmarks'])} landmarks")
                            
                            # Draw skeleton connections for better visualization
                            draw_pose_skeleton(frame, pose_data['landmarks'], color)
                            
                            # Add label for clearer speaker identification near the pose
                            # Find a good position near the top of the skeleton
                            top_landmarks = []
                            for i, landmark in enumerate(pose_data['landmarks']):
                                if landmark and i < 11:  # Check the face/upper body landmarks
                                    top_landmarks.append(landmark)
                                    
                            if top_landmarks:
                                # Use the average position of upper body landmarks
                                avg_x = sum(l[0] for l in top_landmarks) // len(top_landmarks)
                                avg_y = min(l[1] for l in top_landmarks) - 15  # Position above the top landmark
                                
                                # Draw text with speaker identity
                                cv2.putText(
                                    frame,
                                    speaker_id,
                                    (int(avg_x), int(avg_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    color,
                                    2
                                )
                            
                            # Track if successful pose was drawn
                            if 'landmarks' in pose_data and pose_data['landmarks'] and any(pose_data['landmarks']):
                                skeletons_drawn += 1
                    except Exception as e:
                        logger.warning(f"Frame {frame_count}: Error processing body pose for {speaker_id}: {str(e)}")
                        # Add detailed debug info for troubleshooting
                        import traceback
                        logger.debug(f"Detailed error: {traceback.format_exc()}")
                
                # If no skeletons drawn or less than number of speakers, try global fallback
                if skeletons_drawn < len(speakers) and global_pose_estimator:
                    try:
                        logger.debug(f"Frame {frame_count}: Using global pose estimator as fallback for {len(speakers) - skeletons_drawn} missing speakers")
                        
                        # Extract face positions for better matching - do this FIRST before any processing
                        speaker_face_positions = {}
                        speaker_face_sizes = {}
                        for s_id, face_data in speakers.items():
                            region = face_data['region']
                            x, y, w, h = region['x'], region['y'], region['w'], region['h']
                            face_center_x = x + w // 2
                            face_center_y = y + h // 2
                            speaker_face_positions[s_id] = (face_center_x, face_center_y)
                            speaker_face_sizes[s_id] = (w, h)
                            logger.info(f"Frame {frame_count}: Speaker {s_id} face center at ({face_center_x}, {face_center_y}), size {w}x{h}")
                        
                        # Process the whole frame with the global pose estimator - with error handling
                        try:
                            processed_frame, global_pose_data = global_pose_estimator.process_frame(original_frame.copy())
                        except Exception as e:
                            logger.warning(f"Frame {frame_count}: Global estimator failed: {str(e)}")
                            logger.info("Creating fallback pose data and continuing...")
                            global_pose_data = {"landmarks": [], "multi_person_landmarks": []}
                        
                        # If we still have speakers without poses, create artificial poses positioned exactly at faces
                        remaining_speakers = [s for s in speakers if s not in speaker_pose_data]
                        if remaining_speakers:
                            logger.info(f"Frame {frame_count}: Creating artificial poses for {remaining_speakers}")
                            
                            for speaker_id in remaining_speakers:
                                # If speaker face position is available, create a simple pose at that location
                                if speaker_id in speaker_face_positions:
                                    face_x, face_y = speaker_face_positions[speaker_id]
                                    face_w, face_h = speaker_face_sizes[speaker_id]
                                    
                                    # Create basic pose landmarks centered at face position - ensure all values are integers
                                    simple_landmarks = [None] * 33
                                    
                                    # Basic face landmarks - convert all calculations to integers
                                    simple_landmarks[0] = (int(face_x), int(face_y))  # Nose at face center
                                    
                                    # Left eye area
                                    simple_landmarks[2] = (int(face_x - face_w*0.15), int(face_y - face_h*0.1))  # Left eye
                                    
                                    # Right eye area
                                    simple_landmarks[5] = (int(face_x + face_w*0.15), int(face_y - face_h*0.1))  # Right eye
                                    
                                    # Shoulders
                                    shoulder_y = int(face_y + face_h*0.6)
                                    simple_landmarks[11] = (int(face_x - face_w*0.4), shoulder_y)  # Left shoulder
                                    simple_landmarks[12] = (int(face_x + face_w*0.4), shoulder_y)  # Right shoulder
                                    
                                    # Hips - positioned lower
                                    hip_y = int(shoulder_y + face_h*1.5)
                                    simple_landmarks[23] = (int(face_x - face_w*0.3), hip_y)  # Left hip
                                    simple_landmarks[24] = (int(face_x + face_w*0.3), hip_y)  # Right hip
                                    
                                    # Knees - positioned much lower
                                    knee_y = int(hip_y + face_h*1.8)
                                    knee_y = min(knee_y, int(height - face_h*1.0))  # Ensure within frame
                                    simple_landmarks[25] = (int(face_x - face_w*0.35), knee_y)  # Left knee
                                    simple_landmarks[26] = (int(face_x + face_w*0.35), knee_y)  # Right knee
                                    
                                    # Ankles - positioned at the bottom
                                    ankle_y = int(knee_y + face_h*1.8)  
                                    ankle_y = min(ankle_y, int(height - face_h*0.5))  # Ensure within frame
                                    simple_landmarks[27] = (int(face_x - face_w*0.4), ankle_y)  # Left ankle
                                    simple_landmarks[28] = (int(face_x + face_w*0.4), ankle_y)  # Right ankle
                                    
                                    # Instead of hardcoding a posture, create a dictionary representation
                                    # that can be analyzed by the same posture detection algorithm
                                    simple_landmarks_dict = {}
                                    landmark_names = [
                                        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
                                        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
                                        "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
                                        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                                        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
                                        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
                                        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", 
                                        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
                                        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
                                    ]
                                    
                                    for i, landmark in enumerate(simple_landmarks):
                                        if landmark is not None and i < len(landmark_names):
                                            name = landmark_names[i]
                                            simple_landmarks_dict[name] = {
                                                "x": landmark[0],
                                                "y": landmark[1],
                                                "z": 0.0,  # Default z value
                                                "visibility": 1.0  # Full visibility for artificial points
                                            }
                                    
                                    # Use PoseEstimator's analyze posture method
                                    simple_posture = None
                                    if global_pose_estimator:
                                        simple_posture = global_pose_estimator._analyze_posture(simple_landmarks_dict)
                                    else:
                                        # If no estimator, use a basic heuristic
                                        if "NOSE" in simple_landmarks_dict and "LEFT_ANKLE" in simple_landmarks_dict:
                                            # If large vertical distance between nose and ankle, likely standing
                                            nose_y = simple_landmarks_dict["NOSE"]["y"]
                                            ankle_y = simple_landmarks_dict["LEFT_ANKLE"]["y"]
                                            vertical_distance = abs(ankle_y - nose_y)
                                            is_sitting = vertical_distance < height * 0.4
                                            simple_posture = {"position": "sitting" if is_sitting else "standing"}
                                        else:
                                            simple_posture = {"position": "unknown"}
                                    
                                    # Create basic pose data
                                    simple_pose_data = {
                                        "landmarks": simple_landmarks,
                                        "posture": simple_posture
                                    }
                                    
                                    # IMPORTANT: Remove any previous pose data
                                    if speaker_id in speaker_pose_data:
                                        del speaker_pose_data[speaker_id]
                                    
                                    # Draw the simple pose
                                    color = speaker_colors.get(speaker_id, (0, 0, 255))
                                    draw_pose_skeleton(frame, simple_landmarks, color)
                                    
                                    # Add speaker ID text
                                    cv2.putText(
                                        frame,
                                        f"{speaker_id}",
                                        (int(face_x), int(face_y - face_h*0.2)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        color,
                                        2
                                    )
                                    
                                    # Store pose data
                                    speaker_pose_data[speaker_id] = simple_pose_data
                                    skeletons_drawn += 1
                                    
                                    logger.info(f"Frame {frame_count}: Created basic pose placeholder for {speaker_id}")
                            
                    except Exception as e:
                        logger.warning(f"Frame {frame_count}: Error in global pose fallback: {str(e)}")
                        import traceback
                        logger.debug(f"Global pose estimator error: {traceback.format_exc()}")

            # Log emotions and pose every 1 second
            current_second = int(frame_count / fps)
            if log_file and current_second != last_logged_second:
                # Initialize data for all possible speakers
                speaker_emotions = {"speaker1": "unknown", "speaker2": "unknown"}
                speaker_postures = {"speaker1": "unknown", "speaker2": "unknown"}
                
                # Update with detected data
                for speaker_id, face_data in list(speakers.items()):  # Use list() to create a copy
                    emotion = face_data['dominant_emotion']
                    speaker_emotions[speaker_id] = emotion
                
                    # Add pose data if available
                    if with_pose and speaker_id in speaker_pose_data:
                        pose_data = speaker_pose_data[speaker_id]
                        if 'posture' in pose_data:
                            posture = pose_data['posture']
                            # Handle None values or unexpected types
                            position = str(posture.get('position', 'unknown') or 'unknown')
                            arms = str(posture.get('arms', 'unknown') or 'unknown')
                            speaker_postures[speaker_id] = f"position:{position};arms:{arms}"
                
                # Write to log
                log_line = f"{current_second},{frame_count},{speaker_emotions['speaker1']},{speaker_emotions['speaker2']}"
                
                # Add pose data to log if enabled
                if with_pose:
                    log_line += f",{speaker_postures['speaker1']},{speaker_postures['speaker2']}"
                
                    # Store pose data for this second - ensure all values are JSON serializable
                    safe_pose_data = {}
                    for s_id, pose_data in list(speaker_pose_data.items()):  # Use list() to create a copy
                        if isinstance(pose_data, dict):
                            # Deep copy to avoid modifying the original
                            safe_pose = {}
                            for k, v in list(pose_data.items()):  # Use list() to create a copy
                                if k != 'landmarks':  # Skip landmarks for JSON serialization
                                    # Handle None values or special strings
                                    if v is None:
                                        safe_pose[k] = "unknown"
                                    elif isinstance(v, dict):
                                        # Handle nested dictionaries
                                        safe_dict = {}
                                        for sub_k, sub_v in v.items():
                                            if sub_v is None or (isinstance(sub_v, str) and sub_v.lower() in ['n', 'none']):
                                                safe_dict[sub_k] = "unknown"
                                            else:
                                                safe_dict[sub_k] = sub_v
                                        safe_pose[k] = safe_dict
                                    else:
                                        safe_pose[k] = v
                            safe_pose_data[s_id] = safe_pose
                    
                    speaker_pose_data_by_second[str(current_second)] = {
                        "speaker1": safe_pose_data.get("speaker1", {}),
                        "speaker2": safe_pose_data.get("speaker2", {})
                    }
                
                log_file.write(log_line + "\n")
                last_logged_second = current_second

            # Only perform video-related operations if not in log_only mode
            if (output_path and not log_only) or show_preview:
                # Annotate frame with speaker emotions
                for speaker_id, face_data in list(speakers.items()):  # Use list() to create a copy
                    emotion = face_data['dominant_emotion']
                    region = face_data['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    # Use consistent color for each speaker
                    color = speaker_colors.get(speaker_id, (255, 255, 255))
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Get pose information for this speaker if available
                    pose_info = ""
                    if with_pose and speaker_id in speaker_pose_data and 'posture' in speaker_pose_data[speaker_id]:
                        posture = speaker_pose_data[speaker_id]['posture']
                        position = posture.get('position', '').capitalize() or ''
                        arms = posture.get('arms', '')
                        
                        # Format the arm position nicely
                        if arms:
                            arms = arms.replace('_', ' ').capitalize()
                        
                        # Add pose info if available
                        if position and arms:
                            pose_info = f" | {position}, {arms}"
                        elif position:
                            pose_info = f" | {position}"
                        elif arms:
                            pose_info = f" | Arms: {arms}"
                    
                    # Draw emotion label with speaker ID and pose info
                    cv2.putText(
                        frame, 
                        f"{speaker_id}: {emotion}{pose_info}", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        color, 
                        2
                    )
                
                # Overlay pose information at the top-right for each speaker
                if with_pose:
                    y_offset = 30
                    # Always show info for both possible speakers regardless of pose detection
                    for possible_speaker_id in ["speaker1", "speaker2"]:
                        # Default info if no pose data
                        position = "unknown"
                        arms = "unknown"
                        confidence = 0.0
                        color = speaker_colors.get(possible_speaker_id, (255, 255, 255))
                        
                        # Get pose data if available for this speaker
                        if possible_speaker_id in speaker_pose_data:
                            pose_data = speaker_pose_data[possible_speaker_id]
                            if "posture" in pose_data:
                                posture = pose_data["posture"]
                                position = str(posture.get("position", "unknown") or "unknown")
                                arms = str(posture.get("arms", "unknown") or "unknown")
                                confidence = float(posture.get("confidence", 0.0) or 0.0)
                        
                        # Add speaker info text - always show for both possible speakers
                        text = f"{possible_speaker_id} - "
                        if possible_speaker_id in speakers:
                            if position != "unknown" or arms != "unknown":
                                conf_text = f" (conf: {confidence:.2f})" if confidence > 0 else ""
                                text += f"Posture: {position.capitalize()}{conf_text}, Arms: {arms.replace('_', ' ').capitalize()}"
                            else:
                                text += f"No pose data available"
                        else:
                            text += f"Not detected"
                            
                        cv2.putText(
                            frame,
                            text,
                            (width - 550, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )
                        y_offset += 30  # Move down for next speaker's info

                if output_path and not log_only:
                    out.write(frame)
                    
                if show_preview:
                    cv2.imshow('Multi-Speaker Emotion & Pose Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_count += 1
            
            # Free up memory periodically
            if frame_count % 100 == 0:
                utils.clean_memory()

        logger.info(f"Processed {frame_count} frames")
        
        # Save pose data log if requested
        if with_pose and pose_log_path and speaker_pose_data_by_second:
            try:
                with open(pose_log_path, 'w') as f:
                    # Use a custom serializer to handle numpy types and potential non-serializable objects
                    def json_serializer(obj):
                        if isinstance(obj, (np.integer, np.floating, np.bool_)):
                            return float(obj)
                        elif isinstance(obj, (np.ndarray,)):
                            return obj.tolist()
                        elif obj is None:
                            return "unknown"
                        # Handle directional indicators and other special strings
                        elif isinstance(obj, str) and obj.upper() in ['N', 'NONE', 'R', 'L', 'RIGHT', 'LEFT', 'M', 'MIDDLE']:
                            return obj.upper()  # Preserve directional information as standardized uppercase strings
                        try:
                            return str(obj)
                        except:
                            return "non-serializable"
                    
                    json.dump(speaker_pose_data_by_second, f, default=json_serializer)
                logger.info(f"Saved multi-speaker pose data log to {pose_log_path}")
            except Exception as e:
                logger.error(f"Error saving pose data log: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return False
        
    finally:
        # Clean up resources
        if progress_bar:
            progress_bar.close()
        
        # Close all pose estimators with improved error handling
        if with_pose and POSE_AVAILABLE:
            try:
                # Close per-speaker estimators
                for speaker_id, pose_estimator in pose_estimators.items():
                    try:
                        pose_estimator.close()
                    except Exception as e:
                        logger.warning(f"Error closing pose estimator for {speaker_id}: {str(e)}")
                
                # Close global estimator if it exists
                if global_pose_estimator:
                    try:
                        global_pose_estimator.close()
                    except Exception as e:
                        logger.warning(f"Error closing global pose estimator: {str(e)}")
            except Exception as e:
                logger.warning(f"Error during pose estimator cleanup: {str(e)}")
            
        cap.release()
        if output_path and 'out' in locals():
            out.release()
        if log_file:
            log_file.close()
        if show_preview:
            cv2.destroyAllWindows()
    
    logger.info("Video processing complete")
    return True

def batch_process_videos(input_dir, output_dir, log_dir=None, file_extension="mp4", with_pose=True):
    """
    Process all videos with the specified extension in the input directory.
    
    Args:
        input_dir (str): Directory containing input videos
        output_dir (str): Directory to save output videos
        log_dir (str, optional): Directory to save emotion logs
        file_extension (str): File extension to filter input videos
        with_pose (bool): Whether to perform pose estimation
        
    Returns:
        dict: Dictionary with processing results for each file
    """
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return {}
        
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create log directory if specified
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        logger.info(f"Created log directory: {log_dir}")
    
    # Find all video files
    input_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(f".{file_extension.lower()}"):
            input_files.append(os.path.join(input_dir, file))
    
    if not input_files:
        logger.warning(f"No {file_extension} files found in {input_dir}")
        return {}
    
    logger.info(f"Found {len(input_files)} video files to process")
    
    # Process each video
    results = {}
    for input_path in input_files:
        video_name = os.path.basename(input_path)
        base_name = os.path.splitext(video_name)[0]
        
        output_path = os.path.join(output_dir, f"{base_name}_emotions_and_pose.mp4")
        log_path = os.path.join(log_dir, f"{base_name}_emotions.csv") if log_dir else None
        pose_log_path = os.path.join(log_dir, f"{base_name}_pose.json") if log_dir and with_pose else None
        
        logger.info(f"Processing {video_name}...")
        success = process_video(
            input_path, 
            output_path, 
            log_path,
            backend="opencv",  # Explicitly set backend to most reliable option
            with_pose=with_pose,
            pose_log_path=pose_log_path
        )
        results[video_name] = success
        
        # Clean memory after each video
        utils.clean_memory()
    
    return results
