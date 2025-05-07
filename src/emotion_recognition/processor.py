"""
Core functionality for detecting faces and recognizing emotions in videos.
"""
import os
import cv2
import logging
import numpy as np
import torch
from deepface import DeepFace
from pathlib import Path
from . import utils

# Try importing from utils package
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

def process_video(input_path, output_path=None, log_path=None, show_preview=False, 
                  skip_frames=0, backend="opencv", model_name="emotion", log_only=False):
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
        log_file.write('time_seconds,frame_number,emotions\n')
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

            # Log emotions every 1 second
            current_second = int(frame_count / fps)
            if log_file and current_second != last_logged_second:
                emotions = []
                for face_idx, face in enumerate(faces):
                    emotion = face['dominant_emotion']
                    emotions.append(f"face{face_idx+1}:{emotion}")
                
                log_file.write(f"{current_second},{frame_count}," + ";".join(emotions) + "\n")
                last_logged_second = current_second

            # Only perform video-related operations if not in log_only mode
            if (output_path and not log_only) or show_preview:
                # Annotate frame
                for face_idx, face in enumerate(faces):
                    emotion = face['dominant_emotion']
                    region = face['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw emotion label
                    cv2.putText(
                        frame, 
                        f"{face_idx+1}: {emotion}", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2
                    )

                if output_path and not log_only:
                    out.write(frame)
                    
                if show_preview:
                    cv2.imshow('Emotion Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            frame_count += 1
            
            # Free up memory periodically
            if frame_count % 100 == 0:
                utils.clean_memory()

        logger.info(f"Processed {frame_count} frames")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return False
        
    finally:
        # Clean up resources
        if progress_bar:
            progress_bar.close()
            
        cap.release()
        if output_path and 'out' in locals():
            out.release()
        if log_file:
            log_file.close()
        if show_preview:
            cv2.destroyAllWindows()
    
    logger.info("Video processing complete")
    return True

def batch_process_videos(input_dir, output_dir, log_dir=None, file_extension="mp4"):
    """
    Process all videos with the specified extension in the input directory.
    
    Args:
        input_dir (str): Directory containing input videos
        output_dir (str): Directory to save output videos
        log_dir (str, optional): Directory to save emotion logs
        file_extension (str): File extension to filter input videos
        
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
        
        output_path = os.path.join(output_dir, f"{base_name}_emotions.mp4")
        log_path = os.path.join(log_dir, f"{base_name}_emotions.csv") if log_dir else None
        
        logger.info(f"Processing {video_name}...")
        success = process_video(
            input_path, 
            output_path, 
            log_path,
            backend="opencv"  # Explicitly set backend to most reliable option
        )
        results[video_name] = success
        
        # Clean memory after each video
        utils.clean_memory()
    
    return results
