"""
Utility functions for the emotion_recognition module.
"""
import os
import gc
import logging
import torch

# Try importing from utils package
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

def clean_memory():
    """Clean up memory by calling garbage collector and emptying CUDA cache if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        logger.warning("psutil not installed. Cannot monitor memory usage.")
        return 0

def check_dependencies():
    """
    Check if all required dependencies for emotion recognition are installed.
    
    Returns:
        bool: True if all dependencies are available, False otherwise.
    """
    missing_deps = []
    
    # Check for OpenCV
    try:
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
    
    # Check for DeepFace
    try:
        from deepface import DeepFace
        # Print DeepFace version if available
        try:
            import deepface
            logger.info(f"DeepFace version: {deepface.__version__}")
        except (ImportError, AttributeError):
            logger.info("DeepFace is available (version unknown)")
    except ImportError:
        missing_deps.append("deepface")
    
    # Check for NumPy
    try:
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    # Check for TensorFlow
    try:
        import tensorflow
        logger.info(f"TensorFlow version: {tensorflow.__version__}")
        
        # For TF 2.13+, check for tf-keras
        major, minor = map(int, tensorflow.__version__.split('.')[:2])
        if (major == 2 and minor >= 13) or major > 2:
            try:
                import tf_keras
                logger.info("tf-keras is available")
            except ImportError:
                logger.warning("TensorFlow 2.13+ detected but tf-keras not found")
                missing_deps.append("tf-keras")
    except ImportError:
        missing_deps.append("tensorflow")
    
    # Check for MediaPipe (for body pose estimation)
    try:
        import mediapipe
        logger.info(f"MediaPipe version: {mediapipe.__version__}")
    except ImportError:
        logger.warning("MediaPipe not installed. Body pose estimation will not be available.")
        logger.warning("Install with: pip install mediapipe")
        # Not considered a critical dependency
        
    # Check for SciPy (used for speaker tracking)
    try:
        import scipy
        logger.info(f"SciPy version: {scipy.__version__}")
    except ImportError:
        logger.warning("SciPy not installed. Multi-speaker tracking may be less accurate.")
        logger.warning("Install with: pip install scipy")
    
    # Optional: Check for tqdm for progress bars
    try:
        import tqdm
        logger.info(f"tqdm version: {tqdm.__version__}")
    except ImportError:
        logger.info("tqdm not installed (optional for progress bars)")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def get_available_backends():
    """
    Get a list of available face detection backends in DeepFace.
    
    Returns:
        list: List of available backends
    """
    try:
        from deepface.detectors import detector_factory
        available_backends = detector_factory.get_available_detectors()
        return available_backends
    except Exception as e:
        logger.error(f"Error getting available backends: {e}")
        return ["opencv", "ssd", "mtcnn", "retinaface", "mediapipe"]

def find_video_files(paths=None, recursive=False):
    """
    Find all video files in the given paths.
    
    Args:
        paths: List of file or directory paths to search
        recursive: Whether to search directories recursively
        
    Returns:
        List of video file paths
    """
    import os
    from pathlib import Path
    
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    # If no paths provided, use default video directory
    if not paths:
        paths = ['data/videos']
        # Create the default directory if it doesn't exist
        os.makedirs('data/videos', exist_ok=True)
    
    # Convert to list if a single path is provided
    if isinstance(paths, (str, Path)):
        paths = [paths]
    
    for path in paths:
        path = Path(path)
        
        # Create the directory if it doesn't exist
        if not path.exists() and not path.is_file():
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
        
        if path.is_file() and path.suffix.lower() in supported_formats:
            # Single file that matches supported formats
            video_files.append(path)
        elif path.is_dir():
            # Directory - search for matching files
            if recursive:
                for file_path in path.glob('**/*'):
                    if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                        video_files.append(file_path)
            else:
                for file_path in path.glob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                        video_files.append(file_path)
    
    return video_files

def select_files_from_list(file_list, batch_mode=False):
    """
    Allow user to select files from a provided list.
    
    Args:
        file_list: List of file paths to select from
        batch_mode: If True, automatically select all files without prompting
        
    Returns:
        Tuple of (selected file paths, log_only mode)
    """
    # Sort files alphabetically
    all_files = sorted(file_list)
    
    # In batch mode, return all files without prompting
    if batch_mode:
        logger.info(f"Batch mode: automatically selecting all {len(all_files)} files")
        return all_files, False
    
    # Print the list of available files
    print(f"\nAvailable video files:")
    for i, file_path in enumerate(all_files, 1):
        # Get just the filename part without directory path
        display_name = os.path.basename(file_path)
        print(f"{i}. {display_name}")
    
    selected_files = []
    
    # Prompt for selection
    try:
        while True:
            print("\nOptions:")
            print("- Enter numbers (e.g., '1,3,5') to select specific files")
            print("- Enter 'all' to process all files")
            print("- Enter 'q' to quit")
            
            choice = input("\nYour selection: ").strip()
            
            if choice.lower() == 'q':
                print("Quitting...")
                return [], False
            
            if choice.lower() == 'all':
                print(f"Selected all {len(all_files)} files")
                selected_files = all_files
                break
            
            try:
                # Parse the selection
                indices = [int(idx.strip()) for idx in choice.split(',') if idx.strip()]
                
                # Validate indices
                valid_indices = []
                for idx in indices:
                    if 1 <= idx <= len(all_files):
                        valid_indices.append(idx - 1)  # Convert to 0-based index
                    else:
                        print(f"Warning: {idx} is not a valid file number")
                
                if not valid_indices:
                    print("No valid files selected, please try again")
                    continue
                
                # Get the selected files
                selected_files = [all_files[idx] for idx in valid_indices]
                
                # Print the selected files
                print(f"\nSelected {len(selected_files)} files:")
                for i, file_path in enumerate(selected_files, 1):
                    # Get just the filename part for display
                    display_name = os.path.basename(file_path)
                    print(f"{i}. {display_name}")
                
                # Proceeding without confirmation
                break
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
    except EOFError:
        # Handle EOF error gracefully when in non-interactive environment
        logger.warning("EOF error detected during file selection - falling back to selecting all files")
        print("Non-interactive environment detected, selecting all files")
        selected_files = all_files
    
    # Now prompt for output format
    log_only = False
    
    return selected_files, log_only

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as [x, y, width, height].
    
    Args:
        box1: First bounding box [x, y, width, height]
        box2: Second bounding box [x, y, width, height]
        
    Returns:
        float: IoU score between 0 and 1
    """
    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    # Check if boxes overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

def assign_speaker_ids(current_faces, previous_speakers=None, iou_threshold=0.3):
    """
    Assign consistent speaker IDs to detected faces based on their positions.
    
    Args:
        current_faces: List of faces with 'region' data from DeepFace
        previous_speakers: Dictionary mapping speaker_id to face data from previous frame
        iou_threshold: Threshold for considering a face to be the same speaker
        
    Returns:
        dict: Updated speakers dictionary mapping IDs to face data
    """
    if not current_faces:
        return {}
    
    if not previous_speakers:
        # First frame or no previous speakers, assign new IDs
        speakers = {}
        for i, face in enumerate(current_faces[:2]):  # Limit to max 2 speakers
            speaker_id = f"speaker{i+1}"
            face['speaker_id'] = speaker_id
            speakers[speaker_id] = face
        return speakers
    
    # We have previous speakers, try to match with current faces
    speakers = {}
    unassigned_faces = list(current_faces)
    
    # For each previous speaker, find the best matching face
    for speaker_id, prev_face in previous_speakers.items():
        prev_region = prev_face['region']
        prev_box = [prev_region['x'], prev_region['y'], prev_region['w'], prev_region['h']]
        
        best_match = None
        best_iou = iou_threshold  # Minimum threshold
        best_idx = -1
        
        # Find the face with highest IoU
        for i, face in enumerate(unassigned_faces):
            region = face['region']
            current_box = [region['x'], region['y'], region['w'], region['h']]
            iou = calculate_iou(prev_box, current_box)
            
            if iou > best_iou:
                best_match = face
                best_iou = iou
                best_idx = i
        
        if best_match:
            # Assign the previous speaker_id to this face
            best_match['speaker_id'] = speaker_id
            speakers[speaker_id] = best_match
            # Remove the assigned face
            unassigned_faces.pop(best_idx)
    
    # Assign new IDs to any remaining unassigned faces (up to 2 total speakers)
    existing_ids = set(speakers.keys())
    available_ids = [f"speaker{i+1}" for i in range(2) if f"speaker{i+1}" not in existing_ids]
    
    for i, face in enumerate(unassigned_faces):
        if i < len(available_ids):
            speaker_id = available_ids[i]
            face['speaker_id'] = speaker_id
            speakers[speaker_id] = face
    
    return speakers

def get_speaker_colors():
    """
    Returns consistent colors for each speaker for visualization purposes.
    
    Returns:
        dict: Speaker colors mapping speaker IDs to BGR color tuples
    """
    return {
        "speaker1": (0, 255, 0),   # Green for speaker 1 (BGR format)
        "speaker2": (0, 0, 255)    # Red for speaker 2 (BGR format)
    }
