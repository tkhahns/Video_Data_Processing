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

def select_output_format():
    """
    Prompt the user to select the output format for emotion recognition.
    
    Returns:
        bool: True for log-only output, False for video+log output
    """
    while True:
        print("\nWhich output format do you want?")
        print("1. Annotated video + emotion log (slower)")
        print("2. Emotion log only (faster)")
        
        choice = input("\nYour choice (1-2): ").strip()
        
        if choice == '1':
            return False  # Not log-only (produce both video and log)
        elif choice == '2':
            return True   # Log-only mode
        else:
            print("Invalid choice. Please enter 1 or 2.")

def select_files_from_list(file_list):
    """
    Allow user to select files from a provided list.
    
    Args:
        file_list: List of file paths to select from
        
    Returns:
        Tuple of (selected file paths, log_only mode)
    """
    # Sort files alphabetically
    all_files = sorted(file_list)
    
    # Print the list of available files
    print(f"\nAvailable video files:")
    for i, file_path in enumerate(all_files, 1):
        # Get just the filename part without directory path
        display_name = os.path.basename(file_path)
        print(f"{i}. {display_name}")
    
    selected_files = []
    
    # Prompt for selection
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
    
    # Now prompt for output format
    log_only = select_output_format()
    
    return selected_files, log_only
