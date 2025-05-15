"""
Speaker diarization functionality using pyannote.audio.
"""
import os
import logging
import torch
import tempfile
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Try importing from utils package
try:
    from utils import init_logging
    logger = init_logging.get_logger(__name__)
except ImportError:
    # Fall back to standard logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

def check_huggingface_login():
    """
    Check if user is logged in with huggingface-cli.
    If not, guide them through the login process.
    
    Returns:
        bool: True if login is successful, False otherwise
    """
    try:
        # Check if huggingface_hub is installed
        try:
            from huggingface_hub import HfFolder, whoami
            
            # First, check if token exists in the cache
            token = HfFolder.get_token()
            if token is not None:
                try:
                    # Verify the token is valid
                    user_info = whoami()
                    logger.info(f"Using cached Hugging Face credentials for: {user_info.get('name', 'unknown user')}")
                    return True
                except Exception as e:
                    # Token exists but might be invalid
                    if "401" in str(e) or "unauthorized" in str(e).lower():
                        logger.warning("Cached Hugging Face token is invalid or expired")
                    else:
                        logger.warning(f"Error validating Hugging Face token: {e}")
            
            # Need to perform login
            logger.info("Valid Hugging Face login required. Initiating login process...")
            
            print("\n" + "="*70)
            print("HUGGING FACE AUTHENTICATION REQUIRED".center(70))
            print("="*70)
            print("\nTo use speaker diarization, you need to:")
            print("1. Visit hf.co/pyannote/speaker-diarization and accept user conditions")
            print("2. Visit hf.co/pyannote/segmentation and accept user conditions")
            print("3. Login using the huggingface-cli tool (will open now)")
            
            # Check if user wants to proceed with login
            proceed = input("\nReady to login with huggingface-cli? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Login aborted. Speaker diarization requires Hugging Face authentication.")
                return False
            
            # Run the huggingface-cli login command
            print("\nRunning 'huggingface-cli login'...")
            print("A browser window will open to complete the login process.\n")
            
            import subprocess
            import sys
            try:
                # Running the login command - this will open a browser window
                result = subprocess.run(
                    [sys.executable, "-m", "huggingface_hub.cli.cli", "login"],
                    check=True
                )
                
                # Verify login was successful by checking token again
                if HfFolder.get_token() is not None:
                    print("\nLogin successful!")
                    return True
                else:
                    print("\nLogin process completed but no valid token found.")
                    return False
            except subprocess.SubprocessError as e:
                print(f"\nError running login command: {e}")
                print("You can manually login with: huggingface-cli login")
                return False
                
        except ImportError:
            logger.error("huggingface_hub is not installed. Installing it now...")
            try:
                import subprocess
                import sys
                subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
                print("huggingface_hub installed. Please run the script again.")
                return False
            except subprocess.SubprocessError:
                logger.error("Failed to install huggingface_hub. Install it manually: pip install huggingface_hub")
                return False
    except Exception as e:
        logger.error(f"Error checking Hugging Face login: {e}")
        return False

def show_license_acceptance_instructions():
    """Show detailed instructions for accepting the model license."""
    print("\n" + "="*70)
    print("MODEL LICENSE ACCEPTANCE REQUIRED".center(70))
    print("="*70)
    print("\nYou need to accept the license for both required models:")
    print("\n1. Visit: https://huggingface.co/pyannote/speaker-diarization")
    print("   - Sign in with your Hugging Face account")
    print("   - Click 'Accept license and access repository'")
    print("\n2. Visit: https://huggingface.co/pyannote/segmentation")
    print("   - Sign in with your Hugging Face account")
    print("   - Click 'Accept license and access repository'")
    print("\n3. Then login again with: huggingface-cli login")
    print("\nOnce completed, run this script again.")

def download_required_models(auth_token):
    """
    Explicitly download and cache the required models.
    
    Args:
        auth_token: Hugging Face authentication token
        
    Returns:
        bool: True if all models were downloaded successfully, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
        import os
        
        # Models that need to be downloaded
        models = [
            "pyannote/speaker-diarization",
            "pyannote/segmentation"
        ]
        
        logger.info("Pre-downloading required models:")
        
        for model in models:
            try:
                logger.info(f"Downloading {model}...")
                
                # Explicitly download the model to cache
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                model_dir = snapshot_download(
                    repo_id=model,
                    token=auth_token,
                    cache_dir=cache_dir,
                    local_dir_use_symlinks=False  # Ensure full download, not symlinks
                )
                
                logger.info(f"Successfully downloaded {model} to {model_dir}")
                
            except Exception as e:
                if "401" in str(e):
                    logger.error(f"Authentication error for {model}: Invalid token")
                    return False
                elif "403" in str(e) or "access" in str(e).lower():
                    logger.error(f"Access denied for {model}: License acceptance required")
                    logger.error(f"Please visit https://huggingface.co/{model} and accept the license")
                    return False
                else:
                    logger.error(f"Error downloading {model}: {e}")
                    return False
        
        return True
        
    except ImportError:
        logger.error("huggingface_hub is not installed. Please install it.")
        return False
    except Exception as e:
        logger.error(f"Error in model download: {e}")
        return False

def load_diarization_model(model_path: Optional[Path] = None, device: str = "cpu") -> any:
    """
    Load the pyannote.audio diarization model.
    
    Args:
        model_path: Path to save or load the model from
        device: Device to run the model on ("cpu" or "cuda")
        
    Returns:
        Loaded diarization model or None if loading fails
    """
    try:
        from pyannote.audio import Pipeline
        import torch
        from huggingface_hub import HfFolder
        
        # Check available device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            
        logger.info(f"Loading diarization model on {device}...")
        
        # Check if user is logged in with huggingface-cli
        if not check_huggingface_login():
            return None
        
        # Get the token from huggingface folder
        auth_token = HfFolder.get_token()
        if not auth_token:
            logger.error("No Hugging Face token found after login")
            return None
        
        # First, explicitly download the required models
        if not download_required_models(auth_token):
            logger.error("Failed to download required models")
            show_detailed_license_instructions()
            return None
            
        try:
            # Try to create the pipeline
            pipeline = None
            if model_path is None:
                # According to official documentation, use version 2.1
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=auth_token  # Pass the actual token instead of True
                )
            else:
                # Use a locally saved model
                pipeline = Pipeline.from_pretrained(model_path)
            
            # Move the pipeline to the specified device only if pipeline was created successfully
            if pipeline is not None:
                pipeline.to(torch.device(device))
                logger.info("Diarization model loaded successfully")
                return pipeline
            else:
                logger.error("Pipeline creation failed")
                return None
                
        except Exception as inner_e:
            logger.error(f"Error creating pipeline: {inner_e}")
            return None
                
    except ImportError:
        logger.error("Failed to import pyannote.audio. Install with: pip install pyannote-audio==2.1.1")
        return None
    
    except Exception as e:
        logger.error(f"Failed to load diarization model: {e}")
        return None

def show_detailed_license_instructions():
    """Show comprehensive instructions for accepting the model license."""
    print("\n" + "="*80)
    print("COMPLETE LICENSE ACCEPTANCE INSTRUCTIONS".center(80))
    print("="*80)
    
    print("\nYou must follow ALL these steps in order:")
    
    print("\n1. LOGIN TO HUGGING FACE FIRST:")
    print("   a. Go to https://huggingface.co/login")
    print("   b. Log in with your account or create one")
    print("   c. Ensure you're logged in before proceeding")
    
    print("\n2. ACCEPT THE SPEAKER DIARIZATION LICENSE:")
    print("   a. Go directly to: https://huggingface.co/pyannote/speaker-diarization")
    print("   b. Look for a banner or a button that says 'Access repository' or 'Accept license'")
    print("   c. Scroll down if necessary and click 'Accept'")
    print("   d. Wait until you see a success message")
    
    print("\n3. ACCEPT THE SEGMENTATION MODEL LICENSE:")
    print("   a. Go directly to: https://huggingface.co/pyannote/segmentation")
    print("   b. Look for a banner or a button that says 'Access repository' or 'Accept license'")
    print("   c. Scroll down if necessary and click 'Accept'")
    print("   d. Wait until you see a success message")
    
    print("\n4. LOGIN WITH THE CLI TOOL:")
    print("   a. Run 'huggingface-cli login' in your terminal")
    print("   b. A browser will open - complete the login process")
    print("   c. You should see 'Login successful' in the terminal")
    
    print("\n5. VERIFY PERMISSIONS:")
    print("   a. Go to https://huggingface.co/settings/tokens")
    print("   b. Generate a new token if you don't see one")
    print("   c. Make sure you've actually clicked 'Accept' on BOTH model pages")
    
    print("\n6. RESTART THE APPLICATION")
    print("   Run the transcription command again after completing all steps")
    
    print("\nNOTE: Just visiting the pages is not enough - you MUST click 'Accept'!")
    print("      Sometimes the accept button may be hidden or appear at the bottom of the page.\n")

def perform_diarization(audio_path: Union[str, Path], pipeline) -> List[Tuple[float, float, str]]:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_path: Path to the audio file
        pipeline: Loaded diarization pipeline
        
    Returns:
        List of (start_time, end_time, speaker_id) tuples
    """
    # Check if pipeline is None
    if pipeline is None:
        logger.error("Cannot perform diarization: Pipeline is None")
        return []
        
    try:
        logger.info(f"Performing speaker diarization on {audio_path}...")
        
        # Ensure the path is a string
        audio_path_str = str(audio_path)
        
        # Run diarization with minimal parameters to avoid errors
        # Different pyannote versions support different parameters
        try:
            # First try with commonly supported parameters
            diarization = pipeline(
                audio_path_str,
                min_speakers=1,
                max_speakers=3
            )
        except TypeError as param_error:
            logger.warning(f"Parameter error in diarization: {param_error}")
            logger.warning("Falling back to default parameters")
            # Fall back to no parameters if the above fails
            diarization = pipeline(audio_path_str)
        
        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            segments.append((start, end, speaker))
        
        # Optionally write RTTM file for debugging
        temp_rttm = os.path.join(tempfile.gettempdir(), f"diarization_{os.path.basename(audio_path_str)}.rttm")
        with open(temp_rttm, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
        logger.debug(f"RTTM file written to {temp_rttm}")
        
        logger.info(f"Identified {len(segments)} speaker segments")
        if segments:
            logger.info(f"Speakers detected: {set(speaker for _, _, speaker in segments)}")
        
        return segments
    
    except Exception as e:
        logger.error(f"Error performing speaker diarization: {e}")
        return []

def segment_audio(audio_path: Union[str, Path], segments: List[Tuple[float, float, str]]) -> Dict:
    """
    Segment audio by speaker for transcription.
    
    Args:
        audio_path: Path to the audio file
        segments: List of (start_time, end_time, speaker_id) tuples
        
    Returns:
        Dictionary mapping segment indices to (audio_data, sr, start, end, speaker) tuples
    """
    try:
        # Load the full audio file
        audio_data, sr = librosa.load(str(audio_path), sr=None)
        
        # Create segments
        audio_segments = {}
        for i, (start, end, speaker) in enumerate(segments):
            # Convert time to sample indices
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # Ensure valid indices
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if end_sample > start_sample:
                # Extract the segment
                segment_data = audio_data[start_sample:end_sample]
                audio_segments[i] = (segment_data, sr, start, end, speaker)
        
        logger.info(f"Created {len(audio_segments)} audio segments")
        return audio_segments
    
    except Exception as e:
        logger.error(f"Error segmenting audio: {e}")
        return {}

def merge_transcriptions(segment_transcriptions: Dict) -> Dict:
    """
    Merge speaker-specific transcriptions into a single result.
    
    Args:
        segment_transcriptions: Dictionary of {idx: (start, end, speaker, text)}
        
    Returns:
        Dictionary with merged transcription results
    """
    if not segment_transcriptions:
        return {"text": "", "segments": []}
    
    # Sort segments by start time
    sorted_segments = sorted(segment_transcriptions.values(), key=lambda x: x[0])
    
    # Create a merged document
    merged_text = ""
    merged_segments = []
    
    # Create a mapping of speaker IDs to simple numbered labels
    speaker_ids = sorted(set(speaker for _, _, speaker, _ in sorted_segments))
    speaker_mapping = {speaker_id: f"Speaker {i+1}" for i, speaker_id in enumerate(speaker_ids)}
    
    for start, end, speaker, text in sorted_segments:
        # Format speaker name using the mapping (e.g. "SPEAKER_00" becomes "Speaker 1")
        speaker_label = speaker_mapping.get(speaker, f"Speaker {speaker.split('_')[-1] if '_' in speaker else speaker}")
        
        # Add to the full text
        if merged_text:
            merged_text += " "
        merged_text += f"[{speaker_label}] {text}"
        
        # Add to segments
        merged_segments.append({
            "start": start,
            "end": end,
            "text": text,
            "speaker": speaker_label
        })
    
    return {
        "text": merged_text,
        "segments": merged_segments
    }
