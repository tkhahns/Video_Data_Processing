"""
Main module for speech-to-text transcription.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import tqdm

# Handle imports differently when run as script vs. as module
if __name__ == "__main__":
    # Add the parent directory to sys.path for direct script execution
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, parent_dir)
    
    from src.speech_to_text import transcription, utils
    from src.speech_to_text import DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_AUDIO_DIR
    from src.speech_to_text import DEFAULT_MODEL, DEFAULT_LANGUAGE, DEFAULT_SEGMENT_SIZE, SUPPORTED_MODELS
else:
    # Use relative imports when imported as a module
    from . import transcription, utils
    from . import DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_AUDIO_DIR
    from . import DEFAULT_MODEL, DEFAULT_LANGUAGE, DEFAULT_SEGMENT_SIZE, SUPPORTED_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def select_files(directory: Path, file_type: str = "audio"):
    """
    Interactive file selection function.
    
    Args:
        directory: Directory containing files to select from
        file_type: Type of files to look for ("audio")
        
    Returns:
        List of selected file paths
    """
    from src.speech_to_text import SUPPORTED_AUDIO_FORMATS
    
    # Check if separated speech directory exists, prioritize that
    separated_speech_dir = Path("./output/separated_speech")
    if separated_speech_dir.exists() and separated_speech_dir.is_dir():
        has_files = any(f.suffix.lower() in SUPPORTED_AUDIO_FORMATS for f in separated_speech_dir.glob("*"))
        if has_files:
            logger.info(f"Found separated speech files in {separated_speech_dir}")
            directory = separated_speech_dir
    
    # Find all audio files in the directory
    all_files = []
    if directory.exists():
        for file_path in directory.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                all_files.append(file_path)
    
    if not all_files:
        print(f"\nNo {file_type} files found in {directory}")
        return []
    
    # Sort files alphabetically
    all_files = sorted(all_files)
    
    # Print the list of available files
    print(f"\nAvailable {file_type} files:")
    for i, file_path in enumerate(all_files, 1):
        is_separated = utils.is_separated_speech_file(file_path)
        source_indicator = " (separated speech)" if is_separated else ""
        print(f"{i}. {file_path}{source_indicator}")
    
    selected_files = []
    
    # Prompt for selection
    while True:
        print("\nOptions:")
        print("- Enter numbers (e.g., '1,3,5') to select specific videos")
        print("- Enter 'all' to process all videos")
        print("- Enter 'q' to quit")
        
        choice = input("\nYour selection: ").strip()
        
        if choice.lower() == 'q':
            print("Quitting...")
            return []
        
        if choice.lower() == 'all':
            print(f"Selected all {len(all_files)} files")
            return all_files
        
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
                is_separated = utils.is_separated_speech_file(file_path)
                source_indicator = " (separated speech)" if is_separated else ""
                print(f"{i}. {file_path}{source_indicator}")
            
            confirmation = input("\nConfirm selection? (y/n): ").strip().lower()
            if confirmation == 'y':
                return selected_files
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Transcribe speech audio files to text")
    parser.add_argument(
        "input",
        nargs="*",
        help="Input audio file(s) or directory. If not provided, separated speech files will be used."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save transcription files"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help="Speech-to-text model to use"
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing downloaded models"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process audio files in subdirectories recursively"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive audio selection mode"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Language code for transcription (e.g., 'en', 'fr', 'es')"
    )
    parser.add_argument(
        "--use-separated-speech",
        action="store_true", 
        help="Use output from speech separation module as input"
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    utils.ensure_dir_exists(args.output_dir)
    
    # Check for dependencies
    if not utils.check_dependencies():
        logger.error("Missing dependencies. Please install required packages.")
        return 1
    
    # If use_separated_speech flag is set, override the default audio directory
    if args.use_separated_speech or not args.input:
        separated_speech_dir = Path("./output/separated_speech")
        if separated_speech_dir.exists():
            logger.info(f"Looking for separated speech files in {separated_speech_dir}")
            if not args.input:
                args.input = [str(separated_speech_dir)]
    
    # Find all available audio files
    all_audio_files = utils.find_audio_files(
        args.input, 
        args.recursive,
        DEFAULT_AUDIO_DIR
    )
    
    # If no audio files found or interactive mode requested, prompt user to select
    if not all_audio_files or args.interactive:
        logger.info("No audio files found or interactive mode requested.")
        
        # Use our own file selection interface
        all_audio_files = select_files(DEFAULT_AUDIO_DIR, "audio")
        
        if not all_audio_files:
            logger.info("Please provide audio file(s) as arguments or use --recursive to search for files.")
            return 0
    
    logger.info(f"Found {len(all_audio_files)} audio file(s) to process")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    utils.ensure_dir_exists(output_dir)
    
    # Process each audio file
    for audio_file in tqdm.tqdm(all_audio_files, desc="Transcribing audio files"):
        logger.info(f"Processing {audio_file}")
        
        try:
            # Generate output path for this audio file
            output_path = utils.get_output_path(audio_file, output_dir)
            
            # Transcribe the audio file
            result = transcription.transcribe_audio(
                audio_file,
                output_path,
                args.model,
                Path(args.models_dir),
                args.language
            )
            
            if "error" in result:
                logger.error(f"Error transcribing {audio_file}: {result['error']}")
            else:
                logger.info(f"Successfully transcribed {audio_file}")
                
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
    
    logger.info("Transcription process completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
