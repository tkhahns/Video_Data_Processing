"""
Main module for speech separation.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import tqdm

# Handle imports differently when run as script vs. as module
if __name__ == "__main__" or os.path.basename(sys.argv[0]) == "__main__.py":
    # Add the parent directory to sys.path for direct script execution
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, parent_dir)
    
    from src.separate_speech import extraction, interface, processor, utils
    from src.separate_speech import DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_VIDEOS_DIR, DEFAULT_MODEL, DEFAULT_CHUNK_SIZE
    
    # Import from utils package
    try:
        from utils import colored_logging, init_logging
        logger = init_logging.get_logger(__name__)
    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
else:
    # Use relative imports when imported as a module
    from . import extraction, interface, processor, utils
    from . import DEFAULT_MODELS_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_VIDEOS_DIR, DEFAULT_MODEL, DEFAULT_CHUNK_SIZE
    
    # Try using different approaches for importing the logging modules
    try:
        # First try absolute imports
        from utils import colored_logging, init_logging
        logger = init_logging.get_logger(__name__)
    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract and separate speech from video files")
    parser.add_argument(
        "input",
        nargs="*",
        help="Input video file(s) or directory. If not provided, interactive selection will be used."
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing input video files to process"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save separated speech files"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["sepformer", "conv-tasnet", "dual-path-rnn"],
        help="Speech separation model to use"
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing downloaded models"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process video files in subdirectories recursively"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive video selection mode"
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=DEFAULT_CHUNK_SIZE,
        help="Size of audio chunks to process in seconds"
    )
    parser.add_argument(
        "--file-type",
        type=str,
        choices=["wav", "mp3", "both", "1", "2", "3"],
        default="mp3",
        help="Output file format: wav (1), mp3 (2), or both (3)"
    )
    parser.add_argument(
        "--skip-no-speech",
        action="store_true",
        help="Skip files where no significant human speech is detected"
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=1.0,
        help="Minimum seconds of speech required to process a file (with --skip-no-speech)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files without manual selection"
    )
    parser.add_argument(
        "--files-from",
        help="Read file paths from a file, one per line"
    )
    
    args = parser.parse_args()
    
    # Process file type argument
    file_type = args.file_type
    if file_type == "1":
        file_type = "wav"
    elif file_type == "2":
        file_type = "mp3"
    elif file_type == "3":
        file_type = "both"
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    utils.ensure_dir_exists(args.output_dir)
    
    # Check for dependencies
    try:
        import speechbrain
        import moviepy
        import tqdm
        # Try to import pydub for MP3 conversion
        try:
            from pydub import AudioSegment
        except ImportError:
            # Check if ffmpeg is available as fallback
            try:
                utils.check_ffmpeg_dependencies()
            except Exception:
                logger.warning("Neither pydub nor ffmpeg found. MP3 conversion may not work.")
                logger.warning("Install with: pip install pydub")
                logger.warning("Or install ffmpeg: apt install ffmpeg / brew install ffmpeg")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install speechbrain moviepy torchaudio tqdm pydub")
        return 1
    
    # Handle input_dir if provided - find all video files in that directory
    input_args = args.input
    if args.input_dir:
        logger.info(f"Processing videos from input directory: {args.input_dir}")
        # Add input_dir to the input arguments if provided
        if os.path.isdir(args.input_dir):
            input_args = [args.input_dir]
        else:
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1
    
    # Handle files from list if provided
    if args.files_from and os.path.isfile(args.files_from):
        try:
            with open(args.files_from, 'r') as f:
                file_paths = [line.strip() for line in f if line.strip()]
                # Filter to ensure files exist
                existing_files = [path for path in file_paths if os.path.exists(path)]
                if existing_files:
                    logger.info(f"Found {len(existing_files)} video files from provided list")
                    input_args = existing_files
                else:
                    logger.warning(f"No valid files found in {args.files_from}")
        except Exception as e:
            logger.error(f"Error reading file list: {e}")
    
    # Find all available video files
    all_video_files = extraction.find_video_files(
        input_args, 
        args.recursive,
        DEFAULT_VIDEOS_DIR
    )
    
    # Collect video files - either from arguments or interactive selection
    video_files = []
    file_type_from_interactive = None
    
    # Check if batch mode is enabled
    batch_mode = args.batch if hasattr(args, 'batch') else False
    
    # Use interactive mode if no input args or --interactive flag
    # But skip interactive selection if batch mode is enabled
    if (not args.input or args.interactive) and not batch_mode:
        # Interactive video selection
        try:
            # Import the enhanced interface module if it exists
            from . import interface_extended
            video_files, file_type_from_interactive = interface_extended.select_videos_interactively(all_video_files, batch_mode)
        except ImportError:
            # Fall back to standard interface
            video_files, file_type_from_interactive = interface.select_videos_interactively(all_video_files)
    else:
        # Use provided input arguments or all files in batch mode
        video_files = all_video_files
        
    # If batch mode is enabled, log this information
    if batch_mode:
        logger.info(f"Batch mode enabled: Processing all {len(video_files)} files without manual selection")
    
    if not video_files:
        logger.error("No video files selected for processing")
        return 1
    
    # Process file type argument - interactive selection overrides command line if provided
    file_type = file_type_from_interactive or file_type
    
    # Process each video file
    successful = 0
    total_files = len(video_files)
    no_speech_files = 0
    processed_files = []
    
    # Show overall progress
    with tqdm.tqdm(total=total_files, desc="Overall progress", unit="file") as overall_pbar:
        for i, video_path in enumerate(video_files):
            overall_pbar.set_description(f"File {i+1}/{total_files}")
            
            # Clear memory before processing each file
            utils.clean_memory()
            
            # Process the file and check if speech was detected
            success, has_speech = processor.process_file(
                video_path, args.output_dir, args.model, args.models_dir, 
                args.chunk_size, file_type, args.skip_no_speech, args.min_speech_seconds
            )
            
            if success:
                if has_speech:
                    successful += 1
                    processed_files.append(os.path.basename(video_path))
                else:
                    no_speech_files += 1
                overall_pbar.set_postfix(
                    success_rate=f"{successful}/{i+1}", 
                    no_speech=f"{no_speech_files}"
                )
            else:
                logger.error(f"Failed to process {os.path.basename(video_path)}")
            
            overall_pbar.update(1)
    
    print(f"\n✅ Processed {successful + no_speech_files}/{total_files} videos successfully")
    print(f"✅ {successful} videos had speech and were processed")
    
    if no_speech_files > 0:
        print(f"ℹ️ {no_speech_files} videos had no significant speech and were skipped")
    
    # Display processed files if there are not too many
    if processed_files and len(processed_files) < 10:
        print("\nProcessed files with speech:")
        for file in processed_files:
            print(f"- {file}")
            
    print(f"🎵 Audio files saved to: {args.output_dir}")
    
    # Update message based on file type
    if file_type == "wav":
        print("Files were saved in WAV format.")
    elif file_type == "mp3":
        print("Files were saved in MP3 format.")
    else:
        print("Files were saved in both WAV and MP3 formats.")
    
    return 0 if (successful + no_speech_files) == total_files else 1

if __name__ == "__main__":
    sys.exit(main())
