"""
Main module for downloading videos from SharePoint.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Handle imports differently when run as script vs. as module
if __name__ == "__main__":
    # Add the parent directory to sys.path for direct script execution
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, parent_dir)
    
    from src.download_videos.browser import setup_browser, authenticate_with_selenium
    from src.download_videos.sharepoint import find_all_files
    from src.download_videos.interface import display_file_list, prompt_for_file_selection
    from src.download_videos.download import download_selected_files
    from src.download_videos.utils import ensure_dir_exists
    
    # Import from utils package
    from utils import colored_logging, init_logging
else:
    # Use relative imports when imported as a module
    from .browser import setup_browser, authenticate_with_selenium
    from .sharepoint import find_all_files
    from .interface import display_file_list, prompt_for_file_selection
    from .download import download_selected_files
    from .utils import ensure_dir_exists
    
    # Try using different approaches for importing the logging modules
    try:
        # First try absolute imports
        from utils import colored_logging, init_logging
    except ImportError:
        # Fall back to relative imports
        try:
            from ...utils import colored_logging, init_logging
        except ImportError:
            # Last resort: add parent directory to sys.path
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
            from utils import colored_logging, init_logging

# Get logger with colored output
logger = init_logging.get_logger(__name__)

# Videos directory
VIDEOS_DIR = Path("./data/videos")

def run_download(url, output_dir=str(VIDEOS_DIR), list_only=False, headless=False, debug=False):
    """Core download function that can be called from various entry points."""
    # Set debug logging if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - detailed logging activated")
    
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # Setup browser with download directory configured
    browser = setup_browser(headless=headless, output_dir=output_dir)
    if not browser:
        logger.error("Failed to set up browser")
        return 1
    
    try:
        # Authenticate to SharePoint
        if not authenticate_with_selenium(browser, url):
            logger.error("Authentication failed")
            return 1
        
        # Find all files
        all_files = find_all_files(browser)
        if not all_files:
            logger.error("No files found")
            return 1
        
        logger.info(f"Found {len(all_files)} files")
        
        # Display file list
        display_file_list(all_files)
        
        # If list-only mode, exit here
        if list_only:
            logger.info("List-only mode, exiting without downloading")
            return 0
        
        # Prompt for file selection
        selected_files = prompt_for_file_selection(all_files)
        
        if not selected_files:
            logger.info("No files selected for download")
            return 0
            
        logger.info(f"Selected {len(selected_files)} files for download")
        
        # Download selected files
        successful, failed = download_selected_files(browser, selected_files, output_dir)
        logger.info(f"Download complete: {successful} files downloaded successfully, {failed} failed")
        
        return 0 if failed == 0 else 1
        
    finally:
        # Always close the browser
        if browser:
            browser.quit()

def main():
    """Main function to download files from a SharePoint folder using browser authentication."""
    parser = argparse.ArgumentParser(description="Download files from a SharePoint folder using browser authentication")
    parser.add_argument(
        "--url", 
        required=True,
        help="SharePoint folder URL containing files"
    )
    parser.add_argument(
        "--output-dir", 
        default=str(VIDEOS_DIR),
        help="Directory to save the downloaded files"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List files without downloading"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (not recommended for downloads)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    args = parser.parse_args()
    
    return run_download(
        args.url, 
        args.output_dir, 
        args.list_only, 
        args.headless, 
        args.debug
    )

if __name__ == "__main__":
    sys.exit(main())
