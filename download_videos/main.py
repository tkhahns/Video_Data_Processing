"""
Main module for downloading videos from SharePoint.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

from .browser import setup_browser, authenticate_with_selenium
from .sharepoint import find_all_files
from .interface import display_file_list, prompt_for_file_selection
from .download import download_selected_files
from .utils import ensure_dir_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Videos directory
VIDEOS_DIR = Path("./data/videos")

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
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - detailed logging activated")
    
    # Ensure output directory exists
    ensure_dir_exists(args.output_dir)
    
    # Setup browser with download directory configured
    browser = setup_browser(headless=args.headless, output_dir=args.output_dir)
    if not browser:
        logger.error("Failed to set up browser")
        return 1
    
    try:
        # Authenticate to SharePoint
        if not authenticate_with_selenium(browser, args.url):
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
        if args.list_only:
            logger.info("List-only mode, exiting without downloading")
            return 0
        
        # Prompt for file selection
        selected_files = prompt_for_file_selection(all_files)
        
        if not selected_files:
            logger.info("No files selected for download")
            return 0
            
        logger.info(f"Selected {len(selected_files)} files for download")
        
        # Download selected files
        successful, failed = download_selected_files(browser, selected_files, args.output_dir)
        logger.info(f"Download complete: {successful} files downloaded successfully, {failed} failed")
        
        return 0 if failed == 0 else 1
        
    finally:
        # Always close the browser
        if browser:
            browser.quit()
