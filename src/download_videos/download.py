"""
File download functionality.
"""
import os
import time
import zipfile
import glob
import shutil
from pathlib import Path
from utils import init_logging
import threading
from datetime import datetime
import requests
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException, NoSuchElementException, StaleElementReferenceException

# Get logger with colored output
logger = init_logging.get_logger(__name__)

def monitor_downloads(download_dir, output_dir, timeout=600):
    """
    Monitor the downloads directory for new files and process them.
    
    Args:
        download_dir: Directory to monitor for downloads
        output_dir: Directory to move processed files to
        timeout: Maximum time to wait in seconds (default: 10 minutes)
    
    Returns:
        Tuple of (successful downloads, failed downloads)
    """
    logger.info(f"Starting to monitor downloads in {download_dir}")
    logger.info(f"Downloaded files will be processed and moved to {output_dir}")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Get initial list of files
    initial_files = set(os.listdir(download_dir)) if os.path.exists(download_dir) else set()
    logger.info(f"Initial files in download directory: {len(initial_files)}")
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    print("\n==== Manual Download Instructions ====")
    print("1. Select and download files from the browser")
    print("2. For a single file: right-click and select 'Download'")
    print("3. For multiple files: select multiple files, then download as ZIP")
    print("4. This tool will automatically process downloads and move them to the output directory")
    print("5. Press Ctrl+C to stop monitoring and finish")
    print("=====================================")
    
    try:
        # Start monitoring in a loop
        while time.time() - start_time < timeout:
            # Check current files in download directory
            if not os.path.exists(download_dir):
                time.sleep(2)
                continue
                
            current_files = set(os.listdir(download_dir))
            
            # Find new files
            new_files = current_files - initial_files
            
            # Process each new file
            for filename in new_files:
                file_path = os.path.join(download_dir, filename)
                
                # Skip temporary/partial download files
                if filename.endswith('.crdownload') or filename.endswith('.part') or filename.endswith('.tmp'):
                    continue
                    
                # Wait a moment to ensure the file is completely downloaded
                time.sleep(2)
                
                try:
                    # Check if file is a ZIP archive
                    if filename.lower().endswith('.zip'):
                        logger.info(f"Processing ZIP archive: {filename}")
                        zip_success = process_zip_file(file_path, output_dir)
                        if zip_success:
                            successful += zip_success
                        else:
                            failed += 1
                    else:
                        # Check if file is a video file
                        if is_video_file(filename):
                            # Move the file to the output directory
                            dest_path = os.path.join(output_dir, filename)
                            shutil.move(file_path, dest_path)
                            logger.info(f"Moved {filename} to {output_dir}")
                            successful += 1
                            print(f"✅ Downloaded and processed: {filename}")
                        else:
                            logger.info(f"Skipping non-video file: {filename}")
                    
                    # Add file to initial files so we don't process it again
                    initial_files.add(filename)
                    
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    failed += 1
            
            # Sleep before checking again
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nDownload monitoring stopped by user.")
    
    # Final check for any remaining downloads
    try:
        if os.path.exists(download_dir):
            final_files = set(os.listdir(download_dir))
            new_files = final_files - initial_files
            
            for filename in new_files:
                file_path = os.path.join(download_dir, filename)
                
                # Skip temporary/partial download files
                if filename.endswith('.crdownload') or filename.endswith('.part') or filename.endswith('.tmp'):
                    continue
                
                try:
                    # Process any remaining files
                    if filename.lower().endswith('.zip'):
                        logger.info(f"Processing ZIP archive: {filename}")
                        zip_success = process_zip_file(file_path, output_dir)
                        if zip_success:
                            successful += zip_success
                        else:
                            failed += 1
                    elif is_video_file(filename):
                        dest_path = os.path.join(output_dir, filename)
                        shutil.move(file_path, dest_path)
                        logger.info(f"Moved {filename} to {output_dir}")
                        successful += 1
                        print(f"✅ Downloaded and processed: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    failed += 1
    except Exception as e:
        logger.error(f"Error during final check: {e}")
    
    return successful, failed

def is_video_file(filename):
    """Check if a file is a video file based on its extension."""
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.mvi', '.m4v', '.3gp']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def process_zip_file(zip_path, output_dir):
    """
    Extract video files from a ZIP archive and move them to the output directory.
    
    Args:
        zip_path: Path to the ZIP file
        output_dir: Directory to extract video files to
        
    Returns:
        Number of video files extracted
    """
    try:
        logger.info(f"Opening ZIP file: {zip_path}")
        extracted_videos = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only video files
            for file_info in zip_ref.infolist():
                if is_video_file(file_info.filename):
                    # Get just the filename without directory structure
                    filename = os.path.basename(file_info.filename)
                    
                    # Skip if empty filename (directory)
                    if not filename:
                        continue
                    
                    # Extract the file to the output directory
                    zip_ref.extract(file_info, output_dir)
                    
                    # If file was extracted within a subdirectory, move it to the output dir root
                    extracted_path = os.path.join(output_dir, file_info.filename)
                    target_path = os.path.join(output_dir, filename)
                    
                    if extracted_path != target_path:
                        shutil.move(extracted_path, target_path)
                        
                        # Clean up empty directories
                        dir_path = os.path.dirname(extracted_path)
                        try:
                            if os.path.exists(dir_path) and not os.listdir(dir_path):
                                os.rmdir(dir_path)
                        except:
                            pass
                    
                    extracted_videos += 1
                    logger.info(f"Extracted video: {filename}")
                    print(f"✅ Extracted from ZIP: {filename}")
        
        # Delete the ZIP file after extraction
        try:
            os.remove(zip_path)
            logger.info(f"Deleted ZIP file after extraction: {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to delete ZIP file {zip_path}: {e}")
        
        return extracted_videos
    except Exception as e:
        logger.error(f"Error processing ZIP file {zip_path}: {e}")
        return 0
