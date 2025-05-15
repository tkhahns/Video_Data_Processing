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

def sanitize_filename(filename):
    """
    Sanitize a filename by replacing spaces with underscores and removing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace spaces with underscores
    sanitized = filename.replace(' ', '_')
    
    # Remove parentheses and brackets as they can cause issues in some contexts
    sanitized = sanitized.replace('(', '').replace(')', '')
    sanitized = sanitized.replace('[', '').replace(']', '')
    
    # Remove other potentially problematic characters
    sanitized = sanitized.replace(',', '').replace(';', '')
    
    # Ensure we don't have double underscores from removing characters
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    
    return sanitized

def monitor_downloads(download_dir, output_dir, timeout=1800, idle_timeout=60):
    """
    Monitor the downloads directory for new files and process them.
    
    Args:
        download_dir: Directory to monitor for downloads
        output_dir: Directory to move processed files to
        timeout: Maximum total time to wait in seconds (default: 30 minutes)
        idle_timeout: Time to wait after last file activity before auto-completing (default: 60 seconds)
    
    Returns:
        Tuple of (successful downloads, failed downloads)
    """
    logger.info(f"Starting to monitor downloads in {download_dir}")
    logger.info(f"Downloaded files will be processed and moved to {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Ensured output directory exists: {output_dir}")
    
    # Get initial list of files
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"Created downloads directory: {download_dir}")
        
    initial_files = set(os.listdir(download_dir)) if os.path.exists(download_dir) else set()
    logger.info(f"Initial files in download directory: {len(initial_files)}")
    
    start_time = time.time()
    last_activity_time = start_time
    successful = 0
    failed = 0
    processed_files = []  # Track processed files for verification
    
    print("\n==== Manual Download Instructions ====")
    print("1. Select and download files from the browser")
    print("2. For a single file: right-click and select 'Download'")
    print("3. For multiple files: select multiple files, then download as ZIP")
    print("4. This tool will automatically process downloads and move them to the output directory")
    print("5. The process will automatically continue after a period of inactivity")
    print("6. You can press Ctrl+C at any time to stop monitoring and continue")
    print("=====================================")
    
    try:
        # Start monitoring in a loop
        while time.time() - start_time < timeout:
            # Check if we've been idle for too long
            current_time = time.time()
            if (current_time - last_activity_time > idle_timeout) and successful > 0:
                logger.info(f"No new download activity for {idle_timeout} seconds and {successful} file(s) processed. Auto-continuing.")
                print(f"\n✅ Download monitoring completed: Processed {successful} file(s)")
                break
                
            # Check current files in download directory
            if not os.path.exists(download_dir):
                time.sleep(2)
                continue
                
            current_files = set(os.listdir(download_dir))
            
            # Find new files
            new_files = current_files - initial_files
            
            # Process each new file
            activity_occurred = False
            for filename in new_files:
                file_path = os.path.join(download_dir, filename)
                
                # Skip temporary/partial download files
                if filename.endswith('.crdownload') or filename.endswith('.part') or filename.endswith('.tmp'):
                    continue
                    
                # Wait a moment to ensure the file is completely downloaded
                time.sleep(2)
                activity_occurred = True
                
                try:
                    # Check if the source file exists before processing
                    if not os.path.exists(file_path):
                        logger.error(f"Source file doesn't exist: {file_path}")
                        failed += 1
                        continue
                        
                    # Check if file is a ZIP archive
                    if filename.lower().endswith('.zip'):
                        logger.info(f"Processing ZIP archive: {filename}")
                        zip_success, extracted_paths = process_zip_file(file_path, output_dir)
                        if zip_success > 0:
                            successful += zip_success
                            processed_files.extend(extracted_paths)  # Add all extracted files to processed list
                        else:
                            failed += 1
                    else:
                        # Check if file is a video file
                        if is_video_file(filename):
                            # Sanitize the filename to remove spaces and problematic characters
                            sanitized_filename = sanitize_filename(filename)
                            if sanitized_filename != filename:
                                logger.info(f"Sanitized filename: '{filename}' -> '{sanitized_filename}'")
                            
                            # Make sure output directory exists
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Move the file to the output directory with sanitized name
                            dest_path = os.path.join(output_dir, sanitized_filename)
                            
                            # First copy then delete to avoid race conditions
                            logger.info(f"Copying {filename} to {output_dir} as {sanitized_filename}")
                            shutil.copy2(file_path, dest_path)
                            
                            # Verify file was copied correctly
                            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                                # Delete the original only after successful copy
                                try:
                                    os.remove(file_path)
                                except Exception as e:
                                    logger.warning(f"Failed to remove original file {file_path}: {e}")
                                
                                # Add to processed files list for verification
                                processed_files.append(dest_path)
                                    
                                logger.info(f"Successfully moved {filename} to {output_dir} as {sanitized_filename}")
                                successful += 1
                                print(f"✅ Downloaded and processed: {sanitized_filename}")
                            else:
                                logger.error(f"Failed to copy {filename} to {output_dir}")
                                failed += 1
                        else:
                            logger.info(f"Skipping non-video file: {filename}")
                    
                    # Add file to initial files so we don't process it again
                    initial_files.add(filename)
                    
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    # Print stack trace for debugging
                    import traceback
                    logger.error(f"Exception details: {traceback.format_exc()}")
                    failed += 1
            
            # Update the last activity time if there was any file processing
            if activity_occurred:
                last_activity_time = time.time()
            
            # Sleep before checking again
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nDownload monitoring stopped by user.")
    
    # Final check for any remaining downloads and verify successful transfers
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
                        zip_success, extracted_paths = process_zip_file(file_path, output_dir)
                        if zip_success > 0:
                            successful += zip_success
                            processed_files.extend(extracted_paths)  # Add all extracted files to processed list
                        else:
                            failed += 1
                    elif is_video_file(filename):
                        dest_path = os.path.join(output_dir, filename)
                        shutil.move(file_path, dest_path)
                        processed_files.append(dest_path)  # Track processed file location
                        logger.info(f"Moved {filename} to {output_dir}")
                        successful += 1
                        print(f"✅ Downloaded and processed: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    failed += 1
                    
        # Verify that files actually exist in the output directory
        valid_files = 0
        for file_path in processed_files:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                valid_files += 1
            else:
                logger.warning(f"Processed file not found or empty: {file_path}")
                
        if valid_files != successful:
            logger.warning(f"Mismatch between successful count ({successful}) and verified files ({valid_files})")
            logger.info(f"Processed files tracked: {len(processed_files)}")
            logger.info(f"Files found in output directory: {os.listdir(output_dir) if os.path.exists(output_dir) else 'directory not found'}")
            successful = valid_files
    except Exception as e:
        logger.error(f"Error during final check: {e}")
    
    if successful > 0:
        print(f"\n✅ Download complete: Successfully processed {successful} video files")
        logger.info(f"Confirmed {successful} video files in output directory: {output_dir}")
    else:
        print("\n⚠️ No video files were processed. Check if downloads completed correctly.")
    
    # Small pause to let user see the results before continuing
    time.sleep(3)
    
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
        Tuple of (number of video files extracted, list of paths)
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Opening ZIP file: {zip_path}")
        extracted_videos = 0
        extracted_paths = []  # Track extracted file paths
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only video files
            for file_info in zip_ref.infolist():
                if is_video_file(file_info.filename):
                    # Get just the filename without directory structure
                    filename = os.path.basename(file_info.filename)
                    
                    # Skip if empty filename (directory)
                    if not filename:
                        continue
                        
                    # Sanitize the filename to remove spaces and problematic characters
                    sanitized_filename = sanitize_filename(filename)
                    if sanitized_filename != filename:
                        logger.info(f"Sanitized extracted filename: '{filename}' -> '{sanitized_filename}'")
                    
                    # Extract with original name first (ZIP extraction doesn't support renaming during extraction)
                    zip_ref.extract(file_info, output_dir)
                    
                    # Original extraction path and target path with sanitized name
                    extracted_path = os.path.join(output_dir, file_info.filename)
                    target_path = os.path.join(output_dir, sanitized_filename)
                    
                    # If names are different or file was extracted within a subdirectory, move it to the output dir root
                    if extracted_path != target_path:
                        # Make sure we don't overwrite existing files with same sanitized name
                        if os.path.exists(target_path):
                            logger.warning(f"File with sanitized name already exists: {target_path}")
                            # Add a timestamp to make the filename unique
                            name, ext = os.path.splitext(sanitized_filename)
                            timestamp = int(time.time())
                            sanitized_filename = f"{name}_{timestamp}{ext}"
                            target_path = os.path.join(output_dir, sanitized_filename)
                            
                        # Move the file to the target path with sanitized name
                        shutil.move(extracted_path, target_path)
                        
                        # Clean up empty directories
                        dir_path = os.path.dirname(extracted_path)
                        try:
                            if os.path.exists(dir_path) and not os.listdir(dir_path):
                                os.rmdir(dir_path)
                        except:
                            pass
                    
                    extracted_paths.append(target_path)
                    extracted_videos += 1
                    logger.info(f"Extracted video: {sanitized_filename}")
                    print(f"✅ Extracted from ZIP: {sanitized_filename}")
        
        # Verify extracted files exist
        valid_files = 0
        for path in extracted_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                valid_files += 1
            else:
                logger.warning(f"Extracted file not found or empty: {path}")
        
        if valid_files != extracted_videos:
            logger.warning(f"Mismatch between extracted count ({extracted_videos}) and verified files ({valid_files})")
            extracted_videos = valid_files
            # Keep only valid paths
            extracted_paths = [path for path in extracted_paths if os.path.exists(path) and os.path.getsize(path) > 0]
        
        # Delete the ZIP file after extraction
        try:
            os.remove(zip_path)
            logger.info(f"Deleted ZIP file after extraction: {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to delete ZIP file {zip_path}: {e}")
        
        return extracted_videos, extracted_paths
    except Exception as e:
        logger.error(f"Error processing ZIP file {zip_path}: {e}")
        return 0, []
