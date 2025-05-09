"""
SharePoint file discovery functionality.
"""
import logging
import time
import urllib.parse
import sys
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

# Try importing from utils package
try:
    from utils import colored_logging, init_logging
except ImportError:
    # Fall back to adding the parent directory to sys.path
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from utils import colored_logging, init_logging

# Get logger with colored output
logger = init_logging.get_logger(__name__)

# Define common video file extensions
VIDEO_EXTENSIONS = ['.MP4', '.MOV', '.AVI', '.WMV', '.MKV', '.M4V', '.WEBM', '.FLV', '.3GP']

def find_all_files(browser):
    """Find all files in the current SharePoint page using classic view selectors."""
    try:
        # Wait for the page to fully load
        logger.info("Waiting for SharePoint page to fully load...")
        time.sleep(3)  # Give the page some time to initialize
        
        try:
            # Try to wait for a more specific element to ensure page is loaded
            WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.ms-List-cell, a.ms-listlink, div.ms-vb"))
            )
            logger.info("SharePoint page elements detected")
        except TimeoutException:
            logger.warning("Timed out waiting for specific elements - continuing with basic detection")
        
        # Try classic SharePoint view selectors first (based on the HTML sample)
        logger.info("Looking for files in classic SharePoint view...")
        
        # Method 1: Look for ms-listlink class (direct links to files)
        file_items = browser.find_elements(By.CSS_SELECTOR, "a.ms-listlink")
        if file_items:
            logger.info(f"Found {len(file_items)} files with ms-listlink selector")
        else:
            # Method 2: Try looking for the container divs
            logger.info("Looking for file containers with ms-vb class...")
            container_items = browser.find_elements(By.CSS_SELECTOR, "div.ms-vb")
            
            if container_items:
                logger.info(f"Found {len(container_items)} containers, checking for file links inside...")
                file_items = []
                for container in container_items:
                    try:
                        link = container.find_element(By.TAG_NAME, "a")
                        if link:
                            file_items.append(link)
                    except:
                        pass
                logger.info(f"Extracted {len(file_items)} file links from containers")
            else:
                # Method 3: Try modern SharePoint view selectors
                logger.info("Trying modern SharePoint view selectors...")
                file_items = browser.find_elements(By.CSS_SELECTOR, "div.ms-List-cell a[role='link']")
                
                if not file_items:
                    # Method 4: Try any links with video extensions
                    logger.info("Trying to find any links with video extensions...")
                    all_links = browser.find_elements(By.TAG_NAME, "a")
                    file_items = [link for link in all_links if 
                                 any(ext in (link.get_attribute("href") or "").upper() for ext in VIDEO_EXTENSIONS) or
                                 any(ext in (link.get_attribute("aria-label") or "").upper() for ext in 
                                    [ext.strip('.') for ext in VIDEO_EXTENSIONS])]
                    logger.info(f"Found {len(file_items)} links that appear to be video files")
        
        file_list = []
        
        # Process each file link found
        for index, item in enumerate(file_items):
            try:
                # Get the href to check if it's a file
                href = item.get_attribute("href") or ""
                
                # Skip if it doesn't look like a file link
                if not href or not "/" in href:
                    continue
                
                # Get file name from aria-label first (it's more complete with extension)
                aria_label = item.get_attribute("aria-label") or ""
                if aria_label and "File" in aria_label:
                    # Format typically: "FILENAME, mp4 File"
                    file_name = aria_label.split(",")[0].strip()
                    # Check if we need to add the extension
                    if not any(file_name.upper().endswith(ext) for ext in VIDEO_EXTENSIONS):
                        # Try to extract extension from aria-label
                        for ext in VIDEO_EXTENSIONS:
                            ext_name = ext.strip('.').lower()
                            if ext_name in aria_label.lower():
                                file_name += ext.lower()
                                break
                else:
                    # Fall back to the text content
                    file_name = item.text.strip()
                    
                    # If the text doesn't include extension, try to add it from href
                    if not any(file_name.upper().endswith(ext) for ext in VIDEO_EXTENSIONS):
                        # Try to extract from the href
                        for ext in VIDEO_EXTENSIONS:
                            if ext in href.upper():
                                file_name += ext.lower()
                                break
                
                # Last resort: extract filename from the href
                if not file_name:
                    path_parts = href.split("/")
                    file_name = path_parts[-1]
                    # URL decode if necessary
                    file_name = urllib.parse.unquote(file_name)
                
                # Skip if we still don't have a valid file name
                if not file_name or file_name.strip() in ["Name", "Modified", "Size", "Type"]:
                    continue
                
                logger.info(f"Found file: {file_name}")
                
                # Add to the list of files
                file_list.append({
                    'index': len(file_list) + 1,
                    'name': file_name,
                    'size': "Unknown size",  # Size may not be readily available in classic view
                    'element': item,
                    'is_video': any(file_name.lower().endswith(ext.lower()) for ext in VIDEO_EXTENSIONS)
                })
            except StaleElementReferenceException:
                logger.warning(f"Element {index} is stale, skipping")
                continue
            except Exception as e:
                logger.warning(f"Error processing item {index}: {e}")
                continue
        
        if file_list:
            logger.info(f"Successfully identified {len(file_list)} files")
            return file_list
        else:
            logger.error("No files could be extracted from the page")
            print("\n❌ TROUBLESHOOTING:")
            print("1. Try running with a URL directly to the folder with videos")
            print("2. Check sharepoint_page.png to see if the files are visible in the browser")
            print("3. The SharePoint page might use a different structure than expected")
            return []
            
    except Exception as e:
        logger.error(f"Error finding files: {e}")
        print("\n❌ ERROR: Could not process the SharePoint page")
        print(f"Detailed error: {str(e)}")
        return []
