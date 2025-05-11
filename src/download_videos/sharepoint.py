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
# Common file patterns for videos (like MVI_xxxx)
VIDEO_PATTERNS = ['MVI_', 'IMG_', 'VIDEO_', 'MOV', 'VID_', 'CLIP_']

def take_screenshot(browser, filename="sharepoint_page.png"):
    """Take a screenshot to help with debugging"""
    try:
        screenshot_path = os.path.abspath(filename)
        browser.save_screenshot(screenshot_path)
        logger.info(f"Screenshot saved to {screenshot_path}")
        print(f"Screenshot saved to {screenshot_path} for troubleshooting")
    except Exception as e:
        logger.error(f"Failed to take screenshot: {e}")

def find_all_files(browser):
    """Find all files in the current SharePoint page using classic view selectors."""
    try:
        # Wait for the page to fully load
        logger.info("Waiting for SharePoint page to fully load...")
        time.sleep(5)  # Give the page more time to initialize
        
        # # Take a screenshot for debugging
        # take_screenshot(browser)
        
        # Try multiple methods to find files
        file_items = []
        
        # Method 1: Modern SharePoint view with heroTextWithHeroCommandsWrapped class
        logger.info("Looking for files in modern SharePoint hero view...")
        hero_elements = browser.find_elements(By.CLASS_NAME, "heroTextWithHeroCommandsWrapped_cbea99d3")
        
        if hero_elements:
            logger.info(f"Found {len(hero_elements)} files using hero text elements")
            file_items = hero_elements
        
        # If no files found, try other methods from the existing implementation
        if not file_items:
            # Try classic SharePoint view selectors first (based on the HTML sample)
            logger.info("Looking for files in classic SharePoint view...")
            
            # Method 2: Look for ms-listlink class (direct links to files)
            file_items = browser.find_elements(By.CSS_SELECTOR, "a.ms-listlink")
            if file_items:
                logger.info(f"Found {len(file_items)} files with ms-listlink selector")
            
        # Try more methods if needed
        if not file_items:
            logger.info("Trying modern SharePoint view selectors...")
            file_items = browser.find_elements(By.CSS_SELECTOR, "div.ms-List-cell a[role='link']")
            if file_items:
                logger.info(f"Found {len(file_items)} files with modern SharePoint selector")
                
        # Add other fallback methods
        if not file_items:
            logger.info("Trying to find any links with video extensions or patterns...")
            all_links = browser.find_elements(By.TAG_NAME, "a")
            file_items = [link for link in all_links if 
                        any(ext.lower() in (link.get_attribute("href") or "").lower() for ext in [e.lower() for e in VIDEO_EXTENSIONS]) or
                        any(pattern in (link.get_attribute("href") or "").upper() for pattern in VIDEO_PATTERNS)]
            logger.info(f"Found {len(file_items)} links that appear to be video files by extension/pattern")
        
        file_list = []
        file_index = 1
        
        # Process each file item found
        for index, item in enumerate(file_items):
            try:
                # Extract different attributes depending on the item type
                file_name = None
                file_size = "Unknown size"
                is_link = item.tag_name.lower() == "a"
                
                # Case 1: For hero text elements (specific to the provided SharePoint layout)
                if item.get_attribute("class") and "heroTextWithHeroCommandsWrapped" in item.get_attribute("class"):
                    file_name = item.text.strip()
                    
                    # Try to find file size in nearby cells
                    try:
                        # Navigate up to the row and find size cell
                        row_element = item
                        for _ in range(5):  # Try navigating up a few levels to find the row
                            if row_element.get_attribute("role") == "row":
                                break
                            row_element = row_element.find_element(By.XPATH, "./..")
                        
                        # Find size element
                        if row_element.get_attribute("role") == "row":
                            size_cells = row_element.find_elements(By.CSS_SELECTOR, "[data-automationid='field-FileSizeDisplay']")
                            if size_cells:
                                file_size = size_cells[0].text.strip()
                    except Exception as e:
                        logger.debug(f"Could not extract size info: {e}")
                
                # Case 2: For link elements (from previous methods)
                elif is_link:
                    href = item.get_attribute("href") or ""
                    aria_label = item.get_attribute("aria-label") or ""
                    text_content = item.text.strip()
                    
                    # Use multiple methods to extract filename
                    if aria_label and "File" in aria_label:
                        parts = aria_label.split(",")
                        if len(parts) > 1:
                            file_name = parts[0].strip()
                        else:
                            parts = aria_label.split("File")
                            if len(parts) > 1:
                                file_name = parts[0].strip()
                    
                    if not file_name and text_content:
                        file_name = text_content
                    
                    if not file_name and href:
                        path_parts = href.split("/")
                        url_filename = path_parts[-1] if path_parts else ""
                        if "?" in url_filename:
                            url_filename = url_filename.split("?")[0]
                        file_name = urllib.parse.unquote(url_filename)
                
                # Skip if no valid file name found
                if not file_name or file_name.strip() in ["Name", "Modified", "Size", "Type"]:
                    continue
                
                # Determine if this is a video file
                is_video = any(file_name.upper().endswith(ext.upper()) for ext in VIDEO_EXTENSIONS) or \
                           any(pattern in file_name.upper() for pattern in VIDEO_PATTERNS)
                
                logger.info(f"Found file: {file_name} ({file_size}) (Video: {is_video})")
                
                # Add to the list of files
                file_list.append({
                    'index': file_index,
                    'name': file_name,
                    'size': file_size,
                    'element': item,
                    'is_video': is_video
                })
                file_index += 1
                
            except Exception as e:
                logger.warning(f"Error processing item {index}: {e}")
                continue
        
        # Return results
        if file_list:
            logger.info(f"Successfully identified {len(file_list)} files")
            return file_list
        else:
            logger.error("No files could be extracted from the page")
            print("\n❌ TROUBLESHOOTING:")
            print("1. Check the screenshot at sharepoint_page.png")
            print("2. Make sure you're using a URL that points directly to a folder with files")
            print("3. Try a different browser or disable any SharePoint browser extensions")
            return []
            
    except Exception as e:
        logger.error(f"Error finding files: {e}")
        print("\n❌ ERROR: Could not process the SharePoint page")
        print(f"Detailed error: {str(e)}")
        # if browser:
        #     take_screenshot(browser, "error_screenshot.png")
        return []
