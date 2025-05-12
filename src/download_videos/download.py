"""
File download functionality.
"""
import os
from utils import init_logging
import time
import requests
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException, NoSuchElementException, StaleElementReferenceException

# Get logger with colored output
logger = init_logging.get_logger(__name__)

def download_selected_files(browser, selected_files, output_dir):
    """Download files directly to the machine."""
    try:
        successful_downloads = 0
        failed_downloads = 0
        
        for file in selected_files:
            try:
                file_name = file['name']
                output_path = os.path.join(output_dir, file_name)
                
                # Check if file already exists
                if os.path.exists(output_path):
                    logger.info(f"File already exists: {output_path}")
                    successful_downloads += 1
                    continue
                
                logger.info(f"Downloading {file_name}...")
                
                # Configure Chrome to download to specified directory
                download_prefs = {
                    "download.default_directory": os.path.abspath(output_dir),
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "safebrowsing.enabled": False
                }
                browser.execute_cdp_cmd('Page.setDownloadBehavior', {
                    'behavior': 'allow',
                    'downloadPath': os.path.abspath(output_dir)
                })
                
                # Method 1: Use right-click context menu to download
                try:
                    logger.info("Attempting context menu download...")
                    
                    # Right-click on the element to open context menu
                    webdriver.ActionChains(browser).context_click(file['element']).perform()
                    time.sleep(1.5)  # Give more time for context menu to appear
                    
                    # # Take a screenshot of the context menu for debugging
                    # browser.save_screenshot(os.path.join(output_dir, "context_menu.png"))
                    
                    # First look for the specific downloadCommand element based on user's HTML
                    download_btn = None
                    try:
                        # Look for the button with data-automationid="downloadCommand"
                        download_btn = browser.find_element(By.CSS_SELECTOR, "button[data-automationid='downloadCommand']")
                        logger.info("Found download button with data-automationid='downloadCommand'")
                    except NoSuchElementException:
                        logger.info("Could not find button with data-automationid='downloadCommand', trying alternative selectors")
                    
                    # If first method failed, try other methods
                    if not download_btn:
                        # Look for elements with download icon
                        icons = browser.find_elements(By.CSS_SELECTOR, "i[data-icon-name='download']")
                        if icons:
                            # Find the parent button that contains this icon
                            for icon in icons:
                                try:
                                    # Navigate up through parents to find the button
                                    parent = icon.find_element(By.XPATH, "./..")  # linkContent div
                                    if parent:
                                        parent = parent.find_element(By.XPATH, "./..")  # button
                                        download_btn = parent
                                        logger.info("Found download button through icon parent")
                                        break
                                except:
                                    continue
                    
                    # If still not found, try looking for text
                    if not download_btn:
                        # Look for elements with "Download" text
                        elements_with_text = browser.find_elements(By.XPATH, "//*[contains(text(), 'Download')]")
                        if elements_with_text:
                            for el in elements_with_text:
                                try:
                                    # Try to find a clickable parent
                                    parent = el
                                    for _ in range(3):  # Try a few levels up
                                        try:
                                            if parent.tag_name.lower() in ["button", "a"]:
                                                download_btn = parent
                                                logger.info(f"Found download button with tag: {parent.tag_name}")
                                                break
                                            parent = parent.find_element(By.XPATH, "./..")
                                        except:
                                            break
                                    if download_btn:
                                        break
                                except:
                                    continue
                    
                    # Last resort - try className patterns for SharePoint context menus
                    if not download_btn:
                        menu_items = browser.find_elements(By.CSS_SELECTOR, ".ms-ContextualMenu-item")
                        for item in menu_items:
                            try:
                                text = item.text.lower()
                                if "download" in text:
                                    # Find the button inside this menu item
                                    buttons = item.find_elements(By.TAG_NAME, "button")
                                    if buttons:
                                        download_btn = buttons[0]
                                        logger.info("Found download button in menu item with 'download' text")
                                        break
                            except:
                                continue
                    
                    # Click the download button if found
                    if download_btn:
                        # Add explicit wait before clicking
                        try:
                            logger.info("Clicking download button...")
                            browser.execute_script("arguments[0].scrollIntoView(true);", download_btn)
                            time.sleep(0.5)
                            download_btn.click()
                            logger.info("Successfully clicked download button")
                        except ElementNotInteractableException:
                            # Try JavaScript click if direct click fails
                            logger.info("Using JavaScript to click download button")
                            browser.execute_script("arguments[0].click();", download_btn)
                        except StaleElementReferenceException:
                            logger.warning("Download button became stale, trying again")
                            # Right-click again and retry
                            webdriver.ActionChains(browser).context_click(file['element']).perform()
                            time.sleep(1.5)
                            # Try simpler approach now
                            browser.find_element(By.XPATH, "//*[contains(text(), 'Download')]").click()
                    else:
                        logger.warning("Could not find download button in context menu")
                        raise Exception("Download button not found in context menu")
                    
                    # Wait for download to begin and complete
                    max_wait = 180  # seconds
                    poll_interval = 2  # seconds
                    file_downloading = True
                    start_time = time.time()
                    
                    print(f"\n⬇️ Downloading {file_name}...")
                    
                    # Check for both common download files and completed files
                    while file_downloading and (time.time() - start_time) < max_wait:
                        # Check if file exists
                        if os.path.exists(output_path):
                            # Check if download is complete (not a partial file)
                            if not output_path.endswith('.crdownload') and not os.path.exists(f"{output_path}.crdownload"):
                                file_downloading = False
                                break
                        
                        # Check for common partial download extensions
                        for ext in ['.crdownload', '.part', '.download']:
                            partial_path = f"{output_path}{ext}"
                            if os.path.exists(partial_path):
                                try:
                                    partial_size = os.path.getsize(partial_path)
                                    print(f"\rDownloading... {partial_size/1024/1024:.2f} MB", end="")
                                except:
                                    pass
                        
                        time.sleep(poll_interval)
                    
                    if not file_downloading and os.path.exists(output_path):
                        logger.info(f"File successfully downloaded to {output_path}")
                        successful_downloads += 1
                        continue
                    else:
                        logger.warning("Context menu download timed out or couldn't be tracked")
                        
                except Exception as e:
                    logger.warning(f"Context menu download failed: {e}")
                    
                    # Try to close any open context menus
                    try:
                        webdriver.ActionChains(browser).move_by_offset(10, 10).click().perform()
                    except:
                        pass
                
                # Method 2: Try direct URL method if the context menu approach failed
                # ... existing code ...
                
                # Method 3: Last resort - Manual download instructions
                print(f"\n⚠️ Automatic download methods failed for {file_name}")
                print("Please try one of these methods:")
                print("1. Right-click the file and select 'Download' from the context menu")
                print("2. Click the file and use the download button in the toolbar")
                print(f"3. Save the file to: {output_path}")
                
                # Show interactive prompt for manual intervention
                user_response = input(f"\nWas the file '{file_name}' downloaded successfully? (y/n): ")
                if user_response.lower() == 'y':
                    logger.info(f"User confirmed successful manual download of {file_name}")
                    successful_downloads += 1
                else:
                    logger.warning(f"Manual download reported as unsuccessful for {file_name}")
                    failed_downloads += 1
                
            except Exception as e:
                logger.error(f"Error downloading {file['name']}: {e}")
                failed_downloads += 1
        
        return successful_downloads, failed_downloads
    except Exception as e:
        logger.error(f"Error in download process: {e}")
        return 0, 0
