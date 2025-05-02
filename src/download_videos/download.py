"""
File download functionality.
"""
import os
import logging
import time
import requests
import sys
from selenium import webdriver

# Try multiple approaches for importing the logging modules
try:
    # First try absolute imports
    import init_logging
except ImportError:
    # Fall back to relative imports if the module structure permits it
    try:
        from .. import init_logging
    except ImportError:
        # Last resort: add parent directory to path
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        import init_logging

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
                
                # Get the direct URL to the file
                direct_url = file['element'].get_attribute("href")
                logger.info(f"File URL: {direct_url}")
                
                # Method 1: Direct download using requests and current browser cookies
                try:
                    # Get all cookies from the browser
                    cookies = browser.get_cookies()
                    
                    # Create a requests session and add cookies
                    session = requests.Session()
                    for cookie in cookies:
                        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])
                    
                    logger.info(f"Downloading file directly using requests: {direct_url}")
                    
                    # Make the request with proper headers
                    headers = {
                        'User-Agent': browser.execute_script("return navigator.userAgent"),
                        'Accept': 'application/octet-stream'
                    }
                    
                    # Stream the download
                    response = session.get(direct_url, headers=headers, stream=True)
                    response.raise_for_status()
                    
                    # Save the file
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kibibyte
                    
                    print(f"\n⬇️ Downloading {file_name} ({total_size/1024/1024:.2f} MB)")
                    
                    with open(output_path, 'wb') as f:
                        downloaded = 0
                        for data in response.iter_content(block_size):
                            downloaded += len(data)
                            f.write(data)
                            done = downloaded / total_size * 100 if total_size > 0 else 0
                            print(f"\rProgress: {done:.2f}% ({downloaded/1024/1024:.2f} MB)", end="")
                    
                    print()  # New line after progress bar
                    
                    logger.info(f"File successfully downloaded to {output_path}")
                    successful_downloads += 1
                    continue
                    
                except Exception as e:
                    logger.warning(f"Direct download failed: {e}")
                    
                # Method 2: Try Selenium download capabilities
                try:
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
                    
                    # Try right-click context menu first
                    webdriver.ActionChains(browser).context_click(file['element']).perform()
                    time.sleep(1)
                    
                    # Look for download option in context menu
                    context_menu_download = browser.find_elements("xpath", 
                                                               "//button[contains(@name, 'Download') or contains(text(), 'Download')]")
                    if context_menu_download:
                        context_menu_download[0].click()
                        logger.info("Initiated download via context menu")
                    else:
                        # Click elsewhere to close context menu
                        webdriver.ActionChains(browser).move_by_offset(10, 10).click().perform()
                        
                        # Fall back to direct click with modified URL if possible
                        logger.info("No context menu download option found, trying direct click")
                        file['element'].click()
                    
                    # Wait for download to begin and complete
                    max_wait = 120  # seconds
                    poll_interval = 2  # seconds
                    file_downloading = True
                    start_time = time.time()
                    
                    print(f"\n⬇️ Downloading {file_name}...")
                    
                    while file_downloading and (time.time() - start_time) < max_wait:
                        # Check if file exists
                        if os.path.exists(output_path):
                            # Check if download is complete (not a .crdownload file)
                            if not output_path.endswith('.crdownload') and not os.path.exists(f"{output_path}.crdownload"):
                                file_downloading = False
                                break
                        
                        # Check for common partial download extensions
                        partial_download_path = f"{output_path}.crdownload"
                        if os.path.exists(partial_download_path):
                            partial_size = os.path.getsize(partial_download_path)
                            print(f"\rDownloading... {partial_size/1024/1024:.2f} MB", end="")
                        
                        time.sleep(poll_interval)
                    
                    if not file_downloading:
                        logger.info(f"File successfully downloaded to {output_path}")
                        successful_downloads += 1
                    else:
                        logger.warning(f"Download timed out or couldn't be tracked")
                        print("\nCouldn't automatically track the download.")
                        input(f"Press Enter after '{file_name}' has finished downloading...")
                        if os.path.exists(output_path):
                            successful_downloads += 1
                        else:
                            failed_downloads += 1
                    
                    continue
                        
                except Exception as e:
                    logger.warning(f"Selenium download failed: {e}")
                
                # Method 3: Last resort - Manual download instructions
                print(f"\n⚠️ Automatic download methods failed for {file_name}")
                print(f"Please download the file manually following these steps:")
                print(f"1. A new tab will open with the file")
                print(f"2. Right-click the video and select 'Save video as...'")
                print(f"3. Save it to: {output_path}")
                
                # Open in new tab
                browser.execute_script("window.open(arguments[0]);", direct_url)
                time.sleep(2)
                
                # Switch to new tab
                browser.switch_to.window(browser.window_handles[-1])
                
                # Wait for user to manually download
                input(f"\nPress Enter after you've manually downloaded '{file_name}'...")
                
                # Switch back to main tab
                browser.switch_to.window(browser.window_handles[0])
                
                if os.path.exists(output_path):
                    logger.info(f"File successfully downloaded to {output_path}")
                    successful_downloads += 1
                else:
                    logger.warning(f"File not found at {output_path} after manual download")
                    failed_downloads += 1
                
            except Exception as e:
                logger.error(f"Error downloading {file['name']}: {e}")
                failed_downloads += 1
        
        return successful_downloads, failed_downloads
    except Exception as e:
        logger.error(f"Error in download process: {e}")
        return 0, 0
