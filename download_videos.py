"""
Script to download files from SharePoint using browser automation.
Handles various SharePoint layouts including classic view.
"""
import os
import sys
import argparse
import logging
import json
import time
import urllib.parse
from pathlib import Path
import requests
import msal
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Videos directory
VIDEOS_DIR = Path("./data/videos")

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def get_auth_token_interactive(tenant_id="organizations"):
    """
    Get authentication token using interactive sign-in with MFA support.
    """
    # Microsoft Graph API scope
    scopes = ["https://graph.microsoft.com/.default"]
    
    # Using Microsoft's public client app ID for device code flow
    client_id = "14d82eec-204b-4c2f-b7e8-296a70dab67e"  # Microsoft Graph PowerShell client ID
    
    # Create a public client app
    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}"
    )
    
    # Attempt to get token from cache first
    accounts = app.get_accounts()
    result = None
    if accounts:
        logger.info("Account found in cache, attempting to use cached token")
        result = app.acquire_token_silent(scopes, account=accounts[0])
    
    if not result:
        logger.info("No suitable token in cache, getting new token via interactive login")
        try:
            result = app.acquire_token_interactive(scopes=scopes)
        except Exception as e:
            logger.error(f"Error during interactive authentication: {e}")
            return None
    
    if "access_token" in result:
        logger.info("Authentication successful")
        return result["access_token"]
    else:
        logger.error(f"Authentication failed: {result.get('error_description', 'Unknown error')}")
        return None

def setup_browser(headless=False, output_dir=None):
    """Set up and return a browser instance."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    if output_dir:
        prefs = {
            "download.default_directory": os.path.abspath(output_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False
        }
        chrome_options.add_experimental_option("prefs", prefs)
    
    try:
        browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        return browser
    except Exception as e:
        logger.error(f"Error setting up browser: {e}")
        return None

def authenticate_with_selenium(browser, url):
    """
    Authenticate to SharePoint using Selenium browser automation.
    Returns the browser session with authentication cookies.
    """
    try:
        logger.info(f"Opening SharePoint URL in browser: {url}")
        browser.get(url)
        
        # Wait for authentication to complete (URL should change to SharePoint domain)
        WebDriverWait(browser, 300).until(
            lambda driver: urllib.parse.urlparse(driver.current_url).netloc.endswith("sharepoint.com")
        )
        
        logger.info("Authentication complete. SharePoint page loaded.")
        return True
    except Exception as e:
        logger.error(f"Error during browser authentication: {e}")
        return False

def find_all_files(browser):
    """Find all files in the current SharePoint page using classic view selectors."""
    try:
        # Wait for the page to fully load
        logger.info("Waiting for SharePoint page to fully load...")
        time.sleep(5)  # Give the page some time to initialize
        
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
                # Method 3: Try any links with MP4/video extensions
                logger.info("Trying to find any links with video extensions...")
                all_links = browser.find_elements(By.TAG_NAME, "a")
                file_items = [link for link in all_links if 
                             any(ext in link.get_attribute("href").upper() for ext in [".MP4", ".MOV", ".AVI"]) or
                             any(ext in (link.get_attribute("aria-label") or "").upper() for ext in ["MP4", "MOV", "AVI"])]
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
                    if not any(file_name.upper().endswith(ext) for ext in [".MP4", ".MOV", ".AVI"]):
                        # Try to extract extension from aria-label
                        if "mp4" in aria_label.lower():
                            file_name += ".mp4"
                        elif "mov" in aria_label.lower():
                            file_name += ".mov"
                        elif "avi" in aria_label.lower():
                            file_name += ".avi"
                else:
                    # Fall back to the text content
                    file_name = item.text.strip()
                    
                    # If the text doesn't include extension, try to add it from href
                    if not any(file_name.upper().endswith(ext) for ext in [".MP4", ".MOV", ".AVI"]):
                        # Try to extract from the href
                        if ".MP4" in href.upper():
                            file_name += ".mp4"
                        elif ".MOV" in href.upper():
                            file_name += ".mov"
                        elif ".AVI" in href.upper():
                            file_name += ".avi"
                
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
                    'is_video': any(file_name.lower().endswith(ext) for ext in 
                                   ['.mp4', '.mov', '.avi', '.wmv', '.mkv'])
                })
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

def display_file_list(file_list):
    """Display the list of files with indices for selection."""
    video_files = [f for f in file_list if f['is_video']]
    other_files = [f for f in file_list if not f['is_video']]
    
    print("\n=== Files available in SharePoint folder ===")
    
    if video_files:
        print("\nVIDEO FILES:")
        print("-----------")
        for file in video_files:
            print(f"[{file['index']}] {file['name']} ({file['size']})")
    
    if other_files:
        print("\nOTHER FILES:")
        print("-----------")
        for file in other_files:
            print(f"[{file['index']}] {file['name']} ({file['size']})")
    
    print("\n=========================================")

def prompt_for_file_selection(file_list):
    """Prompt user to select files to download."""
    while True:
        print("\nOptions:")
        print("- Enter numbers (e.g., '1,3,5') to download specific files")
        print("- Enter 'videos' to download all video files")
        print("- Enter 'all' to download all files")
        print("- Enter 'q' to quit")
        
        selection = input("\nYour selection: ")
        
        if selection.lower() == 'q':
            return []
            
        if selection.lower() == 'all':
            return file_list
            
        if selection.lower() == 'videos':
            return [f for f in file_list if f['is_video']]
        
        try:
            # Parse comma-separated numbers
            indices = [int(idx.strip()) for idx in selection.split(',')]
            selected_files = []
            
            for idx in indices:
                matching_files = [f for f in file_list if f['index'] == idx]
                if matching_files:
                    selected_files.append(matching_files[0])
                else:
                    print(f"Error: {idx} is not a valid file number")
                    break
            else:
                # If no break occurred in the for loop
                return selected_files
                
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")

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
                    context_menu_download = browser.find_elements(By.XPATH, 
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
        return
    
    try:
        # Authenticate to SharePoint
        if not authenticate_with_selenium(browser, args.url):
            logger.error("Authentication failed")
            return
        
        # Find all files
        all_files = find_all_files(browser)
        if not all_files:
            logger.error("No files found")
            return
        
        logger.info(f"Found {len(all_files)} files")
        
        # Display file list
        display_file_list(all_files)
        
        # If list-only mode, exit here
        if args.list_only:
            logger.info("List-only mode, exiting without downloading")
            return
        
        # Prompt for file selection
        selected_files = prompt_for_file_selection(all_files)
        
        if not selected_files:
            logger.info("No files selected for download")
            return
            
        logger.info(f"Selected {len(selected_files)} files for download")
        
        # Download selected files
        successful, failed = download_selected_files(browser, selected_files, args.output_dir)
        logger.info(f"Download complete: {successful} files downloaded successfully, {failed} failed")
        
    finally:
        # Always close the browser
        if browser:
            browser.quit()

if __name__ == "__main__":
    main()