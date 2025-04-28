"""
Browser setup and authentication functionality.
"""
import os
import logging
import time
import urllib.parse
import msal
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

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
        webdriver.support.ui.WebDriverWait(browser, 300).until(
            lambda driver: urllib.parse.urlparse(driver.current_url).netloc.endswith("sharepoint.com")
        )
        
        logger.info("Authentication complete. SharePoint page loaded.")
        return True
    except Exception as e:
        logger.error(f"Error during browser authentication: {e}")
        return False
