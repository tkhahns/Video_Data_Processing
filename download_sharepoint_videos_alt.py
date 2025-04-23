"""
Script to download mp4 videos from a SharePoint folder with MFA support.
Uses Microsoft Graph SDK for Python instead of Office365-REST-Python-Client.
"""
import os
import argparse
import logging
import urllib.parse
from pathlib import Path
import requests
import msal
import json

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

def parse_sharepoint_url(url):
    """
    Parse a SharePoint URL to extract site information and folder path.
    Handles various SharePoint URL formats including sharing links.
    """
    try:
        parsed_url = urllib.parse.urlparse(url)
        
        # Get the host (tenant)
        host = parsed_url.netloc
        
        # Try to extract root folder path from query parameters
        query_params = urllib.parse.parse_qs(parsed_url.query)
        root_folder = None
        
        if 'RootFolder' in query_params:
            root_folder = query_params['RootFolder'][0]
            logger.info(f"Found RootFolder in URL: {root_folder}")
        
        # If we have a root folder path, parse it
        if root_folder:
            # Extract site path and folder path from the root folder
            parts = root_folder.split('/Documents/')
            if len(parts) == 2:
                site_path = parts[0]
                folder_path = 'Documents/' + parts[1]
                
                # Clean up the site path
                site_path = site_path.strip('/')
                
                return {
                    'host': host,
                    'site_path': site_path,
                    'folder_path': folder_path
                }
        
        # If we couldn't extract from the RootFolder, try to parse the URL path
        path_parts = parsed_url.path.split('/')
        site_path = ""
        folder_path = ""
        
        # Look for 'Documents' to separate site path from folder path
        for i, part in enumerate(path_parts):
            if part == 'Documents':
                site_path = '/'.join(path_parts[:i])
                folder_path = 'Documents/' + '/'.join(path_parts[i+1:])
                break
        
        return {
            'host': host,
            'site_path': site_path.strip('/'),
            'folder_path': folder_path
        }
        
    except Exception as e:
        logger.error(f"Error parsing SharePoint URL: {e}")
        return None

def get_sharepoint_folder_contents(access_token, url_info):
    """
    Get the contents of a SharePoint folder using Microsoft Graph API directly.
    """
    try:
        host = url_info['host']
        site_path = url_info['site_path']
        folder_path = url_info['folder_path']
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        # For personal sites (OneDrive), the format is different
        if 'personal' in site_path:
            # Extract the username from the site path
            username = site_path.split('/personal/')[1].replace('_', '@').replace('_', '.')
            
            # First, get the drive ID
            drive_url = f"https://graph.microsoft.com/v1.0/users/{username}/drive"
            response = requests.get(drive_url, headers=headers)
            response.raise_for_status()
            drive_data = response.json()
            drive_id = drive_data['id']
            
            logger.info(f"Found OneDrive with ID: {drive_id}")
            
            # Get the folder items
            path = folder_path.replace('Documents/', '')
            encoded_path = urllib.parse.quote(path)
            items_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{encoded_path}:/children"
            
            response = requests.get(items_url, headers=headers)
            response.raise_for_status()
            items_data = response.json()
            
            return items_data.get('value', [])
        else:
            # For team sites
            site_url = f"https://graph.microsoft.com/v1.0/sites/{host}:/{site_path}"
            response = requests.get(site_url, headers=headers)
            response.raise_for_status()
            site_data = response.json()
            site_id = site_data['id']
            
            # Get the drive (document library)
            drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
            response = requests.get(drives_url, headers=headers)
            response.raise_for_status()
            drives_data = response.json()
            
            if not drives_data.get('value'):
                logger.error("Could not find document library")
                return None
                
            drive_id = drives_data['value'][0]['id']
            
            # Get the folder items
            path = folder_path.replace('Documents/', '')
            encoded_path = urllib.parse.quote(path)
            items_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{encoded_path}:/children"
            
            response = requests.get(items_url, headers=headers)
            response.raise_for_status()
            items_data = response.json()
            
            return items_data.get('value', [])
            
    except Exception as e:
        logger.error(f"Error getting folder contents: {e}")
        return None

def download_file(access_token, file_item, output_dir):
    """
    Download a file from SharePoint using its Graph API item.
    """
    try:
        filename = file_item.get('name')
        download_url = file_item.get('@microsoft.graph.downloadUrl')
        
        if not download_url:
            logger.error(f"No download URL available for {filename}")
            return False
            
        output_path = os.path.join(output_dir, filename)
        
        # Check if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            return True
        
        # Skip non-video files
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
            logger.info(f"Skipping non-video file: {filename}")
            return True
            
        # Download file
        logger.info(f"Downloading {filename} to {output_path}")
        
        # Get download URL and download the file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as local_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    local_file.write(chunk)
        
        logger.info(f"Successfully downloaded {filename}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading file {file_item.get('name')}: {e}")
        return False

def main():
    """Main function to download videos from a SharePoint folder with MFA support."""
    parser = argparse.ArgumentParser(description="Download videos from a SharePoint folder with MFA support")
    parser.add_argument(
        "--url", 
        required=True,
        help="SharePoint folder URL containing videos"
    )
    parser.add_argument(
        "--output-dir", 
        default=str(VIDEOS_DIR),
        help="Directory to save the downloaded videos"
    )
    parser.add_argument(
        "--tenant-id",
        default="organizations",
        help="Microsoft tenant ID (default: organizations)"
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir_exists(args.output_dir)
    
    # Parse the SharePoint URL
    url_info = parse_sharepoint_url(args.url)
    if not url_info:
        logger.error("Failed to parse SharePoint URL")
        return
        
    logger.info(f"Parsed URL info: {url_info}")
    
    # Get authentication token interactively (supports MFA)
    access_token = get_auth_token_interactive(args.tenant_id)
    if not access_token:
        logger.error("Failed to get authentication token")
        return
    
    # Get folder contents directly using the access token
    folder_items = get_sharepoint_folder_contents(access_token, url_info)
    if not folder_items:
        logger.error("Failed to get folder contents")
        return
    
    # Download each video file
    successful_downloads = 0
    failed_downloads = 0
    
    for item in folder_items:
        if item.get('file'):
            if download_file(access_token, item, args.output_dir):
                successful_downloads += 1
            else:
                failed_downloads += 1
    
    logger.info(f"Download complete: {successful_downloads} files downloaded successfully, {failed_downloads} failed")

if __name__ == "__main__":
    main()
