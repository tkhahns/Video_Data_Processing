"""
Script to download mp4 videos from a SharePoint folder with MFA support.
Specifically designed for folders shared via sharepoint.com links.
"""
import os
import argparse
import logging
import urllib.parse
import time
from pathlib import Path
import requests
import msal
from office365.graph_client import GraphClient
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.runtime.auth.token_response import TokenResponse
from office365.runtime.auth.client_credential import ClientCredential
from office365.runtime.http.request_options import RequestOptions

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

def get_sharepoint_folder_contents(graph_client, url_info):
    """
    Get the contents of a SharePoint folder using the Microsoft Graph API.
    """
    try:
        host = url_info['host']
        site_path = url_info['site_path']
        folder_path = url_info['folder_path']
        
        # Get the site
        logger.info(f"Getting site information for host: {host}, path: {site_path}")
        
        # For personal sites (OneDrive), the format is different
        if 'personal' in site_path:
            # Extract the username from the site path
            username = site_path.split('/personal/')[1]
            drive_query = f"/drives?$filter=driveType eq 'business' and owner/user/userPrincipalName eq '{username}'"
            drives = graph_client.me.execute_query_with_json(drive_query)
            
            if not drives or 'value' not in drives or not drives['value']:
                logger.error("Could not find OneDrive for the user")
                return None
                
            drive_id = drives['value'][0]['id']
            logger.info(f"Found OneDrive with ID: {drive_id}")
            
            # Get the folder items
            path = folder_path.replace('Documents/', '')
            folder_items = graph_client.drives[drive_id].root.get_by_path(path).children.get().execute_query()
            
            return folder_items
        else:
            # For team sites
            site_query = f"/sites/{host}:/{site_path}"
            site = graph_client.sites.get_by_path(site_path=site_query).get().execute_query()
            
            # Get the drive (document library)
            drives = graph_client.sites[site.id].drives.get().execute_query()
            if not drives or len(drives) == 0:
                logger.error("Could not find document library")
                return None
                
            drive_id = drives[0].id
            
            # Get the folder items
            path = folder_path.replace('Documents/', '')
            folder_items = graph_client.drives[drive_id].root.get_by_path(path).children.get().execute_query()
            
            return folder_items
            
    except Exception as e:
        logger.error(f"Error getting folder contents: {e}")
        return None

def download_file(graph_client, file_item, output_dir):
    """
    Download a file from SharePoint using its Graph API item.
    """
    try:
        filename = file_item.name
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
        with open(output_path, "wb") as local_file:
            file_item.download(local_file).execute_query()
        
        logger.info(f"Successfully downloaded {filename}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading file {file_item.name}: {e}")
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
    
    # Create Graph client with the access token
    auth_context = AuthenticationContext("https://graph.microsoft.com")
    auth_context.acquire_token_function = lambda: TokenResponse.from_jwt_token(access_token)
    graph_client = GraphClient(auth_context)
    
    # Get folder contents
    folder_items = get_sharepoint_folder_contents(graph_client, url_info)
    if not folder_items:
        logger.error("Failed to get folder contents")
        return
    
    # Download each video file
    successful_downloads = 0
    failed_downloads = 0
    
    for item in folder_items:
        if hasattr(item, 'file') and item.file:
            if download_file(graph_client, item, args.output_dir):
                successful_downloads += 1
            else:
                failed_downloads += 1
    
    logger.info(f"Download complete: {successful_downloads} files downloaded successfully, {failed_downloads} failed")

if __name__ == "__main__":
    main()
