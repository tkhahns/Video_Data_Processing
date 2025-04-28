"""
User interface components for file selection and display.
"""
import logging

logger = logging.getLogger(__name__)

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
