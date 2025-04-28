"""
Entry point for running the download_videos package as a module.
Enables execution with: python -m download_videos
"""
import sys
from download_videos.main import main

if __name__ == "__main__":
    sys.exit(main())
