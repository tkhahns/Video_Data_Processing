"""
Entry point for the speech-to-text module.
"""
import os
import sys

# Add the project root directory to sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Now import the main function using an absolute import
from src.speech_to_text.main import main

if __name__ == "__main__":
    main()
