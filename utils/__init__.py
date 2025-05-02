"""
Utility modules for the Video Data Processing project.
"""

# Import commonly used modules for easy access
from . import colored_logging
from . import init_logging

# Provide direct access to key functions
from .init_logging import get_logger, initialize_logging
from .colored_logging import setup_colored_logging, setup_colorama_logging

__all__ = [
    'colored_logging',
    'init_logging',
    'get_logger',
    'initialize_logging',
    'setup_colored_logging',
    'setup_colorama_logging',
]
