"""
Initialize colored logging for the entire repository.
Import this module to set up colored logging for any module.
"""

import logging
import sys
import os

# Import from the same package
from .colored_logging import setup_colorama_logging

# Set up colored logging for the root logger
def initialize_logging(log_level=logging.INFO):
    """
    Initialize colored logging for the entire application.
    
    Args:
        log_level: Logging level to set (default: INFO)
    """
    root_logger = logging.getLogger()
    setup_colorama_logging(root_logger, log_level)
    return root_logger

def get_logger(name, level=None):
    """
    Get a logger with colored logging configured.
    
    Args:
        name: Name of the logger (typically __name__)
        level: Optional specific level for this logger
    
    Returns:
        Logger with colored logging configured
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger

# Initialize logging when the module is imported
initialize_logging()
