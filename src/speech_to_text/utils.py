"""
Utility functions for speech-to-text processing.
This is a compatibility layer that imports from speech_features.py.
"""

import logging
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Import all functions from speech_features to maintain backward compatibility
try:
    from .speech_features import *
    
    # Show deprecation warning when utils is imported directly
    warnings.warn(
        "The utils module is deprecated and will be removed in a future version. "
        "Please import from speech_features instead.", 
        DeprecationWarning, 
        stacklevel=2
    )
    logger.info("Using speech_features module through utils compatibility layer")
    
except ImportError as e:
    logger.error(f"Failed to import from speech_features: {e}")
    # Re-raise to avoid silent failures
    raise
