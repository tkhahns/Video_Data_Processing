"""
Utility modules for the Video Data Processing project.
"""

# Import commonly used modules for easy access
from . import colored_logging
from . import init_logging
from . import merge_features
from . import json_utils

# Provide direct access to key functions
from .init_logging import get_logger, initialize_logging
from .colored_logging import setup_colored_logging, setup_colorama_logging
from .merge_features import create_pipeline_output, create_summary_report
from .json_utils import save_json, load_json, format_json, JSON_INDENT

__all__ = [
    'colored_logging',
    'init_logging',
    'merge_features',
    'json_utils',
    'get_logger',
    'initialize_logging',
    'setup_colored_logging',
    'setup_colorama_logging',
    'create_pipeline_output',
    'create_summary_report',
    'save_json',
    'load_json',
    'format_json',
    'JSON_INDENT',
]
