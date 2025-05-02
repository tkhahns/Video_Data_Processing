"""
Utility module for setting up colored logging in terminal output.
"""

import logging
import sys

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log records based on their level."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BLUE,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE + Colors.BOLD
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        # Save original format
        original_format = self._style._fmt
        
        # Apply color formatting based on log level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        
        # Add color codes to level name and remove them after
        record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        
        # Call the original format method
        result = super().format(record)
        
        # Restore original format
        self._style._fmt = original_format
        
        return result


class ColoredConsoleHandler(logging.StreamHandler):
    """Handler for outputting colored logs to the console."""
    
    def __init__(self, stream=sys.stdout):
        super().__init__(stream)
        self.setFormatter(ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))


def setup_colored_logging(logger=None, level=logging.INFO):
    """
    Set up colored logging for the specified logger or the root logger.
    
    Args:
        logger: Logger to set up colored logging for (or None for root logger)
        level: Logging level to set
    
    Returns:
        The configured logger
    """
    # Use the specified logger or get the root logger
    if logger is None:
        logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set the log level
    logger.setLevel(level)
    
    # Add colored console handler
    handler = ColoredConsoleHandler()
    logger.addHandler(handler)
    
    return logger


# Alternative implementation using colorama (requires pip install colorama)
try:
    import colorama
    
    def setup_colorama_logging(logger=None, level=logging.INFO):
        """
        Set up colored logging using the colorama library for cross-platform support.
        
        Args:
            logger: Logger to set up colored logging for (or None for root logger)
            level: Logging level to set
            
        Returns:
            The configured logger
        """
        from colorama import Fore, Back, Style
        colorama.init()
        
        class ColoramaFormatter(logging.Formatter):
            LEVEL_COLORS = {
                logging.DEBUG: Fore.BLUE,
                logging.INFO: Fore.GREEN,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Fore.WHITE + Back.RED + Style.BRIGHT
            }
            
            def format(self, record):
                color = self.LEVEL_COLORS.get(record.levelno, '')
                record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
                return super().format(record)
        
        # Use the specified logger or get the root logger
        if logger is None:
            logger = logging.getLogger()
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Set the log level
        logger.setLevel(level)
        
        # Add handler with colorama formatter
        handler = logging.StreamHandler()
        formatter = ColoramaFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
except ImportError:
    # If colorama is not available, provide a fallback
    def setup_colorama_logging(logger=None, level=logging.INFO):
        """Fallback if colorama is not installed."""
        print("Warning: colorama not installed. Use 'pip install colorama' for better cross-platform color support.")
        return setup_colored_logging(logger, level)
