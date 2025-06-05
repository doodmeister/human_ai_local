"""
Logging configuration for the Human-AI Cognition Framework
"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    include_timestamps: bool = True,
    include_module: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the cognitive system
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        include_timestamps: Whether to include timestamps in log messages
        include_module: Whether to include module names in log messages
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("cognitive_ai")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    format_parts = []
    if include_timestamps:
        format_parts.append("%(asctime)s")
    format_parts.append("%(levelname)s")
    if include_module:
        format_parts.append("%(name)s")
    format_parts.append("%(message)s")
    
    formatter = logging.Formatter(" - ".join(format_parts))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_cognitive_logger(name: str = "cognitive_ai") -> logging.Logger:
    """Get a logger instance for cognitive components"""
    return logging.getLogger(name)

# Create default logger
logger = setup_logging()
