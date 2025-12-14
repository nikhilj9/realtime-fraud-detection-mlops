"""Centralized logging configuration."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

_LOGGERS = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Path = Path("logs")
) -> logging.Logger:
    """
    Get or create configured logger.
    
    Creates separate log files per module with date stamp.
    Example: logs/api_2025-01-15.log
    """
    
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (always enabled)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler (auto-generated)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract module name from full path (e.g., "src.api.app" -> "api")
    module_parts = name.split('.')
    if len(module_parts) >= 2:
        module_name = module_parts[-1]
    else:
        module_name = name
    
    # Add date stamp
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{module_name}_{date_str}.log"
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    _LOGGERS[name] = logger
    return logger