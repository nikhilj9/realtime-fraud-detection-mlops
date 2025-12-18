"""Centralized logging configuration for production MLOps."""

import logging
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from contextvars import ContextVar

# Stores the current request ID (thread-safe)
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Unique ID for this application run (set once at startup)
RUN_ID: str = uuid.uuid4().hex[:8]
RUN_START_TIME: datetime = datetime.now(timezone.utc)

# Logger cache
_LOGGERS = {}


class UTCFormatter(logging.Formatter):
    """Formatter that always uses UTC timezone."""
    
    converter = lambda *args: datetime.now(timezone.utc).timetuple()
    
    def format(self, record):
        # Add request ID to the record
        record.request_id = request_id_var.get()
        return super().format(record)


def get_run_id() -> str:
    """Get the current run ID."""
    return RUN_ID


def set_request_id(request_id: str = None) -> str:
    """Set request ID for current context. Returns the ID."""
    if request_id is None:
        request_id = uuid.uuid4().hex[:8]
    request_id_var.set(request_id)
    return request_id


def clear_request_id():
    """Clear request ID after request completes."""
    request_id_var.set("-")


def log_session_start(logger: logging.Logger):
    """Log a clear session start banner."""
    banner = "=" * 80
    logger.info(banner)
    logger.info(f"SESSION START | Run ID: {RUN_ID}")
    logger.info(banner)


def log_session_end(logger: logging.Logger):
    """Log a clear session end banner."""
    duration = datetime.now(timezone.utc) - RUN_START_TIME
    minutes, seconds = divmod(int(duration.total_seconds()), 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"
    
    banner = "=" * 80
    logger.info(banner)
    logger.info(f"SESSION END | Run ID: {RUN_ID} | Duration: {duration_str}")
    logger.info(banner)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Path = Path("logs")
) -> logging.Logger:
    """
    Get or create configured logger.
    
    Features:
    - UTC timestamps (no timezone confusion)
    - Request ID tracking (trace individual requests)
    - Session separators (clear start/end markers)
    - Fixed-width formatting (easy to read)
    """
    
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    if logger.handlers:
        return logger
    
    # Format: timestamp | level | module | [request_id] | message
    log_format = (
        "%(asctime)s UTC | "
        "%(levelname)-8s | "
        "%(name)-20s | "
        "[%(request_id)s] | "
        "%(message)s"
    )
    
    formatter = UTCFormatter(
        fmt=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract module name
    module_parts = name.split(".")
    if len(module_parts) >= 2:
        module_name = module_parts[-1]
    else:
        module_name = name
    
    # Date stamp (UTC)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"{module_name}_{date_str}.log"
    
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    _LOGGERS[name] = logger
    return logger