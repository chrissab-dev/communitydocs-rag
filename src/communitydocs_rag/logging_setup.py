import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure application-wide logging.

    - Log level comes from LOG_LEVEL env var (default INFO) unless overridden.
    - Includes timestamps, level, logger name (module), and message.
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    # Convert string level (e.g., "INFO") to logging constant
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Configure root logger once. force=True resets handlers if already configured.
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-level logger. Use like: logger = get_logger(__name__)
    """
    return logging.getLogger(name)
