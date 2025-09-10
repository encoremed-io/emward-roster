# utils/logger.py
import logging
import sys
from config.paths import LOG_PATH

# Ensure directory exists
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("schedule")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if imported multiple times
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Stream handler (stdout -> docker logs)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # Add both handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
