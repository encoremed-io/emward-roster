# utils/logger.py
import logging
import sys

logger = logging.getLogger("schedule")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if re-imported
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
