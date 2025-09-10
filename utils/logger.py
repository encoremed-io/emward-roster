# utils/logger.py
import logging
from config.paths import LOG_PATH

# Make sure the directory exists
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("schedule")
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers if re-imported
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
