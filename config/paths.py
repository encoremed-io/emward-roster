from pathlib import Path

# === Base project path ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# === Common directories ===
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"
LOG_DIR = PROJECT_ROOT  # or change to PROJECT_ROOT / "logs" in future

# === Default log file path ===
LOG_PATH = LOG_DIR / "schedule_run.log"
