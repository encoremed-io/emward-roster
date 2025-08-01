import json
from config.paths import CONFIG_DIR

"""
Loads configuration constants from config/constants.json and exposes them as module-level variables.
Edit constants.json to change values; import from utils.constants to use in code.
"""

# Build the path to the JSON config file
CONSTANTS_PATH = CONFIG_DIR / "constants.json"

with open(CONSTANTS_PATH, "r", encoding="utf-8") as f:
    _constants = json.load(f)

# Expose constants as variables
SHIFT_LABELS = _constants["SHIFT_LABELS"]
SHIFT_DURATIONS = _constants["SHIFT_DURATIONS"]
AVG_HOURS = _constants["AVG_HOURS"]
DAYS_PER_WEEK = _constants["DAYS_PER_WEEK"]
NO_WORK_LABELS = _constants["NO_WORK_LABELS"]

MIN_NURSES_PER_SHIFT = _constants["MIN_NURSES_PER_SHIFT"]
MIN_SENIORS_PER_SHIFT = _constants["MIN_SENIORS_PER_SHIFT"]
MAX_WEEKLY_HOURS = _constants["MAX_WEEKLY_HOURS"]
MIN_WEEKLY_REST = _constants["MIN_WEEKLY_REST"]
MAX_MC_DAYS_PER_WEEK = _constants["MAX_MC_DAYS_PER_WEEK"]
MAX_CONSECUTIVE_MC = _constants["MAX_CONSECUTIVE_MC"]

PREFERRED_WEEKLY_HOURS = _constants["PREFERRED_WEEKLY_HOURS"]
MIN_ACCEPTABLE_WEEKLY_HOURS = _constants["MIN_ACCEPTABLE_WEEKLY_HOURS"]
PREF_HOURS_PENALTY = _constants["PREF_HOURS_PENALTY"]

DOUBLE_SHIFT_PENALTY = _constants["DOUBLE_SHIFT_PENALTY"]

AM_COVERAGE_MIN_PERCENT = _constants["AM_COVERAGE_MIN_PERCENT"]
AM_COVERAGE_PENALTY = _constants["AM_COVERAGE_PENALTY"]
AM_COVERAGE_RELAX_STEP = _constants["AM_COVERAGE_RELAX_STEP"]

AM_SENIOR_MIN_PERCENT = _constants["AM_SENIOR_MIN_PERCENT"]
AM_SENIOR_PENALTY = _constants["AM_SENIOR_PENALTY"]
AM_SENIOR_RELAX_STEP = _constants["AM_SENIOR_RELAX_STEP"]

PREF_MISS_PENALTY = _constants["PREF_MISS_PENALTY"]
FAIRNESS_GAP_PENALTY = _constants["FAIRNESS_GAP_PENALTY"]
FAIRNESS_GAP_THRESHOLD = _constants["FAIRNESS_GAP_THRESHOLD"]
SHIFT_IMBALANCE_PENALTY = _constants["SHIFT_IMBALANCE_PENALTY"]
SHIFT_IMBALANCE_THRESHOLD = _constants["SHIFT_IMBALANCE_THRESHOLD"]
