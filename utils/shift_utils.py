import pandas as pd
from datetime import date as dt_date
from collections import defaultdict
from typing import Any, Dict, Set, List, Union, Optional, Tuple

# --- Utility to convert column label to day-index ---
def compute_label_offset(label: Any, date_start: dt_date) -> int:
    """Given a column label (Timestamp, date, string, or any hashable), return the day-index offset."""
    if isinstance(label, pd.Timestamp):
        d = label.date()
    elif isinstance(label, dt_date):
        d = label
    else:
        d = pd.to_datetime(str(label)).date()
    return (d - date_start).days


def get_mc_days(
    preferences_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    start_date: Union[pd.Timestamp, dt_date],
    active_days: int
) -> Dict[str, Set[int]]:
    """
    Returns nurse_name -> set of MC day-indices (0..active_days-1).
    """
    date_start = start_date.date() if isinstance(start_date, pd.Timestamp) else start_date
    nurse_names = [n.strip().upper() for n in profiles_df['Name']]
    mc_days: Dict[str, Set[int]] = {n: set() for n in nurse_names}

    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        if nm not in mc_days:
            continue
        for label, val in row.items():
            if pd.notna(val) and str(val).strip().upper() == 'MC':
                offset = compute_label_offset(label, date_start)
                if 0 <= offset < active_days:
                    mc_days[nm].add(offset)
    return mc_days


def get_shift_preferences(
    preferences_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    start_date: Union[pd.Timestamp, dt_date],
    active_days: int,
    shift_labels: List[str]
) -> Dict[str, Dict[int, int]]:
    """
    Returns nurse_name -> { day_index: shift_index } for non-MC preferences.
    """
    date_start = start_date.date() if isinstance(start_date, pd.Timestamp) else start_date
    nurse_names = [n.strip().upper() for n in profiles_df['Name']]
    shift_str_to_idx = {label.upper(): idx for idx, label in enumerate(shift_labels)}
    shift_prefs: Dict[str, Dict[int, int]] = {n: {} for n in nurse_names}

    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        if nm not in shift_prefs:
            continue
        for label, val in row.items():
            if pd.notna(val):
                code = str(val).strip().upper()
                if code in shift_str_to_idx:
                    offset = compute_label_offset(label, date_start)
                    if 0 <= offset < active_days:
                        shift_prefs[nm][offset] = shift_str_to_idx[code]
    return shift_prefs


def get_senior_set(profiles_df):
    """Get set of senior nurses."""
    return {
        str(row["Name"]).strip().upper()
        for _, row in profiles_df.iterrows()
        if row.get("Title", "").upper() == "SENIOR"
    }


def get_el_days(fixed_assignments, nurse_names):
    """Extract EL days from fixed assignments."""
    el_days = {n: set() for n in nurse_names}
    for (nurse, d), label in fixed_assignments.items():
        if label.strip().upper() == "EL":
            el_days[nurse].add(d)
    return el_days


def normalize_fixed_assignments(
    fixed: Optional[Dict[Tuple[str,int],str]],
    valid_names: Set[str],
    num_days: int
) -> Dict[Tuple[str,int],str]:
    """Upper‐cases names/shifts, strips whitespace, and validates keys/indices."""
    if fixed is None:
        return {}
    cleaned: Dict[Tuple[str,int],str] = {}
    for (name, day), shift in fixed.items():
        n = name.strip().upper()
        if n not in valid_names:
            raise ValueError(f"Unknown nurse '{name}' in fixed_assignments")
        if not (0 <= day < num_days):
            raise ValueError(f"Day index {day} out of range for {name}")
        cleaned[(n, day)] = shift.strip().upper()
    return cleaned
