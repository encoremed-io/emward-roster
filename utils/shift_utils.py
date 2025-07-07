import pandas as pd
from datetime import datetime, timedelta, time, date as dt_date
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


def make_shift_index(shift_labels: List[str]) -> Dict[str, int]:
    """Map shift code → int index for your decision variables."""
    return {label.upper(): i for i, label in enumerate(shift_labels)}


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


def get_al_days(
    preferences_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    start_date: Union[pd.Timestamp, dt_date],
    active_days: int
) -> Dict[str, Set[int]]:
    """
    Returns nurse_name -> set of AL day-indices (0..active_days-1).
    """
    date_start = start_date.date() if isinstance(start_date, pd.Timestamp) else start_date
    nurse_names = [n.strip().upper() for n in profiles_df['Name']]
    al_days: Dict[str, Set[int]] = {n: set() for n in nurse_names}

    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        if nm not in al_days:
            continue
        for label, val in row.items():
            if pd.notna(val) and str(val).strip().upper() == 'AL':
                offset = compute_label_offset(label, date_start)
                if 0 <= offset < active_days:
                    al_days[nm].add(offset)
    return al_days


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


def extract_prefs_info(preferences_df, profiles_df, date_start, nurse_names, num_days, shift_labels):
    """ Extracts preferences information and each nurse's preferences from the preferences DataFrame. """
    shift_preferences = get_shift_preferences(preferences_df, profiles_df, date_start, num_days, shift_labels)
    prefs_by_nurse = {n: shift_preferences.get(n, {}) for n in nurse_names}
    return shift_preferences, prefs_by_nurse


def get_el_days(fixed_assignments, nurse_names):
    """Extract EL days from fixed assignments."""
    el_days = {n: set() for n in nurse_names}
    for (nurse, d), label in fixed_assignments.items():
        if label.strip().upper() == "EL":
            el_days[nurse].add(d)
    return el_days


def extract_leave_days(profiles_df, preferences_df, nurse_names,start_date, num_days, fixed_assignments):
    """
    Extracts leave information from profiles and preferences DataFrames.
    
    Returns mc_sets, al_sets, el_sets
    """
    mc_days = get_mc_days(preferences_df, profiles_df, start_date, num_days)
    al_days = get_al_days(preferences_df, profiles_df, start_date, num_days)

    mc_sets = {n: mc_days.get(n, set()) for n in nurse_names}
    al_sets = {n: al_days.get(n, set()) for n in nurse_names}
    el_sets = get_el_days(fixed_assignments, nurse_names)

    return mc_sets, al_sets, el_sets


def get_days_with_el(el_sets):
    """Returns a set of all day-indices (0 ... active_days-1) with EL assignments."""
    return {d for days in el_sets.values() for d in days}


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


def make_weekend_pairs(num_days, date_start):
    weekend_pairs = []
    for i in range(num_days - 1):
        if (date_start + timedelta(days=i)).weekday() == 5:  # Saturday
            if i + 7 < num_days:
                weekend_pairs.append((i, i + 7))
            if i + 8 < num_days:                             # Sunday
                weekend_pairs.append((i + 1, i + 8))
    return weekend_pairs


def shift_duration_minutes(start: time, end: time) -> int:
    """ Returns shift duration in minutes. """
    today = datetime.today().date()
    dt_start = datetime.combine(today, start)
    dt_end = datetime.combine(today, end)
    if dt_end <= dt_start:
        dt_end += timedelta(days=1)
    return int((dt_end - dt_start).total_seconds() // 60)
