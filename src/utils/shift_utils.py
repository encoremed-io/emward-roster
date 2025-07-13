import pandas as pd
from datetime import datetime, timedelta, time, date as dt_date
from typing import Any, Dict, Set, List, Union, Optional, Tuple
from exceptions.custom_errors import FileContentError
import logging

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
    shift_labels: List[str],
    no_work_labels: List[str]
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
                if code == "" or code in (lbl.upper() for lbl in no_work_labels):
                    continue
                elif code in shift_str_to_idx:
                    offset = compute_label_offset(label, date_start)
                    if 0 <= offset < active_days:
                        shift_prefs[nm][offset] = shift_str_to_idx[code]
                else:
                    raise FileContentError(
                        f"Invalid preference “{code}” for nurse {nm} on {label} — "
                        f"expected one of {shift_labels + no_work_labels!r} or blank."
                    )
    return shift_prefs


def filter_fixed_assignments_from_prefs(
    shift_preferences: Dict[str, Dict[int, int]],
    prefs_by_nurse:    Dict[str, Dict[int, int]],
    no_work_labels:    List[str],
    fixed_assignments: Optional[Dict[Tuple[str, int], str]]
) -> None:
    """
    Remove any (nurse, day) preference from both shift_preferences and prefs_by_nurse
    if that (nurse, day) is in fixed_assignments with a NO_WORK_LABEL (REST, MC, EL, AL).
    """
    if not fixed_assignments:
        return
    
    no_work_label_set: set[str] = set(no_work_labels)     # convert list to set for quick lookup

    for nurse, prefs in shift_preferences.items():
        # build list of days to drop for this nurse
        to_drop = [
            day
            for day in prefs
            if fixed_assignments.get((nurse, day), "").upper() in no_work_label_set
        ]

        for day in to_drop:
            # remove from the master dict
            prefs.pop(day, None)
            # remove from per-nurse view as well
            if nurse in prefs_by_nurse:
                prefs_by_nurse[nurse].pop(day, None)


def extract_prefs_info(preferences_df, profiles_df, date_start, nurse_names, num_days, shift_labels, no_work_labels, training_by_nurse, fixed_assignments=None):
    """ Extracts preferences information and each nurse's preferences from the preferences DataFrame. """
    shift_preferences = get_shift_preferences(preferences_df, profiles_df, date_start, num_days, shift_labels, no_work_labels)
    prefs_by_nurse = {n: shift_preferences.get(n, {}) for n in nurse_names}
    filter_fixed_assignments_from_prefs(shift_preferences, prefs_by_nurse, no_work_labels, fixed_assignments)
    filter_prefs_from_training_shifts(shift_preferences, prefs_by_nurse, training_by_nurse)
    return shift_preferences, prefs_by_nurse


def filter_prefs_from_training_shifts(
    shift_preferences: Dict[str, Dict[int, int]],
    prefs_by_nurse: Dict[str, Dict[int, int]],
    training_by_nurse: Optional[Dict[str, Dict[int, int]]] = None
) -> None:
    if not training_by_nurse:
        return

    for nurse in prefs_by_nurse:
        prefs = prefs_by_nurse[nurse]
        training = training_by_nurse.get(nurse, {})
        # find preference equal to training shift
        to_drop = [
            day for day, pref_shift in prefs.items()
            if training.get(day) == pref_shift
        ]
        # logging.info(f"To drop: {to_drop}")
        for day in to_drop:
            prefs.pop(day, None)
            shift_preferences[nurse].pop(day, None)
        # logging.info(prefs_by_nurse)
        # logging.info(shift_preferences)


def get_training_shifts(
    training_shifts_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    start_date: Union[pd.Timestamp, dt_date],
    active_days: int,
    shift_labels: List[str],
) -> Dict[str, Dict[int, int]]:
    """
    Returns nurse_name -> { day_index: shift_index } for non-MC preferences.
    """
    date_start = start_date.date() if isinstance(start_date, pd.Timestamp) else start_date
    nurse_names = [n.strip().upper() for n in profiles_df['Name']]
    shifts_str_to_idx = {label.upper(): idx for idx, label in enumerate(shift_labels)}
    training_shifts: Dict[str, Dict[int, int]] = {n: {} for n in nurse_names}

    for nurse, row in training_shifts_df.iterrows():
        nm = str(nurse).strip().upper()
        if nm not in training_shifts:
            continue
        for label, val in row.items():
            if pd.notna(val):
                code = str(val).strip().upper()
                if code == "":
                    continue
                elif code in shifts_str_to_idx:
                    offset = compute_label_offset(label, date_start)
                    if 0 <= offset < active_days:
                        training_shifts[nm][offset] = shifts_str_to_idx[code]
                else:
                    raise FileContentError(
                        f"Invalid training shift “{code}” for nurse {nm} on {label} — "
                        f"expected one of {shift_labels!r} or blank."
                    )
    return training_shifts


def filter_fixed_assignments_from_training_shifts(
    training_shifts: Dict[str, Dict[int, int]],
    training_by_nurse:    Dict[str, Dict[int, int]],
    no_work_labels:    List[str],
    fixed_assignments: Optional[Dict[Tuple[str, int], str]]
) -> None:
    """
    Remove any (nurse, day) training sessions from both training_shifts and training_by_nurse
    if that (nurse, day) is in fixed_assignments with a NO_WORK_LABEL (REST, MC, EL, AL).
    """
    if not fixed_assignments:
        return
    
    no_work_label_set: set[str] = set(no_work_labels)     # convert list to set for quick lookup

    for nurse, prefs in training_shifts.items():
        # build list of days to drop for this nurse
        to_drop = [
            day
            for day in prefs
            if fixed_assignments.get((nurse, day), "").upper() in no_work_label_set
        ]

        for day in to_drop:
            # remove from the master dict
            prefs.pop(day, None)
            # remove from per-nurse view as well
            if nurse in training_by_nurse:
                training_by_nurse[nurse].pop(day, None)


def extract_training_shifts_info(training_shifts_df, profiles_df, date_start, nurse_names, num_days, shift_labels, no_work_labels, fixed_assignments=None):
    """ Extracts training shifts information from the training_shifts DataFrame. """
    training_shifts = get_training_shifts(training_shifts_df, profiles_df, date_start, num_days, shift_labels)
    training_by_nurse = {n: training_shifts.get(n, {}) for n in nurse_names}
    filter_fixed_assignments_from_training_shifts(training_shifts, training_by_nurse, no_work_labels, fixed_assignments)
    return training_shifts, training_by_nurse


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
