import pandas as pd
from datetime import datetime, timedelta, time, date as dt_date
from typing import Any, Dict, Set, List, Union, Optional, Tuple, Iterable
from exceptions.custom_errors import FileContentError
from schemas.schedule.generate import Shifts
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


def normalise_date(input_date):
    """
    Convert input to a datetime.date object.
    Supports formats like:
      - 'Mon 2025-07-07', '2025/07/07', '20250707', etc.
    """
    if isinstance(input_date, dt_date) and not isinstance(input_date, datetime):
        return input_date
    elif isinstance(input_date, pd.Timestamp):
        return input_date.date()
    elif isinstance(input_date, datetime):
        return input_date.date()
    elif isinstance(input_date, str):
        try:
            # pandas handles all common formats using dateutil.parser under the hood
            return pd.to_datetime(input_date, errors="raise").date()
        except Exception as e:
            raise ValueError(f"Could not parse date string '{input_date}': {e}")
    raise ValueError(f"Unsupported date type: {type(input_date)}")


def make_shift_index(shifts) -> Dict[str, int]:

    if isinstance(shifts, tuple) and len(shifts) == 1 and isinstance(shifts[0], list):
        shifts = shifts[0]

    id_to_idx = {}
    for idx, s in enumerate(shifts):
        sid = str(s.id).strip()
        if sid in id_to_idx:
            raise ValueError(f"Duplicate shift id: {sid}")
        id_to_idx[sid] = idx

    return id_to_idx


def get_mc_days(
    preferences_df: pd.DataFrame,
    prev_sched_df: pd.DataFrame,
    start_date: Union[pd.Timestamp, dt_date],
    nurse_names: List[str],
    active_days: int,
) -> Dict[str, Set[int]]:
    """
    Returns nurse_name -> set of medical leave (MC) day-indices (0..active_days-1), considering both previous schedule and current preferences.
    """
    date_start = normalise_date(start_date)
    mc_days: Dict[str, Set[int]] = {n: set() for n in nurse_names}

    if prev_sched_df is not None and not prev_sched_df.empty:
        for nurse, row in prev_sched_df.iterrows():
            nm = str(nurse).strip().upper()
            if nm not in mc_days:
                continue
            for col in prev_sched_df.columns:
                raw = prev_sched_df.at[nurse, col]
                # unwrap tuple if present
                val = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
                if str(val).strip().upper() == "MC":
                    # this date is before start_date → negative offset
                    col_date = normalise_date(col)
                    offset = (col_date - date_start).days
                    mc_days[nm].add(offset)

    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        if nm not in mc_days:
            continue
        for label, raw in row.items():
            val = raw[0] if isinstance(raw, tuple) else raw
            if str(val).strip().upper() == "MC":
                offset = compute_label_offset(label, date_start)
                if 0 <= offset < active_days:
                    mc_days[nm].add(offset)
    return mc_days


def get_al_days(
    preferences_df: pd.DataFrame,
    prev_sched_df: pd.DataFrame,
    start_date: Union[pd.Timestamp, dt_date],
    nurse_names: List[str],
    active_days: int,
) -> Dict[str, Set[int]]:
    """
    Returns nurse_name -> set of annual leave (AL) day-indices (0..active_days-1), considering both previous schedule and current preferences.
    """
    date_start = normalise_date(start_date)
    al_days: Dict[str, Set[int]] = {n: set() for n in nurse_names}

    if prev_sched_df is not None and not prev_sched_df.empty:
        for nurse, row in prev_sched_df.iterrows():
            nm = str(nurse).strip().upper()
            if nm not in al_days:
                continue
            for col in prev_sched_df.columns:
                raw = prev_sched_df.at[nurse, col]
                # unwrap tuple if present
                val = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
                if str(val).strip().upper() == "AL":
                    # this date is before start_date → negative offset
                    col_date = normalise_date(col)
                    offset = (col_date - date_start).days
                    al_days[nm].add(offset)

    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        if nm not in al_days:
            continue
        for label, raw in row.items():
            val = raw[0] if isinstance(raw, tuple) else raw
            if str(val).strip().upper() == "AL":
                offset = compute_label_offset(label, date_start)
                if 0 <= offset < active_days:
                    al_days[nm].add(offset)
    return al_days


def parse_prefs(raw_pref) -> Tuple[str, pd.Timestamp]:
    """
    Given a raw preference input (e.g. tuple(code, timestamp), string, or None),
    return a normalized tuple of (code, timestamp) where:

    * code is a string (upper-cased) or None
    * timestamp is a pd.Timestamp (NaT if no valid timestamp was present)

    If the input is not a tuple, it is treated as a code string and timestamp is set to NaT.
    If the input is a tuple but the second element is not a valid timestamp, it is set to NaT.
    """
    if isinstance(raw_pref, tuple) and len(raw_pref) == 2:
        code, ts = raw_pref
    else:
        code, ts = raw_pref, pd.NaT

    code = str(code).strip()
    ts = pd.Timestamp.min if pd.isna(ts) else ts
    return code, ts


def get_shift_preferences(
    preferences_df: pd.DataFrame,
    start_date: Union[pd.Timestamp, dt_date],
    nurse_ids: List[str],
    active_days: int,
    shift_labels: List[Shifts],
    no_work_labels: List[str],
) -> Dict[str, Dict[int, Tuple[int, pd.Timestamp]]]:
    """
    Returns nurse_id -> { day_index: (shift_index, timestamp) } for valid preferences.
    """
    date_start = normalise_date(start_date)
    shift_str_to_idx = {str(s.id): idx for idx, s in enumerate(shift_labels)}

    shift_prefs: Dict[str, Dict[int, Tuple[int, pd.Timestamp]]] = {
        str(n): {} for n in nurse_ids
    }

    for id, row in preferences_df.drop(columns=["name"], errors="ignore").iterrows():
        nid = str(id).strip()

        if nid not in shift_prefs:
            continue

        for label, raw in row.items():
            if pd.isna(raw):
                continue

            code, ts = parse_prefs(raw)  # expects (shift_id, timestamp)

            # skip empty or "no work" preferences
            if not code or code.upper() in (lbl.upper() for lbl in no_work_labels):
                continue

            # valid shift
            if code in shift_str_to_idx:
                offset = compute_label_offset(label, date_start)
                if 0 <= offset < active_days:
                    prev = shift_prefs[nid].get(offset)
                    if prev is None or ts < prev[1]:
                        shift_prefs[nid][offset] = (shift_str_to_idx[code], ts)
            else:
                raise FileContentError(
                    f"Invalid preference “{code}” for nurse id {nid} on {label} — "
                    f"expected one of {[s.id for s in shift_labels] + no_work_labels!r} or blank."
                )

    return shift_prefs


def filter_fixed_assignments_from_prefs(
    shift_preferences: Dict[str, Dict[int, Tuple[int, Any]]],
    prefs_by_nurse: Dict[str, Dict[int, Tuple[int, Any]]],
    no_work_labels: List[str],
    fixed_assignments: Optional[Dict[Tuple[str, int], str]],
) -> None:
    """
    Remove any (nurse, day) preference from both shift_preferences and prefs_by_nurse
    if that (nurse, day) is in fixed_assignments with a NO_WORK_LABEL (MC, EL).
    """
    if not fixed_assignments:
        return

    no_work_label_set: set[str] = set(
        no_work_labels
    )  # convert list to set for quick lookup

    for nurse, prefs in shift_preferences.items():
        to_drop = []
        for day, (p, _) in prefs.items():
            if fixed_assignments.get((nurse, day), "").upper() in no_work_label_set:
                to_drop.append(day)

        for day in to_drop:
            # remove from the master dict
            prefs.pop(day, None)
            # remove from per-nurse view as well
            if nurse in prefs_by_nurse:
                prefs_by_nurse[nurse].pop(day, None)


def extract_prefs_info(
    preferences_df,
    date_start,
    nurse_names,
    num_days,
    shift_labels,
    no_work_labels,
    training_by_nurse,
    fixed_assignments=None,
):
    """Extracts preferences information and each nurse's preferences from the preferences DataFrame."""
    shift_preferences = get_shift_preferences(
        preferences_df, date_start, nurse_names, num_days, shift_labels, no_work_labels
    )
    prefs_by_nurse = {n: shift_preferences.get(n, {}) for n in nurse_names}
    filter_fixed_assignments_from_prefs(
        shift_preferences, prefs_by_nurse, no_work_labels, fixed_assignments
    )
    filter_prefs_from_training_shifts(
        shift_preferences, prefs_by_nurse, training_by_nurse
    )
    return shift_preferences, prefs_by_nurse


def filter_prefs_from_training_shifts(
    shift_preferences: Dict[str, Dict[int, Tuple[int, Any]]],
    prefs_by_nurse: Dict[str, Dict[int, Tuple[int, Any]]],
    training_by_nurse: Optional[Dict[str, Dict[int, int]]] = None,
) -> None:
    """
    Remove any (nurse, day) preference from both shift_preferences and prefs_by_nurse
    if that (nurse, day) is in training_by_nurse.
    """
    if not training_by_nurse:
        return

    for nurse in prefs_by_nurse:
        prefs = prefs_by_nurse[nurse]
        training = training_by_nurse.get(nurse, {})
        to_drop = []
        for day, (pref_shift, _) in prefs.items():
            tr = training.get(day)
            if tr == -1 or tr == pref_shift:
                to_drop.append(day)

        # logging.info(f"To drop: {to_drop}")
        for day in to_drop:
            prefs.pop(day, None)
            shift_preferences[nurse].pop(day, None)
        # logging.info(prefs_by_nurse)
        # logging.info(shift_preferences)


def get_training_shifts(
    training_shifts_df: pd.DataFrame,
    start_date: Union[pd.Timestamp, dt_date],
    nurse_ids: List[str],  # now IDs, not names
    active_days: int,
    shift_labels: List[Shifts],
) -> Dict[str, Dict[int, int]]:
    """
    Returns nurse_id -> { day_index: shift_index } for training shifts.
    Uses `id` instead of nurse name for mapping.
    """
    date_start = normalise_date(start_date)
    shifts_str_to_idx = make_shift_index(shift_labels)
    training_shifts: Dict[str, Dict[int, int]] = {str(i): {} for i in nurse_ids}

    # print("[code]", training_shifts_df)

    # Normalize columns
    if "id" in training_shifts_df.columns:
        training_shifts_df["id"] = training_shifts_df["id"].astype(str).str.strip()
    if "date" in training_shifts_df.columns:
        training_shifts_df["date"] = pd.to_datetime(
            training_shifts_df["date"], errors="coerce"
        )
    if "shiftid" in training_shifts_df.columns:
        training_shifts_df["shiftid"] = (
            training_shifts_df["shiftid"].astype(str).str.strip().str.upper()
        )

    # Process each row
    for _, row in training_shifts_df.iterrows():
        nurse_id = row.get("id", "")
        if nurse_id not in training_shifts:
            continue  # skip if not in current schedule

        date_val = row.get("date")
        if pd.isna(date_val):
            continue

        offset = compute_label_offset(date_val, date_start)
        if not (0 <= offset < active_days):
            continue

        code = row.get("shiftid", "")

        if code in shifts_str_to_idx:
            training_shifts[nurse_id][offset] = shifts_str_to_idx[code]
        else:
            raise FileContentError(
                f"Invalid training shift “{code}” for nurse ID {nurse_id} on {date_val.date()} — "
                f"expected one of {shift_labels!r}."
            )

    return training_shifts


def filter_fixed_assignments_from_training_shifts(
    training_shifts: Dict[str, Dict[int, int]],
    training_by_nurse: Dict[str, Dict[int, int]],
    no_work_labels: List[str],
    fixed_assignments: Optional[Dict[Tuple[str, int], str]],
) -> None:
    """
    Remove any (nurse, day) training sessions from both training_shifts and training_by_nurse
    if that (nurse, day) is in fixed_assignments with a NO_WORK_LABEL (MC, EL).
    """
    if not fixed_assignments:
        return

    # REST and TR are valid assignments when there is full/partial day training
    # TR label for full day training
    no_work_label_set: set[str] = set(no_work_labels) - {"REST", "TR"}

    for nurse, prefs in training_shifts.items():
        to_drop: list[int] = []
        for day, tr in prefs.items():
            if fixed_assignments.get((nurse, day), "").upper() in no_work_label_set:
                to_drop.append(day)

        for day in to_drop:
            # remove from the master dict
            prefs.pop(day, None)
            # remove from per-nurse view as well
            if nurse in training_by_nurse:
                training_by_nurse[nurse].pop(day, None)


def extract_training_shifts_info(
    training_shifts_df,
    date_start,
    nurse_names,
    num_days,
    shift_labels,
    no_work_labels,
    fixed_assignments=None,
):
    """Extracts training shifts information from the training_shifts DataFrame."""
    training_shifts = get_training_shifts(
        training_shifts_df, date_start, nurse_names, num_days, shift_labels
    )
    training_by_nurse = {n: training_shifts.get(n, {}) for n in nurse_names}
    filter_fixed_assignments_from_training_shifts(
        training_shifts, training_by_nurse, no_work_labels, fixed_assignments
    )

    return training_shifts, training_by_nurse


def get_el_days(fixed_assignments, nurse_names):
    """Extract EL days from fixed assignments."""
    el_days = {n: set() for n in nurse_names}
    for (nurse, d), label in fixed_assignments.items():
        if label.strip().upper() == "EL":
            el_days[nurse].add(d)
    return el_days


def extract_leave_days(
    preferences_df, prev_sched_df, nurse_names, start_date, num_days, fixed_assignments
):
    """
    Extracts leave information from profiles and preferences DataFrames, considering previous schedule and fixed assignments, if any.

    Returns mc_sets, al_sets, el_sets
    """
    mc_days = get_mc_days(
        preferences_df, prev_sched_df, start_date, nurse_names, num_days
    )
    al_days = get_al_days(
        preferences_df, prev_sched_df, start_date, nurse_names, num_days
    )

    mc_sets = {n: mc_days.get(n, set()) for n in nurse_names}
    al_sets = {n: al_days.get(n, set()) for n in nurse_names}
    el_sets = get_el_days(fixed_assignments, nurse_names)

    return mc_sets, al_sets, el_sets


def get_days_with_el(el_sets):
    """Returns a set of all day-indices (0 ... active_days-1) with EL assignments."""
    return {d for days in el_sets.values() for d in days}


def normalize_fixed_assignments(
    fixed: Optional[Dict[Tuple[str, int], str]], valid_names: Set[str], num_days: int
) -> Dict[Tuple[str, int], str]:
    """Upper-cases names/shifts, strips whitespace, and validates keys/indices."""
    if fixed is None:
        return {}
    cleaned: Dict[Tuple[str, int], str] = {}
    for (name, day), shift in fixed.items():
        n = name.strip().upper()
        if n not in valid_names:
            raise ValueError(f"Unknown nurse '{name}' in fixed_assignments")
        if not (0 <= day < num_days):
            raise ValueError(f"Day index {day} out of range for {name}")
        cleaned[(n, day)] = shift.strip().upper()
    return cleaned


def make_weekend_pairs(num_days, date_start):
    """Generates a list of pairs of weekend days."""
    weekend_pairs = []
    for i in range(num_days - 1):
        if (date_start + timedelta(days=i)).weekday() == 5:  # Saturday
            if i + 7 < num_days:
                weekend_pairs.append((i, i + 7))
            if i + 8 < num_days:  # Sunday
                weekend_pairs.append((i + 1, i + 8))
    return weekend_pairs


def shift_duration_minutes(start: time, end: time) -> int:
    """Returns shift duration in minutes."""
    today = datetime.today().date()
    dt_start = datetime.combine(today, start)
    dt_end = datetime.combine(today, end)
    if dt_end <= dt_start:
        dt_end += timedelta(days=1)
    return int((dt_end - dt_start).total_seconds() // 60)
