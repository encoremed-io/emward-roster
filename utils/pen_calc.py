from datetime import timedelta, date as dt_date
from collections import defaultdict
import pandas as pd
import numpy as np
import json

with open('config/constants.json', 'r') as f:
    constants = json.load(f)

SHIFT_LABELS = constants["SHIFT_LABELS"]
SHIFT_HOURS = constants["SHIFT_HOURS"]
AVG_HOURS = constants["AVG_HOURS"]
DAYS_PER_WEEK = constants["DAYS_PER_WEEK"]
MIN_NURSES_PER_SHIFT = constants["MIN_NURSES_PER_SHIFT"]
MIN_SENIORS_PER_SHIFT = constants["MIN_SENIORS_PER_SHIFT"]
MAX_WEEKLY_HOURS = constants["MAX_WEEKLY_HOURS"]
MAX_MC_DAYS_PER_WEEK = constants["MAX_MC_DAYS_PER_WEEK"]
MIN_ACCEPTABLE_WEEKLY_HOURS = constants["MIN_ACCEPTABLE_WEEKLY_HOURS"]
PREFERRED_WEEKLY_HOURS = constants["PREFERRED_WEEKLY_HOURS"]
PREF_HOURS_PENALTY = constants["PREF_HOURS_PENALTY"]
AM_COVERAGE_MIN_PERCENT = constants["AM_COVERAGE_MIN_PERCENT"]
AM_COVERAGE_PENALTIES = constants["AM_COVERAGE_PENALTIES"]
PREF_MISS_PENALTY = constants["PREF_MISS_PENALTY"]
FAIRNESS_GAP_PENALTY = constants["FAIRNESS_GAP_PENALTY"]
FAIRNESS_GAP_THRESHOLD = constants["FAIRNESS_GAP_THRESHOLD"]

HARD_CONSTRAINT_PENALTY = 1000      # penalty for breaking a hard constraint in CP-SAT


def compute_total_penalty(assignment: np.ndarray,
                          profiles_df: pd.DataFrame,
                          preferences_df: pd.DataFrame,
                          start_date: pd.Timestamp | dt_date,
                          fixed_assignments,
                          active_days: int) -> int:
    """
    Compute the total penalty for a given assignment.
    """

    if isinstance(start_date, pd.Timestamp):
        date_start : dt_date = start_date.date()
    else:
        date_start = start_date

    # Prepare name/index mappings
    nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
    shift_str_to_idx = {'AM': 0, 'PM': 1, 'NIGHT': 2}
    senior_set = {
        str(row["Name"]).strip().upper()
        for _, row in profiles_df.iterrows()
        if row.get("Title", "").upper() == "SENIOR"
    }

    el_days = defaultdict(set)
    for (nurse, d), label in fixed_assignments.items():
        if label.upper() == "EL":
            el_days[nurse].add(d)

    # Build preferences and MC-day sets
    shift_prefs = {n: {} for n in nurse_names}
    mc_days = {n: set() for n in nurse_names}
    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        for label, val in row.items():
            if isinstance(label, pd.Timestamp):
                d = label.date()
            elif isinstance(label, dt_date):
                d = label
            else:
                d = pd.to_datetime(str(label)).date()

            offset = (d - date_start).days
            if pd.notna(val) and 0 <= offset < active_days:
                v = str(val).strip().upper()
                if v == 'MC':
                    mc_days[nm].add(offset)
                elif v in shift_str_to_idx:
                    shift_prefs[nm][offset] = shift_str_to_idx[v]

    total_penalty = 0
    N, D, S = assignment.shape

    # ------------------------
    # PHASE 1: HIGH PRIORITY
    # ------------------------

    # 1) If nurse have >1 shift per day, big penalty
    for i, nurse in enumerate(nurse_names):
        for d in range(active_days):
            assigned_count = int(assignment[i, d, :].sum())
            if assigned_count > 1:
                extras = assigned_count - 1
                total_penalty += extras * HARD_CONSTRAINT_PENALTY

    # 2) If assigned shift on MC, big penalty
    for i, nurse in enumerate(nurse_names):
        for d in mc_days[nurse] & set(range(active_days)):
            if assignment[i, d, :].sum() > 0:
                total_penalty += HARD_CONSTRAINT_PENALTY

    # 3) If a day has <4 nurses or <1 senior nurse per shift, big penalty
    for d in range(active_days):
        for s in range(S):
            on_shift = assignment[:, d, s]
            total_on_shift = int(on_shift.sum())
            if total_on_shift < MIN_NURSES_PER_SHIFT:
                total_penalty += (MIN_NURSES_PER_SHIFT - total_on_shift) * HARD_CONSTRAINT_PENALTY

            senior_count = sum(
                1
                for i, nurse in enumerate(nurse_names)
                if on_shift[i] == 1 and nurse in senior_set
            )
            if total_on_shift > 0 and senior_count < MIN_SENIORS_PER_SHIFT:
                total_penalty += HARD_CONSTRAINT_PENALTY

    # 4) Weekly hours soft penalty; If >42 hours per week, big penalty
    for i, nurse in enumerate(nurse_names):
        for w in range((active_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK):
            days = list(range(w*DAYS_PER_WEEK, min((w+1)*DAYS_PER_WEEK, active_days)))
            hours = sum(assignment[i, d, s] * SHIFT_HOURS[s]
                        for d in days for s in range(S))
            mc_cnt = sum(1 for d in days if d in mc_days[nurse])
            el_cnt = sum(1 for d in days if d in el_days[nurse])
            adj = (mc_cnt + el_cnt) * int(AVG_HOURS)
            eff_max = max(0, MAX_WEEKLY_HOURS - adj)
            eff_pref = max(0, PREFERRED_WEEKLY_HOURS - adj)
            eff_min = max(0, MIN_ACCEPTABLE_WEEKLY_HOURS - adj)
            if hours > eff_max:
                total_penalty += HARD_CONSTRAINT_PENALTY
            if hours < eff_min:
                total_penalty += HARD_CONSTRAINT_PENALTY
            if eff_min <= hours < eff_pref:
                total_penalty += PREF_HOURS_PENALTY


    # 5) AM coverage penalty
    for d in range(active_days):
        total_shifts = assignment[:, d, :].sum()
        if total_shifts == 0:
            continue
        am_count = assignment[:, d, 0].sum()
        pm_count = assignment[:, d, 1].sum()
        night_count = assignment[:, d, 2].sum()
        pct = 100 * am_count / total_shifts
        if pct < AM_COVERAGE_MIN_PERCENT - 20:
            total_penalty += AM_COVERAGE_PENALTIES[2]
        elif pct < AM_COVERAGE_MIN_PERCENT - 10:
            total_penalty += AM_COVERAGE_PENALTIES[1]
        elif pct < AM_COVERAGE_MIN_PERCENT:
            total_penalty += AM_COVERAGE_PENALTIES[0]
        elif am_count <= pm_count or am_count <= night_count:
            total_penalty += HARD_CONSTRAINT_PENALTY

    # 6) If MC >2 per week or >2 consecutive MC days, big penalty
    for nurse in nurse_names:
        days = sorted(mc_days[nurse] & set(range(active_days)))
        if len(days) > MAX_MC_DAYS_PER_WEEK:
            total_penalty += (len(days) - MAX_MC_DAYS_PER_WEEK) * HARD_CONSTRAINT_PENALTY

        consec = 1
        for idx in range(1, len(days)):
            if days[idx] == days[idx - 1] + 1:
                consec += 1
                if consec > 2:
                    total_penalty += HARD_CONSTRAINT_PENALTY
                    break
            else:
                consec = 1

    # 7) If worked on certain weekend day of a week and assigned work on same day of next week, big penalty
    weekend_days = [
        (i, i + 1) for i in range(active_days - 1)
        if (date_start + timedelta(days = i)).weekday() == 5
    ]

    for i, nurse in enumerate(nurse_names):
        for d1, d2 in weekend_days:
            for day in (d1, d2):
                if day + 7 < D:
                    worked_this = int(assignment[i, day, :].sum()) > 0
                    worked_next = int(assignment[i, day + 7, :].sum()) > 0
                    if worked_this and worked_next:
                        total_penalty += HARD_CONSTRAINT_PENALTY

    # ------------------------
    # PHASE 2: LOW PRIORITY
    # ------------------------

    # 1) Preference-miss and fairness gap penalties
    pct_sat = []
    for i, nurse in enumerate(nurse_names):
        prefs = shift_prefs[nurse]
        met = 0
        if not prefs:
            continue
        for d, s in prefs.items():
            if assignment[i, d, s] == 1:
                met += 1
            else:
                total_penalty += PREF_MISS_PENALTY
        if prefs:
            pct_sat.append(100 * met / len(prefs))
    if pct_sat:
        gap = max(pct_sat) - min(pct_sat)
        if gap >= FAIRNESS_GAP_THRESHOLD:
            over_gap = gap - FAIRNESS_GAP_THRESHOLD
            total_penalty += over_gap * FAIRNESS_GAP_PENALTY

    return int(total_penalty)


def compute_penalty_per_day(assignment: np.ndarray,
                            profiles_df: pd.DataFrame,
                            preferences_df: pd.DataFrame,
                            start_date: pd.Timestamp | dt_date,
                            day_idx: int) -> int:
    """
    Compute penalty only for the given day_idx.
    """

    if isinstance(start_date, pd.Timestamp):
        date_start : dt_date = start_date.date()
    else:
        date_start = start_date

    nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
    shift_str_to_idx = {'AM': 0, 'PM': 1, 'NIGHT': 2}
    senior_set = {
        str(row["Name"]).strip().upper()
        for _, row in profiles_df.iterrows()
        if row.get("Title", "").upper() == "SENIOR"
    }

    N, D, S = assignment.shape
    if day_idx < 0 or day_idx >= D:
        raise ValueError("day_idx out of range")

    # Build preferences for nurses for this day only
    shift_prefs = {n: {} for n in nurse_names}
    mc_for_this_day = {n: set() for n in nurse_names}
    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        for label, val in row.items():
            if isinstance(label, pd.Timestamp):
                d = label.date()
            elif isinstance(label, dt_date):
                d = label
            else:
                d = pd.to_datetime(str(label)).date()
            offset = (d - date_start).days
            if offset == day_idx and pd.notna(val):
                v = str(val).strip().upper()
                if v == "MC":
                    mc_for_this_day[nm].add(offset)
                if v in shift_str_to_idx:
                    shift_prefs[nm][offset] = shift_str_to_idx[v]

    total_penalty = 0

    # 1) If nurse has >1 shift per day, big penalty
    for i, nurse in enumerate(nurse_names):
        assigned_count = int(assignment[i, day_idx, :].sum())
        if assigned_count > 1:
            extras = assigned_count - 1
            total_penalty += extras * HARD_CONSTRAINT_PENALTY

    # 2) If assigned work on MC, big penalty
    for i, nurse in enumerate(nurse_names):
        if day_idx in mc_for_this_day[nurse] and assignment[i, day_idx, :].sum() > 0:
            total_penalty += HARD_CONSTRAINT_PENALTY

    # 3) If a day has <4 nurses per shift or <1 senior nurse per shift, big penalty
    for s in range(S):
        on_shift = assignment[:, day_idx, s]
        total_on_shift = int(on_shift.sum())
        if total_on_shift < MIN_NURSES_PER_SHIFT:
            total_penalty += (MIN_NURSES_PER_SHIFT - total_on_shift) * HARD_CONSTRAINT_PENALTY

        senior_count = sum(
            1
            for i, nurse in enumerate(nurse_names)
            if on_shift[i] == 1 and nurse in senior_set
        )
        if total_on_shift > 0 and senior_count < MIN_SENIORS_PER_SHIFT:
            total_penalty += HARD_CONSTRAINT_PENALTY

    # 4) AM coverage penalty for this day only
    total_shifts = assignment[:, day_idx, :].sum()
    if total_shifts > 0:
        am_count = assignment[:, day_idx, 0].sum()
        pm_count = assignment[:, day_idx, 1].sum()
        night_count = assignment[:, day_idx, 2].sum()
        pct = 100 * am_count / total_shifts
        if pct < AM_COVERAGE_MIN_PERCENT - 20:
            total_penalty += AM_COVERAGE_PENALTIES[2]
        elif pct < AM_COVERAGE_MIN_PERCENT - 10:
            total_penalty += AM_COVERAGE_PENALTIES[1]
        elif pct < AM_COVERAGE_MIN_PERCENT:
            total_penalty += AM_COVERAGE_PENALTIES[0]
        elif am_count <= pm_count or am_count <= night_count:
            total_penalty += HARD_CONSTRAINT_PENALTY

    # 5) Preference-miss penalty for this day only
    for i, nurse in enumerate(nurse_names):
        prefs = shift_prefs[nurse]
        # prefs only for this day (day_idx), so check if nurse has preference this day
        if day_idx in prefs:
            preferred_shift = prefs[day_idx]
            if assignment[i, day_idx, preferred_shift] != 1:
                total_penalty += PREF_MISS_PENALTY

    return int(total_penalty)
