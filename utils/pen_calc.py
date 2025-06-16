from datetime import timedelta, date as dt_date
import pandas as pd
from .shift_utils import *
from utils.constants import *   # Import all constants

HARD_CONSTRAINT_PENALTY = 1000

# High Priority Penalty Components (7 modules)
def hp_multiple_shifts(assignment, nurse_names, active_days):
    """Penalty 1: Multiple shifts per day per nurse."""
    penalty = 0
    for i in range(len(nurse_names)):
        for d in range(active_days):
            if assignment[i, d, :].sum() > 1:
                extras = assignment[i, d, :].sum() - 1
                penalty += extras * HARD_CONSTRAINT_PENALTY
    return penalty


def hp_mc_day_assignments(assignment, nurse_names, mc_days, active_days):
    """Penalty 2: Assignments on MC days."""
    penalty = 0
    for i, nurse in enumerate(nurse_names):
        valid_mc_days = mc_days.get(nurse, set()) & set(range(active_days))
        for d in valid_mc_days:
            if assignment[i, d, :].sum() > 0:
                penalty += HARD_CONSTRAINT_PENALTY
    return penalty


def hp_nurses_staffing_level(assignment, active_days):
    """Penalty 3: Minimum nurses per shift."""
    penalty = 0
    S = assignment.shape[2]
    
    for d in range(active_days):
        for s in range(S):
            total_on_shift = assignment[:, d, s].sum()
            if total_on_shift < MIN_NURSES_PER_SHIFT:
                penalty += (MIN_NURSES_PER_SHIFT - total_on_shift) * HARD_CONSTRAINT_PENALTY
            
    return penalty


def hp_senior_staffing_level(assignment, nurse_names, senior_set, active_days):
    """Penalty 4: Minimum seniors per AM shift."""
    penalty = 0
    S = assignment.shape[2]

    for d in range(active_days):
        for s in range(S):
            # check nurses working on that day > 0
            if assignment[:, d, s].sum() > 0:
                senior_count = sum(
                    assignment[i, d, s] == 1 and nurse_names[i] in senior_set
                    for i in range(len(nurse_names))
                )
                if senior_count < MIN_SENIORS_PER_SHIFT:
                    penalty += (MIN_SENIORS_PER_SHIFT - senior_count) * HARD_CONSTRAINT_PENALTY
    
    return penalty


def hp_weekly_hours(assignment, nurse_names, mc_days, el_days, active_days):
    """Penalty 5: Weekly hours constraints."""
    penalty = 0
    
    for i, nurse in enumerate(nurse_names):
        for w in range((active_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK):
            # Calculate days in week
            start_day = w * DAYS_PER_WEEK
            end_day = min((w + 1) * DAYS_PER_WEEK, active_days)
            days = list(range(start_day, end_day))
            
            # Calculate hours worked
            hours = sum(assignment[i, d, s] * SHIFT_HOURS[s]
                        for d in days for s in range(len(SHIFT_HOURS)))
            
            # Calculate adjustments
            mc_cnt = sum(1 for d in days if d in mc_days.get(nurse, set()))
            el_cnt = sum(1 for d in days if d in el_days.get(nurse, set()))
            adj = (mc_cnt + el_cnt) * int(AVG_HOURS)
            
            # Calculate effective limits
            eff_max = max(0, MAX_WEEKLY_HOURS - adj)
            eff_pref = max(0, PREFERRED_WEEKLY_HOURS - adj)
            eff_min = max(0, MIN_ACCEPTABLE_WEEKLY_HOURS - adj)
            
            # Apply penalties
            if hours > eff_max:
                penalty += HARD_CONSTRAINT_PENALTY
            if hours < eff_min:
                penalty += HARD_CONSTRAINT_PENALTY
            if eff_min <= hours < eff_pref:
                penalty += PREF_HOURS_PENALTY
    return penalty


def hp_am_coverage(assignment, active_days):
    """Penalty 6: AM shift coverage."""
    penalty = 0
    for d in range(active_days):
        total_shifts = assignment[:, d, :].sum()
        if total_shifts == 0:
            continue
            
        am_count = assignment[:, d, 0].sum()
        pm_count = assignment[:, d, 1].sum()
        night_count = assignment[:, d, 2].sum()
        
        # Calculate AM percentage
        pct = 100 * am_count / total_shifts
        
        # Apply penalties based on coverage levels
        if pct < AM_COVERAGE_MIN_PERCENT - 20:
            penalty += AM_COVERAGE_PENALTIES[2]
        elif pct < AM_COVERAGE_MIN_PERCENT - 10:
            penalty += AM_COVERAGE_PENALTIES[1]
        elif pct < AM_COVERAGE_MIN_PERCENT:
            penalty += AM_COVERAGE_PENALTIES[0]
        # Check if AM is less than or equal to other shifts
        elif am_count <= pm_count or am_count <= night_count:
            penalty += HARD_CONSTRAINT_PENALTY
    return penalty


def hp_am_senior_coverage(assignment, nurse_names, senior_names, active_days):
    """Penalty 7: AM senior shift coverage"""
    penalty = 0
    for d in range(active_days):
        total_am = 0
        senior_am = 0
        senior_pm = 0
        senior_night = 0

        for i, nurse in enumerate(nurse_names):
            if assignment[i, d, 0] == 1:  # AM shift
                total_am += 1
                if nurse in senior_names:
                    senior_am += 1
            elif assignment[i, d, 1] == 1:  # PM shift
                if nurse in senior_names:
                    senior_pm += 1
            elif assignment[i, d, 2] == 1:  # Night shift
                if nurse in senior_names:
                    senior_night += 1

        if total_am == 0:
            continue

        pct = 100 * senior_am / total_am

        # Apply penalties based on coverage levels
        if pct < AM_SENIOR_MIN_PERCENT - 20:
            penalty += AM_SENIOR_PENALTIES[2]
        elif pct < AM_SENIOR_MIN_PERCENT - 10:
            penalty += AM_SENIOR_PENALTIES[1]
        elif pct < AM_SENIOR_MIN_PERCENT:
            penalty += AM_SENIOR_PENALTIES[0]
        # Check if AM is less than or equal to other shifts
        elif senior_am <= senior_pm or senior_am <= senior_night:
            penalty += HARD_CONSTRAINT_PENALTY
    return penalty


def hp_mc_day_rules(nurse_names, mc_days, active_days):
    """Penalty 8: MC days rule violations."""
    penalty = 0
    
    for nurse in nurse_names:
        days = sorted(mc_days.get(nurse, set()) & set(range(active_days)))
        
        # Check MC days per week
        for w in range((active_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK):
            week_mc_days = [d for d in days if w*DAYS_PER_WEEK <= d < (w+1)*DAYS_PER_WEEK]
            if len(week_mc_days) > MAX_MC_DAYS_PER_WEEK:
                excess = len(week_mc_days) - MAX_MC_DAYS_PER_WEEK
                penalty += excess * HARD_CONSTRAINT_PENALTY
        
        # Check consecutive MC days
        consec = 1
        for idx in range(1, len(days)):
            if days[idx] == days[idx - 1] + 1:
                consec += 1
                if consec > 2:
                    penalty += HARD_CONSTRAINT_PENALTY
                    break
            else:
                consec = 1
    return penalty


def hp_weekend_consecutive_work(assignment, start_date, nurse_names, active_days):
    """Penalty 9: Consecutive weekend work."""
    penalty = 0
    
    # Find all weekend days (Saturday and Sunday)
    weekend_days = []
    for i in range(active_days - 1):
        if (start_date + timedelta(days=i)).weekday() == 5:  # Saturday
            if i + 1 < active_days:  # Ensure Sunday exists
                weekend_days.append((i, i+1))
    
    # Check for consecutive weekend work
    for i, nurse in enumerate(nurse_names):
        for sat, sun in weekend_days:
            # Check both Saturday and Sunday
            for day in (sat, sun):
                if day + 7 < active_days:
                    worked_this = assignment[i, day, :].sum() > 0
                    worked_next = assignment[i, day + 7, :].sum() > 0
                    if worked_this and worked_next:
                        penalty += HARD_CONSTRAINT_PENALTY
    return penalty


# High Priority Penalty Integration
def compute_high_priority_penalty(assignment, profiles_df, preferences_df, start_date, fixed_assignments, active_days):
    """Compute all 9 high priority penalties."""
    # Parse inputs
    start_date_dt = start_date.date() if isinstance(start_date, pd.Timestamp) else start_date
    nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
    
    # Build data structures
    shift_prefs, mc_days = get_shift_prefs_and_mc_days(preferences_df, profiles_df, start_date_dt, active_days)
    senior_set = get_senior_set(profiles_df)
    el_days = get_el_days(fixed_assignments)
    
    # Compute penalties
    total_penalty = 0
    total_penalty += hp_multiple_shifts(assignment, nurse_names, active_days)
    total_penalty += hp_mc_day_assignments(assignment, nurse_names, mc_days, active_days)
    total_penalty += hp_nurses_staffing_level(assignment, active_days)
    total_penalty += hp_senior_staffing_level(assignment, nurse_names, senior_set, active_days)
    total_penalty += hp_weekly_hours(assignment, nurse_names, mc_days, el_days, active_days)
    total_penalty += hp_am_coverage(assignment, active_days)
    total_penalty += hp_am_senior_coverage(assignment, nurse_names, senior_set, active_days)
    total_penalty += hp_mc_day_rules(nurse_names, mc_days, active_days)
    total_penalty += hp_weekend_consecutive_work(assignment, start_date_dt, nurse_names, active_days)
    
    return int(total_penalty)


# Low Priority Penalty
def compute_low_priority_penalty(assignment, profiles_df, preferences_df, start_date, active_days):
    """Compute low priority penalties."""
    # Parse inputs
    start_date_dt = start_date.date() if isinstance(start_date, pd.Timestamp) else start_date
    nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
    shift_str_to_idx = {'AM': 0, 'PM': 1, 'NIGHT': 2}
    
    # Build preferences
    shift_prefs = {n: {} for n in nurse_names}
    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        for label, val in row.items():
            # Handle date format
            if isinstance(label, pd.Timestamp):
                d = label.date()
            elif isinstance(label, dt_date):
                d = label
            else:
                d = pd.to_datetime(str(label)).date()
                
            offset = (d - start_date_dt).days
            if 0 <= offset < active_days and pd.notna(val):
                v = str(val).strip().upper()
                if v in shift_str_to_idx:
                    shift_prefs[nm][offset] = shift_str_to_idx[v]
    
    total_penalty = 0
    pct_sat = []
    
    # 1) Preference-miss penalties
    for i, nurse in enumerate(nurse_names):
        prefs = shift_prefs.get(nurse, {})
        met = 0
        if not prefs:
            continue
            
        for d, s in prefs.items():
            if d < assignment.shape[1] and assignment[i, d, s] == 1:
                met += 1
            else:
                total_penalty += PREF_MISS_PENALTY
        
        pct_sat.append(100 * met / len(prefs))
    
    # 2) Fairness gap penalty
    if pct_sat:
        gap = max(pct_sat) - min(pct_sat)
        if gap >= FAIRNESS_GAP_THRESHOLD:
            total_penalty += (gap - FAIRNESS_GAP_THRESHOLD) * FAIRNESS_GAP_PENALTY
    
    return int(total_penalty)


def compute_total_penalty(assignment, profiles_df, preferences_df, start_date, fixed_assignments, active_days):
    high = compute_high_priority_penalty(
        assignment,
        profiles_df,
        preferences_df,
        start_date,
        fixed_assignments,
        active_days
    )

    low = compute_low_priority_penalty(
        assignment,
        profiles_df,
        preferences_df,
        start_date,
        active_days
    )

    return high + low