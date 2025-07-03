import pandas as pd
from ortools.sat.python import cp_model
from datetime import timedelta, date as dt_date
from typing import Optional, Dict, Tuple, Set
from pathlib import Path
import logging
from utils.constants import *       # import all constants
from utils.validate import *
from utils.shift_utils import *
from exceptions.custom_errors import *

LOG_PATH = Path(__file__).parent / "schedule_run.log"

logging.basicConfig(
    filename=LOG_PATH,
    filemode='w',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)


# == ANALYSE INFEASIBILITY ==
def analyze_infeasibility(
    nurse_names, 
    senior_names, 
    mc_days, 
    el_days_per_nurse, 
    num_days, 
    shift_types,
    min_nurses_per_shift,
    min_seniors_per_shift,
    min_weekly_hours,
    days_per_week,
    shift_hours
):
    """Diagnoses scheduling conflicts by verifying hard constraint feasibility"""
    reasons = []
    
    # Validate inputs
    if min_nurses_per_shift <= 0 or min_seniors_per_shift <= 0:
        reasons.append("❌ Must have at least 1 nurse and 1 senior in input.\n")
        return reasons

    # Constants for calculations
    max_shift_hours = max(shift_hours)
    avg_shift_hours = sum(shift_hours) / len(shift_hours)
    two_largest_hours = sum(sorted(shift_hours, reverse=True)[:2])
    
    # Core staffing constraints
    total_shifts_required = min_nurses_per_shift * num_days * shift_types
    senior_shifts_required = min_seniors_per_shift * num_days * shift_types
    
    # 1. Global staffing capacity
    total_available_shifts = 0
    for name in nurse_names:
        mc_days_count = len(mc_days.get(name, set()))
        el_days_count = len(el_days_per_nurse.get(name, set()))
        total_available_shifts += (num_days - mc_days_count - el_days_count) * 1 + el_days_count  # Count 1 shift/day when available
        # EL days add potential for double shifts from other nurses
        
    if total_available_shifts < total_shifts_required:
        deficit = total_shifts_required - total_available_shifts
        reasons.append(f"❌ Global staffing shortage: {total_available_shifts} available shifts "
                      f"< {total_shifts_required} required ({deficit} deficit)\n")

    # 2. Senior coverage
    senior_available_shifts = 0
    for name in senior_names:
        mc_days_count = len(mc_days.get(name, set()))
        el_days_count = len(el_days_per_nurse.get(name, set()))
        senior_available_shifts += (num_days - mc_days_count - el_days_count) * 1 + el_days_count
        
    if senior_available_shifts < senior_shifts_required:
        deficit = senior_shifts_required - senior_available_shifts
        reasons.append(f"❌ Senior coverage gap: {len(senior_names)} seniors can only provide "
                      f"{senior_available_shifts} shifts < {senior_shifts_required} required ({deficit} deficit)\n")

    # 3. Daily coverage feasibility
    critical_days = []
    for day in range(num_days):
        # Track availability
        available_nurses = set()
        senior_avail = set()
        el_nurses_on_day = set()
        
        # Identify any EL nurse on this day (for double-shift allowance)
        any_el_day = any(day in el_days_per_nurse.get(name, set()) for name in nurse_names)
        
        # Calculate availability
        for name in nurse_names:
            if day in mc_days.get(name, set()):
                continue  # Skip MC nurses
                
            if day in el_days_per_nurse.get(name, set()):
                el_nurses_on_day.add(name)  # Track EL nurses
                continue  # EL nurses don't work
                
            available_nurses.add(name)
            if name in senior_names:
                senior_avail.add(name)
        
        # Calculate effective capacity for regular nurses
        if any_el_day:
            # EL day allows other nurses to work double shifts
            max_assignable_nurses = len(available_nurses) * 2
            max_assignable_seniors = len(senior_avail) * 2
        else:
            # Non-EL day - each nurse works max 1 shift
            max_assignable_nurses = len(available_nurses)
            max_assignable_seniors = len(senior_avail)
        
        # Calculate total shifts needed per day
        total_shifts_needed = min_nurses_per_shift * shift_types
        total_senior_shifts_needed = min_seniors_per_shift * shift_types
        
        # Check if we can meet requirements
        if max_assignable_nurses < total_shifts_needed:
            critical_days.append(
                f"Day {day+1}: {max_assignable_nurses} nurse slots < "
                f"{total_shifts_needed} required\n"
            )
            
        if max_assignable_seniors < total_senior_shifts_needed:
            critical_days.append(
                f"Day {day+1}: {max_assignable_seniors} senior slots < "
                f"{total_senior_shifts_needed} required\n"
            )
    
    # Format critical days output
    if critical_days:
        reasons.append("🚨 Critical daily staffing shortages:\n")
        reasons.extend([f"   • {day}\n" for day in critical_days[:20]])  # Show first 20 issues
        if len(critical_days) > 20:
            reasons.append(f"   • ... and {len(critical_days) - 20} more\n")

    # 4. Weekly hours feasibility
    num_weeks = (num_days + days_per_week - 1) // days_per_week
    hourly_issues = {week: [] for week in range(num_weeks)}
    
    for name in nurse_names:
        weekly_mc = [0] * num_weeks
        weekly_el = [0] * num_weeks
        
        # Precompute leaves per week
        for day in range(num_days):
            week_idx = day // days_per_week
            if day in mc_days.get(name, set()):
                weekly_mc[week_idx] += 1
            if day in el_days_per_nurse.get(name, set()):
                weekly_el[week_idx] += 1
                
        # Check each week
        for week in range(num_weeks):
            total_leaves = weekly_mc[week] + weekly_el[week]
            work_days = days_per_week - total_leaves
            
            # Calculate min required hours
            adjusted_min = max(0, min_weekly_hours - total_leaves * avg_shift_hours)
            
            # Calculate max possible hours
            max_normal_hours = work_days * max_shift_hours
            max_el_hours = weekly_el[week] * two_largest_hours
            max_possible = max_normal_hours + max_el_hours
            
            if adjusted_min > max_possible:
                hourly_issues[week].append(
                    f"{name}: needs {adjusted_min:.1f}h, max {max_possible:.1f}h\n"
                    f"(MC:{weekly_mc[week]} EL:{weekly_el[week]})\n"
                )
                
    # Format hourly issues
    for week, nurses in hourly_issues.items():
        if nurses:
            week_start = week * days_per_week + 1
            week_end = min((week + 1) * days_per_week, num_days)
            reasons.append(f"⭕ Week {week+1} (Days {week_start}-{week_end}) hour shortages:\n")
            reasons.extend([f"   • {n}\n" for n in nurses[:3]])
            if len(nurses) > 3:
                reasons.append(f"   • ... and {len(nurses) - 3} more nurses\n")

    # 5. Consecutive leave constraint
    for name in nurse_names:
        all_leaves = sorted(set(mc_days.get(name, set())) | set(el_days_per_nurse.get(name, set())))
        consecutive = 0
        prev_day = -10  # Initialize with impossible value
        
        for day in sorted(all_leaves):
            if day == prev_day + 1:  # Consecutive to previous day
                consecutive += 1
            else:
                consecutive = 1  # Reset for new sequence
                
            prev_day = day
            
            if consecutive >= 3:  # 3+ consecutive days
                reasons.append(f"⚠️ {name} has {consecutive}+ consecutive leave days\n")
                break

    # Provide fallback if no clear issues found
    if not reasons:
        reasons.append("🟠 No obvious constraint violations detected. Possible causes:\n")
        reasons.append("   - Shift pattern conflicts\n")
        reasons.append("   - Weekend rest rule violations\n")
        reasons.append("   - Complex preference interactions\n")

    # Add staffing summary
    total_mc = sum(len(s) for s in mc_days.values())
    total_el = sum(len(s) for s in el_days_per_nurse.values())
    reasons.insert(0, "\n📊 Staffing Overview:\n")
    reasons.append(f"   • Nurses: {len(nurse_names)} ({len(senior_names)} seniors)\n")
    reasons.append(f"   • Schedule: {num_days} days, {shift_types} shifts/day\n")
    reasons.append(f"   • Leaves: {total_mc} MC + {total_el} EL days\n")
    reasons.append(f"   • Coverage: {min_nurses_per_shift} nurses/shift ({min_seniors_per_shift} seniors)\n")

    return reasons


# == Build Schedule Model ==
def build_schedule_model(profiles_df: pd.DataFrame,
                         preferences_df: pd.DataFrame,
                         start_date: pd.Timestamp | dt_date,
                         num_days: int,
                         min_nurses_per_shift: int = MIN_NURSES_PER_SHIFT,
                         min_seniors_per_shift: int = MIN_SENIORS_PER_SHIFT,
                         max_weekly_hours: int = MAX_WEEKLY_HOURS,
                         preferred_weekly_hours: int = PREFERRED_WEEKLY_HOURS,
                         min_acceptable_weekly_hours: int = MIN_ACCEPTABLE_WEEKLY_HOURS,
                         am_coverage_min_percent: int = AM_COVERAGE_MIN_PERCENT,
                         am_senior_min_percent: int = AM_SENIOR_MIN_PERCENT,
                         weekend_rest: bool = True,
                         back_to_back_shift: bool = False,
                         use_sliding_window: bool = USE_SLIDING_WINDOW,
                         fixed_assignments: Optional[Dict[Tuple[str,int], str]] = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Builds a nurse schedule satisfying hard constraints and optimizing soft preferences.
    Returns a schedule DataFrame, a summary DataFrame, and a violations dictionary.
    """
    # === Validate inputs ===
    missing, extra = validate_nurse_data(profiles_df, preferences_df)
    if missing or extra:
        raise InputMismatchError(
            f"Mismatch between nurse profiles and preferences:\n"
            f" • Not found in preferences: {sorted(missing)}\n"
            f" • Not found in profiles: {sorted(extra)}\n"
        )

    # === Model setup ===
    import random
    logger.info("📋 Building model...")
    model = cp_model.CpModel()
    nurses = profiles_df.to_dict(orient='records')
    nurse_names = [n['Name'].strip().upper() for n in nurses]
    og_nurse_names = nurse_names.copy()           # Save original order of list
    random.shuffle(nurse_names)                   # Shuffle order for random shift assignments for nurses
    senior_names = get_senior_set(profiles_df)    # Assume senior nurses have ≥3 years experience
    shift_str_to_idx = {label.upper(): i for i, label in enumerate(SHIFT_LABELS)}

    if isinstance(start_date, pd.Timestamp):
        date_start: dt_date = start_date.date()
    else:
        date_start = start_date

    # === Normalise EL days ===
    fixed_assignments = normalize_fixed_assignments(
        fixed_assignments,
        set(nurse_names),
        num_days
    )

    # === Preferences and MC days ===
    shift_preferences, mc_days, al_days = (
        get_shift_preferences(preferences_df, profiles_df, date_start, num_days, SHIFT_LABELS),
        get_mc_days(preferences_df, profiles_df, date_start, num_days),
        get_al_days(preferences_df, profiles_df, date_start, num_days)
    )

    # === Precompute per-nurse lookups ===
    mc_sets = {n: mc_days.get(n, set()) for n in nurse_names}
    al_sets = {n: al_days.get(n, set()) for n in nurse_names}
    el_sets = get_el_days(fixed_assignments, nurse_names)
    days_with_el = {d for days in el_sets.values() for d in days}
    prefs_by_nurse = {n: shift_preferences.get(n, {}) for n in nurse_names}

    weekend_pairs = []
    for i in range(num_days - 1):
        if (date_start + timedelta(days=i)).weekday() == 5:  # Saturday
            if i + 7 < num_days:
                weekend_pairs.append((i, i + 7))
            if i + 8 < num_days:                             # Sunday
                weekend_pairs.append((i + 1, i + 8))

    # === Variables ===
    shift_types = len(SHIFT_LABELS)
    work = {
        (n, d, s): model.NewBoolVar(f'work_{n}_{d}_{s}')
        for n in nurse_names for d in range(num_days) for s in range(shift_types)
    }
    satisfied = {}
    total_satisfied = {}
    high_priority_penalty = []
    low_priority_penalty = []

    # === Hard Constraints ===

    # === Fix MC, REST as no work ===
    for (nurse, day_idx), shift_label in fixed_assignments.items():
        label = shift_label.strip().upper()

        # Fix MC, REST, EL as no work
        if label in {"EL", "MC", "AL", "REST"}:
            # Block all shifts
            for s in range(shift_types):
                model.Add(work[nurse, day_idx, s] == 0)
            # Record MC overrides
            if label == "MC":
                mc_sets[nurse].add(day_idx)
            if label == "AL":
                al_sets[nurse].add(day_idx)
            # EL already recorded in el_sets

        # handle double-shifts, e.g. "AM/PM*"
        elif "/" in label:
            # remove any trailing "*" and split
            parts = label.rstrip("*").split("/")
            # validate
            try:
                idxs = [ shift_str_to_idx[p] for p in parts ]
            except KeyError as e:
                raise ValueError(f"Unknown shift part '{e.args[0]}' in double-shift '{label}' for {nurse}")
            # force both component shifts on, others off
            for s in idxs:
                model.Add(work[nurse, day_idx, s] == 1)
            for other_s in set(range(shift_types)) - set(idxs):
                model.Add(work[nurse, day_idx, other_s] == 0)

        # Force that one shift and turn off the others
        else:
            if label not in shift_str_to_idx:
                raise ValueError(f"Unknown shift '{label}' for {nurse}")
            s = shift_str_to_idx[label]
            model.Add(work[nurse, day_idx, s] == 1)
            for other_s in (set(range(shift_types)) - {s}):
                model.Add(work[nurse, day_idx, other_s] == 0)

    # 1. Number of shift per nurse per day
    # If no EL, each nurse can work at most 1 shift per day
    # If EL, allow at most 2 shifts per nurse if cannot satisfy other hard constraints for that day
    two_shifts = {}  

    for n in nurse_names:
        for d in range(num_days):
            if d not in days_with_el:
            # no EL here: enforce original rule
                model.AddAtMostOne(work[n, d, s] for s in range(shift_types))
            else:
            # EL day: allow either 1 or 2 shifts
                ts = model.NewBoolVar(f"two_shifts_{n}_{d}")
                two_shifts[(n, d)] = ts

                if d in el_sets[n]:     # if nurse has el on that day, explicitly make ts = false for the nurse
                    model.Add(ts == 0)

                # If ts==False → sum_s work ≤ 1
                model.Add(sum(work[n, d, s] for s in range(shift_types)) <= 1).OnlyEnforceIf(ts.Not())
                # If ts==True  → sum_s work ≤ 2
                model.Add(sum(work[n, d, s] for s in range(shift_types)) <= 2).OnlyEnforceIf(ts)

                # If double shift, apply penalty
                high_priority_penalty.append(ts * DOUBLE_SHIFT_PENALTY)

    # 2. Each nurse works <= 42 hours/week (hard), adjustable based on MC; ideally min 40 (soft), at least 30 (hard)
    for n in nurse_names:
        mc = mc_sets[n]
        al = al_sets[n]
        el = el_sets[n]
        num_weeks = (num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK

        # Precompute daily‑hours expressions if using sliding window
        if use_sliding_window:
            hours_by_day = [
                sum(work[n, d, s] * int(SHIFT_HOURS[s]) for s in range(shift_types))
                for d in range(num_days)
            ]

            for d in range(num_days):
                # Maximum working hours every 7 day sliding window (exp: Day 0 to Day 6, then Day 1 to Day 7, etc.)
                if d >= DAYS_PER_WEEK - 1:
                    window = range(d - (DAYS_PER_WEEK - 1), d + 1)
                    window_hours = sum(hours_by_day[i] for i in window)
                    mc_count = sum(1 for i in window if i in mc)
                    al_count = sum(1 for i in window if i in al)
                    el_count = sum(1 for i in window if i in el)
                    adj = (mc_count + al_count + el_count) * AVG_HOURS                  # MC & EL hours deducted from max/pref/min hours
                    eff_max_hours = max(0, max_weekly_hours - adj)           # <= 42 - x
                    model.Add(window_hours <= eff_max_hours)

        # Full‑week minimum at each 7‑day boundary (e.g. Day 6, then Day 13, etc.)
        for w in range(num_weeks):
            days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, num_days))

            if not use_sliding_window and len(days) < DAYS_PER_WEEK:
                continue  # Skip incomplete weeks for extra days if not using sliding window

            if use_sliding_window:
                weekly_hours = sum(hours_by_day[i] for i in days)
            else:
                weekly_hours = sum(
                    work[n, d, s] * int(SHIFT_HOURS[s])
                    for d in days for s in range(shift_types)
                )

            mc_count = sum(1 for i in days if i in mc)
            al_count = sum(1 for i in days if i in al)
            el_count = sum(1 for i in days if i in el)
            adj = (mc_count + al_count + el_count) * AVG_HOURS

            eff_pref_hours = max(0, preferred_weekly_hours - adj)        # >= 40 - x
            eff_min_hours = max(0, min_acceptable_weekly_hours - adj)    # >= 30 - x

            model.Add(weekly_hours >= eff_min_hours)
            if not USE_SLIDING_WINDOW:
                eff_max_hours = max(0, max_weekly_hours - adj)           # <= 42 - x 
                model.Add(weekly_hours <= eff_max_hours)    

            if eff_pref_hours > eff_min_hours:
                flag = model.NewBoolVar(f'pref_{n}_w{w}')
                model.Add(weekly_hours >= eff_pref_hours).OnlyEnforceIf(flag)
                model.Add(weekly_hours < eff_pref_hours).OnlyEnforceIf(flag.Not())

                high_priority_penalty.append(flag.Not() * PREF_HOURS_PENALTY)

    # 3. Each shift must have at least 4 nurses and at least 1 senior
    for d in range(num_days):
        for s in range(shift_types):
            model.Add(sum(work[n, d, s] for n in nurse_names) >= min_nurses_per_shift)
            model.Add(sum(work[n, d, s] for n in senior_names) >= min_seniors_per_shift)

    # 4. Weekend work requires rest on the same day next weekend
    if weekend_rest:
        for n in nurse_names:
            for d1, d2 in weekend_pairs:
                model.Add(sum(work[n, d1, s] for s in range(shift_types)) + 
                        sum(work[n, d2, s] for s in range(shift_types)) <= 1)
            
    # 5. Night shift will never be followed by AM shift
    if not back_to_back_shift:
        for n in nurse_names:
            for d in range(1, num_days):
                model.AddImplication(work[n, d - 1, 2], work[n, d, 0].Not())

    # 6. MC/AL days: cannot assign any shift
    for n in nurse_names:
        for d in mc_sets[n] | al_sets[n]:
            model.Add(sum(work[n, d, s] for s in range(shift_types)) == 0)

    # 7. Max 2 MC/AL days per week and no more than 2 consecutive MC/AL days
    for n in nurse_names:
        mc = mc_sets[n]
        al = al_sets[n]
        num_weeks = (num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK

        for w in range(num_weeks):
            days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, num_days))
            mc_in_week = [d for d in days if d in mc]
            al_in_week = [d for d in days if d in al]
            if len(mc_in_week) > MAX_MC_DAYS_PER_WEEK:
                raise InvalidMCError(
                    f"❌ Nurse {n} has more than {MAX_MC_DAYS_PER_WEEK} MCs in week {w+1}.\n"
                    f"Days: {sorted(mc_in_week)}"
                )
            if len(al_in_week) > MAX_AL_DAYS_PER_WEEK:
                raise InvalidALError(
                    f"❌ Nurse {n} has more than {MAX_AL_DAYS_PER_WEEK} ALs in week {w+1}.\n"
                    f"Days: {sorted(al_in_week)}"
                )

        sorted_mc = sorted(mc)
        sorted_al = sorted(al)
        for i in range(len(sorted_mc) - MAX_CONSECUTIVE_MC):
            if sorted_mc[i + 2] - sorted_mc[i] == MAX_CONSECUTIVE_MC:
                raise ConsecutiveMCError(
                    f"❌ Nurse {n} has more than 2 consecutive MC days: "
                    f"{sorted_mc[i]}, {sorted_mc[i+1]}, {sorted_mc[i+2]}"
                )
        for i in range(len(sorted_al) - MAX_CONSECUTIVE_AL):
            if sorted_al[i + 2] - sorted_al[i] == MAX_CONSECUTIVE_AL:
                raise ConsecutiveALError(
                    f"❌ Nurse {n} has more than 2 consecutive AL days: "
                    f"{sorted_al[i]}, {sorted_al[i+1]}, {sorted_al[i+2]}.\n"
                )

    # === Soft Constraints ===
    # Coverage level flags
    am_lvl1 = am_coverage_min_percent          # typically 60
    am_lvl2 = max(am_lvl1 - 10, 0)              # 50
    am_lvl3 = max(am_lvl1 - 20, 0)              # 40

    senior_lvl1 = am_senior_min_percent            # typically 60
    senior_lvl2 = max(senior_lvl1 - 10, 0)              # 50
    senior_lvl3 = max(senior_lvl1 - 20, 0)              # 40

    # 1. AM coverage per day should be >=60%, ideally
    for d in range(num_days):
        total_shifts = sum(work[n, d, s] for n in nurse_names for s in range(shift_types))
        am_shifts = sum(work[n, d, 0] for n in nurse_names)

        # AM shift coverage level flags
        am_lvl1_ok = model.NewBoolVar(f'day_{d}_am_lvl1')
        am_lvl2_ok = model.NewBoolVar(f'day_{d}_am_lvl2')
        am_lvl3_ok = model.NewBoolVar(f'day_{d}_am_lvl3')

        # Soft targets on AM shift
        model.Add(am_shifts * 100 >= am_lvl1 * total_shifts).OnlyEnforceIf(am_lvl1_ok)
        model.Add(am_shifts * 100 <  am_lvl1 * total_shifts).OnlyEnforceIf(am_lvl1_ok.Not())

        model.Add(am_shifts * 100 >= am_lvl2 * total_shifts).OnlyEnforceIf(am_lvl2_ok)
        model.Add(am_shifts * 100 <  am_lvl2 * total_shifts).OnlyEnforceIf(am_lvl2_ok.Not())

        model.Add(am_shifts * 100 >= am_lvl3 * total_shifts).OnlyEnforceIf(am_lvl3_ok)
        model.Add(am_shifts * 100 <  am_lvl3 * total_shifts).OnlyEnforceIf(am_lvl3_ok.Not())

        # Hard fallback condition
        all_levels_failed = model.NewBoolVar(f'day_{d}_all_levels_failed')
        model.AddBoolAnd([am_lvl1_ok.Not(), am_lvl2_ok.Not(), am_lvl3_ok.Not()]).OnlyEnforceIf(all_levels_failed)
        model.AddBoolOr([am_lvl1_ok, am_lvl2_ok, am_lvl3_ok]).OnlyEnforceIf(all_levels_failed.Not())

        # Explicit PM and Night shift counts
        pm_shift_nurses = sum(work[n, d, 1] for n in nurse_names)
        night_shift_nurses = sum(work[n, d, 2] for n in nurse_names)

        # Enforce AM > PM and AM > Night if all levels fail (hard constraint)
        model.Add(am_shifts > pm_shift_nurses).OnlyEnforceIf(all_levels_failed)
        model.Add(am_shifts > night_shift_nurses).OnlyEnforceIf(all_levels_failed)

        # Penalties for failing soft levels
        high_priority_penalty.append(
            am_lvl1_ok.Not() * AM_COVERAGE_PENALTIES[0] +
            am_lvl2_ok.Not() * AM_COVERAGE_PENALTIES[1] +
            am_lvl3_ok.Not() * AM_COVERAGE_PENALTIES[2]
        )

    # 2) Seniors coverage on AM shift should be >= 60%, ideally
    for d in range(num_days):
        am_shifts = sum(work[n, d, 0] for n in nurse_names)
        am_seniors = sum(work[n, d, 0] for n in senior_names)

        # Senior ratio level flags
        senior_lvl1_ok = model.NewBoolVar(f'day_{d}_am_senior_lvl1')
        senior_lvl2_ok = model.NewBoolVar(f'day_{d}_am_senior_lvl2')
        senior_lvl3_ok = model.NewBoolVar(f'day_{d}_am_senior_lvl3')

        # Soft targets: senior nurses on AM shifts
        model.Add(am_seniors * 100 >= senior_lvl1 * am_shifts).OnlyEnforceIf(senior_lvl1_ok)
        model.Add(am_seniors * 100 <  senior_lvl1 * am_shifts).OnlyEnforceIf(senior_lvl1_ok.Not())

        model.Add(am_seniors * 100 >= senior_lvl2 * am_shifts).OnlyEnforceIf(senior_lvl2_ok)
        model.Add(am_seniors * 100 <  senior_lvl2 * am_shifts).OnlyEnforceIf(senior_lvl2_ok.Not())

        model.Add(am_seniors * 100 >= senior_lvl3 * am_shifts).OnlyEnforceIf(senior_lvl3_ok)
        model.Add(am_seniors * 100 <  senior_lvl3 * am_shifts).OnlyEnforceIf(senior_lvl3_ok.Not())

        # Hard fallback: all levels failed
        senior_all_levels_failed = model.NewBoolVar(f'day_{d}_am_senior_all_levels_failed')
        model.AddBoolAnd([senior_lvl1_ok.Not(), senior_lvl2_ok.Not(), senior_lvl3_ok.Not()]).OnlyEnforceIf(senior_all_levels_failed)
        model.AddBoolOr([senior_lvl1_ok, senior_lvl2_ok, senior_lvl3_ok]).OnlyEnforceIf(senior_all_levels_failed.Not())

        # Explicit PM and Night seniors shift counts
        pm_shift_seniors = sum(work[n, d, 1] for n in senior_names)
        night_shift_seniors = sum(work[n, d, 2] for n in senior_names)

        # Enforce AM seniors > PM seniors and AM seniors > Night seniors if all levels fail (hard constraint)
        model.Add(am_seniors > pm_shift_seniors).OnlyEnforceIf(senior_all_levels_failed)
        model.Add(am_seniors > night_shift_seniors).OnlyEnforceIf(senior_all_levels_failed)

        # Penalties for senior ratio violations
        high_priority_penalty.append(
            senior_lvl1_ok.Not() * AM_SENIOR_PENALTIES[0] +
            senior_lvl2_ok.Not() * AM_SENIOR_PENALTIES[1] +
            senior_lvl3_ok.Not() * AM_SENIOR_PENALTIES[2]
        )


    # 3. Preference satisfaction
    for n in nurse_names:
        prefs = prefs_by_nurse[n]
        satisfied_list = []

        for d in range(num_days):
            if d in prefs:
                s = prefs[d]
                sat = model.NewBoolVar(f'sat_{n}_{d}')
                model.Add(work[n, d, s] == 1).OnlyEnforceIf(sat)
                model.Add(work[n, d, s] == 0).OnlyEnforceIf(sat.Not())
                satisfied[(n, d)] = sat
                satisfied_list.append(sat)
                # Add penalty if preference not satisfied
                low_priority_penalty.append(sat.Not() * PREF_MISS_PENALTY)
            else:
                satisfied_const = model.NewConstant(0)
                satisfied[(n, d)] = satisfied_const
                satisfied_list.append(satisfied_const)

        total_satisfied[n] = model.NewIntVar(0, num_days, f'total_sat_{n}')
        model.Add(total_satisfied[n] == sum(satisfied_list))

    # 4. Fairness constraint on preference satisfaction gap
    pct_sat = {}
    for n, prefs in prefs_by_nurse.items():
        count = len(prefs)
        if count > 0:
            p = model.NewIntVar(0, 100, f"pct_sat_{n}")
            pct_sat[n] = p
            model.Add(p * count == total_satisfied[n] * 100)

        else:
            pct_sat[n] = None

    valid_pcts = [p for p in pct_sat.values() if p is not None]
    if valid_pcts:
        min_pct = model.NewIntVar(0, 100, "min_pct")
        max_pct = model.NewIntVar(0, 100, "max_pct")
        model.AddMinEquality(min_pct, valid_pcts)
        model.AddMaxEquality(max_pct, valid_pcts)

        gap_pct = model.NewIntVar(0, 100, "gap_pct")
        model.Add(gap_pct == max_pct - min_pct)

        # Start penalise fairness when gap_pct >= 60 based on distance from 60
        over_gap  = model.NewIntVar(0, 100, "over_gap")
        model.AddMaxEquality(over_gap, [gap_pct - FAIRNESS_GAP_THRESHOLD, 0])
        low_priority_penalty.append(gap_pct * FAIRNESS_GAP_PENALTY)

    # 5. Balance in number of each shift type assigned to nurse
    # IMBALANCE_PENALTY = 1

    # # (A) Precompute counts just once
    # counts: dict[tuple[str,int], cp_model.IntVar] = {}
    # for n in nurse_names:
    #     for s in range(shift_types):
    #         C = model.NewIntVar(0, num_days, f"count_{n}_s{s}")
    #         model.Add(C == sum(work[n, d, s] for d in range(num_days)))
    #         counts[(n, s)] = C

    # # (B) For each nurse, build min/max/gap
    # for n in nurse_names:
    #     c_vars = [counts[(n, s)] for s in range(shift_types)]
    #     minC = model.NewIntVar(0, num_days, f"min_count_{n}")
    #     maxC = model.NewIntVar(0, num_days, f"max_count_{n}")

    #     model.AddMinEquality(minC, c_vars)
    #     model.AddMaxEquality(maxC, c_vars)

    #     gap = model.NewIntVar(0, num_days, f"gap_{n}")
    #     model.Add(gap == maxC - minC)

    #     # soft-penalize any imbalance
    #     low_priority_penalty.append(gap * IMBALANCE_PENALTY)

    # === Objective ===
    # === Phase 1: minimize total penalties ===
    logger.info("🚀 Phase 1: minimizing penalties…")
    # 1. Tell the model to minimize penalty sum
    model.Minimize(sum(high_priority_penalty))

    # debug: print model size
    proto = model.Proto()
    logger.info(f"→ #constraints_p1 = {len(proto.constraints)},  #bool_vars = {len(proto.variables)}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0  # tunable
    solver.parameters.random_seed = 42
    solver.parameters.relative_gap_limit = 0.01
    solver.parameters.num_search_workers = 8
    solver.parameters.randomize_search = True 
    solver.parameters.log_search_progress = False

    status1 = solver.Solve(model)
    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
    logger.info(f"High Priority Penalty Phase 1: {solver.ObjectiveValue()}")
    logger.info(f"Low Priority Penalty Phase 1: {solver.Value(sum(low_priority_penalty))}")
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Detailed infeasibility analysis
        reasons = analyze_infeasibility(
            nurse_names,
            senior_names,
            mc_sets,
            el_sets,
            num_days,
            len(SHIFT_LABELS),
            min_nurses_per_shift,
            min_seniors_per_shift,
            min_acceptable_weekly_hours,
            DAYS_PER_WEEK,
            SHIFT_HOURS
        )
        
        error_msg = "❌ No feasible solution. Identified issues:\n\n"
        error_msg += "\n".join(reasons)
        raise NoFeasibleSolutionError(error_msg)

    # save "best" solution found
    cached_values = {}
    for n in nurse_names:
        for d in range(num_days):
            for s in range(shift_types):
                cached_values[(n, d, s)] = solver.Value(work[n, d, s])

    cached_total_prefs_met = 0
    for n in nurse_names:
        for d in range(num_days):
            picked = [s for s in range(shift_types) if cached_values[(n, d, s)]]
            pref = prefs_by_nurse[n].get(d)
            if pref is not None and len(picked) == 1 and pref in picked:
                cached_total_prefs_met += 1

    cached_gap = solver.Value(gap_pct) if valid_pcts else "N/A"
    high1 = solver.ObjectiveValue()
    best_penalty = solver.ObjectiveValue() + solver.Value(sum(low_priority_penalty))
    logger.info(f"▶️ Phase 1 complete: best total penalty = {best_penalty}; best fairness gap = {cached_gap}")

    # === Phase 2: maximize preferences under that penalty bound ===
    # only run phase 2 if shift preferences exist
    # if any(shift_preferences.values()):
    if low_priority_penalty:
        logger.info("🚀 Phase 2: maximizing preferences…")
        # 2. Freeze the penalty sum at its optimum
        model.Add(sum(high_priority_penalty) <= int(high1))
        if valid_pcts:
          model.Add(gap_pct <= cached_gap)
            # model.AddLinearConstraint(gap_pct, 0, T)

        # 3. Switch objective to preferences
        # preference_obj = sum(total_satisfied[n] for n in nurse_names)
        # model.Maximize(preference_obj)
        model.Minimize(sum(low_priority_penalty))

        # debug: print model size
        proto = model.Proto()
        logger.info(f"→ #constraints_p2 = {len(proto.constraints)},  #bool_vars = {len(proto.variables)}")

        # 4. Re-solve (you can reset your time budget)
        solver = cp_model.CpSolver()
        for (n, d, s), val in cached_values.items():
            model.AddHint(work[n, d, s], val)
        solver.parameters.max_time_in_seconds = 180.0
        solver.parameters.random_seed = 42
        solver.parameters.relative_gap_limit = 0.01
        solver.parameters.num_search_workers = 8
        solver.parameters.randomize_search = True
        solver.parameters.log_search_progress = False

        status2 = solver.Solve(model)
        logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
        logger.info(f"High Priority Penalty Phase 2: {solver.Value(sum(high_priority_penalty))}")
        logger.info(f"Low Priority Penalty Phase 2: {solver.ObjectiveValue()}")
        use_fallback = status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        if use_fallback:
            logger.info("⚠️ Phase 2 failed: using fallback Phase 1 solution.")
            logger.info(f"Solver Phase 2 status: {solver.StatusName(status2)}")
        else:
            logger.info(f"▶️ Phase 2 complete")
            best_penalty = solver.Value(sum(high_priority_penalty)) + solver.ObjectiveValue()
            new_total_prefs_met = 0
            for n in nurse_names:
                for d in range(num_days):
                    picked = [s for s in range(shift_types) if solver.Value(work[n, d, s])]
                    pref = prefs_by_nurse[n].get(d)
                    if pref is not None and len(picked) == 1 and pref in picked:
                        new_total_prefs_met += 1

    else:
        logger.info("⏭️ Skipping Phase 2: No shift preferences provided.")
        use_fallback = True
    # use_fallback = True

    # === Extract & report ===
    logger.info("✅ Done!")
    logger.info(f"📊 Total penalties = {best_penalty}")
    logger.info(f"🔍 Total preferences met = {cached_total_prefs_met if use_fallback else new_total_prefs_met}")
    if 'gap_pct' in locals():
        logger.info(f"📈 Preference gap (max - min) = {cached_gap if use_fallback else solver.Value(gap_pct)}")
    else:
        logger.info("📈 Preference gap (max - min) = N/A")

    # === Extract Results ===
    dates = [date_start + timedelta(days=i) for i in range(num_days)]
    headers = [d.strftime('%a %Y-%m-%d') for d in dates]
    num_weeks = (num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK     # number of weeks in the schedule
    schedule = {}
    summary = []
    violations = {"Double Shifts": [], "Low_AM_Days": [], "Low_Senior_AM_Days": [], "Low_Hours_Nurses": [], "Preference_Unmet": [], "Fairness_Gap": cached_gap if use_fallback or 'gap_pct' not in locals() else solver.Value(gap_pct)}

    for n in nurse_names:
        row = []
        hours_per_week = [0] * num_weeks
        shift_counts = [0, 0, 0, 0]  # AM, PM, Night, REST
        double_shift_days = []
        prefs_met = 0
        prefs_unmet = []

        for d in range(num_days):
            picked = []
            
            if d in mc_sets[n]:
                shift = "MC"
            elif d in al_sets[n]:
                shift = "AL"
            elif (n, d) in fixed_assignments and fixed_assignments[(n, d)].strip().upper() == "EL":
                shift = "EL"
            else:
                if use_fallback:
                    picked = [s for s in range(shift_types) if cached_values[(n, d, s)]]
                else:
                    picked = [s for s in range(shift_types) if solver.Value(work[n, d, s])]

                if len(picked) == 2:
                    double_shift_days.append(dates[d].strftime('%a %Y-%m-%d'))
                
                match(len(picked)):
                    case 0:
                        shift = "Rest"
                        shift_counts[3] += 1
                    case 1:
                        shift = SHIFT_LABELS[picked[0]]
                    case 2:
                        first, second = sorted(picked)
                        shift = f"{SHIFT_LABELS[first]}/{SHIFT_LABELS[second]}*"
                    case _:
                        shift = "OVER*"
            row.append(shift)

            week_idx = d // DAYS_PER_WEEK
            for p in picked:
                hours_per_week[week_idx] += int(SHIFT_HOURS[p])
                shift_counts[p] += 1

            pref = prefs_by_nurse[n].get(d)
            if pref is not None:
                if len(picked) == 1 and picked[0] == pref:
                    prefs_met += 1
                else:
                    prefs_unmet.append(f"{dates[d].strftime('%a %Y-%m-%d')} (wanted {SHIFT_LABELS[pref]})")

        for w in range(num_weeks):
            days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, num_days))
            if len(days) < DAYS_PER_WEEK:
                continue  # skip incomplete weeks
            mc_count_week = len(mc_sets[n] & set(days))
            el_count_week = len(el_sets[n] & set(days))
            eff_pref_hours = max(0, preferred_weekly_hours - (mc_count_week + el_count_week) * AVG_HOURS)

            if hours_per_week[w] < eff_pref_hours:
                violations["Low_Hours_Nurses"].append(f"{n} Week {w+1}: {hours_per_week[w]}h; pref {eff_pref_hours}")

        if prefs_unmet:
            violations["Preference_Unmet"].append(f"{n}: {'; '.join(prefs_unmet)}")

        if double_shift_days:
            violations["Double Shifts"].append(f"{n}: {'; '.join(double_shift_days)}")

        schedule[n] = row
        summary_row = {
            "Nurse":    n,
            "AL":       len(al_sets[n]),
            "MC":       len(mc_sets[n]),
            "EL":       len(el_sets[n]),
            "Rest":     shift_counts[3],
            "AM":       shift_counts[0],
            "PM":       shift_counts[1],
            "Night":    shift_counts[2],
            "Double Shifts": len(double_shift_days),
        }
        for w in range(num_weeks):
            summary_row[f"Hours_Week{w+1}"] = hours_per_week[w]
        summary_row.update({
            "Prefs_Met": prefs_met,
            "Prefs_Unmet": len(prefs_unmet),
            "Unmet_Details": "; ".join(prefs_unmet),
        })
        summary.append(summary_row)

    for d in range(num_days):
        if use_fallback:
            am_n = sum(cached_values[(n, d, 0)] for n in nurse_names)
            total_n = sum(cached_values[(n, d, s)] for n in nurse_names for s in range(shift_types))
            am_snr = sum(cached_values[(n, d, 0)] for n in senior_names)
        else:
            am_n = sum(solver.Value(work[n, d, 0]) for n in nurse_names)
            total_n = sum(solver.Value(work[n, d, s]) for n in nurse_names for s in range(shift_types))
            am_snr = sum(solver.Value(work[n, d, 0]) for n in senior_names)

        if total_n and am_n / total_n < (am_coverage_min_percent / 100):
            violations["Low_AM_Days"].append(f"{dates[d].strftime('%a %Y-%m-%d')} ({am_n/total_n:.0%})")
        if am_n and am_snr / am_n < (am_senior_min_percent / 100):
            violations["Low_Senior_AM_Days"].append(f"{dates[d].strftime('%a %Y-%m-%d')} (Seniors {am_snr/am_n:.0%})")

    logger.info("\n⚠️ Soft Constraint Violations Summary:")
    for key, items in violations.items():
        match key:
            case "Preference_Unmet":
                total_unmet = sum(s["Prefs_Unmet"] for s in summary)
                logger.info(f"🔸 {key}: {total_unmet} unmet preferences across {len(items)} nurses")
            case "Fairness_Gap":
                logger.info(f"🔸 {key}: {len(items) if isinstance(items, list) else items} %")
            case _:
                logger.info(f"🔸 {key}: {len(items) if isinstance(items, list) else items} cases")
        if isinstance(items, list):
            for item in items:
                logger.info(f"   - {item}")

    logger.info("📁 Schedule and summary generated.")
    schedule_df = pd.DataFrame.from_dict(schedule, orient='index', columns=headers).reindex(og_nurse_names)
    summary_df = pd.DataFrame(summary).set_index("Nurse").reindex(og_nurse_names).reset_index()
    return schedule_df, summary_df, violations