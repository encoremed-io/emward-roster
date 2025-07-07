from matplotlib.pyplot import step
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
from core.hard_rules import define_hard_rules
from model.setup import *

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

# == Build Schedule Model ==
def build_schedule_model(profiles_df: pd.DataFrame,
                         preferences_df: pd.DataFrame,
                         start_date: pd.Timestamp | dt_date,
                         num_days: int,
                         shift_durations: List[int] = SHIFT_DURATIONS,
                         min_nurses_per_shift: int = MIN_NURSES_PER_SHIFT,
                         min_seniors_per_shift: int = MIN_SENIORS_PER_SHIFT,
                         max_weekly_hours: int = MAX_WEEKLY_HOURS,
                         preferred_weekly_hours: int = PREFERRED_WEEKLY_HOURS,
                         min_acceptable_weekly_hours: int = MIN_ACCEPTABLE_WEEKLY_HOURS,
                         min_weekly_hours_hard: bool = False,
                         am_coverage_min_percent: int = AM_COVERAGE_MIN_PERCENT,
                         am_coverage_min_hard: bool = False,
                         am_coverage_relax_step: int = AM_COVERAGE_RELAX_STEP,
                         am_senior_min_percent: int = AM_SENIOR_MIN_PERCENT,
                         am_senior_min_hard: bool = False,
                         am_senior_relax_step: int = AM_SENIOR_RELAX_STEP,
                         weekend_rest: bool = True,
                         back_to_back_shift: bool = False,
                         use_sliding_window: bool = USE_SLIDING_WINDOW,
                         fixed_assignments: Optional[Dict[Tuple[str,int], str]] = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Builds a nurse schedule satisfying hard constraints and optimizing soft preferences.
    Returns a schedule DataFrame, a summary DataFrame, and a violations dictionary.
    """
    # === Validate inputs ===
    validate_data(profiles_df, preferences_df)

    # === Model setup ===
    logger.info("📋 Building model...")
    model, nurse_names, og_nurse_names, senior_names, shift_str_to_idx, date_start, \
    hard_rules, shift_preferences, prefs_by_nurse, fixed_assignments, mc_sets, \
    al_sets, el_sets, weekend_pairs, shift_types, work = setup_model(
        profiles_df, preferences_df, start_date, num_days, SHIFT_LABELS, fixed_assignments
    )

    # === Variables ===
    days_with_el = {d for days in el_sets.values() for d in days}
    is_satisfied = {}
    total_satisfied = {}
    high_priority_penalty = []
    low_priority_penalty = []

    # === Phase 1 Constraints ===

    # === Handle special assignments ===
    for (nurse, day_idx), shift_label in fixed_assignments.items():
        label = shift_label.strip().upper()

        # Fix MC, REST, AL, EL as no work
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
    # Convert to minutes
    max_weekly_minutes = max_weekly_hours * 60
    preferred_weekly_minutes = preferred_weekly_hours * 60
    min_acceptable_weekly_minutes = min_acceptable_weekly_hours * 60
    avg_minutes = min(shift_durations)

    for n in nurse_names:
        mc = mc_sets[n]
        al = al_sets[n]
        el = el_sets[n]
        num_weeks = (num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK

        # Precompute daily-hours expressions if using sliding window
        if use_sliding_window:
            minutes_by_day = [
                sum(work[n, d, s] * int(shift_durations[s]) for s in range(shift_types))
                for d in range(num_days)
            ]

            for d in range(num_days):
                # Maximum working hours every 7 day sliding window (exp: Day 0 to Day 6, then Day 1 to Day 7, etc.)
                if d >= DAYS_PER_WEEK - 1:
                    window = range(d - (DAYS_PER_WEEK - 1), d + 1)
                    window_minutes = sum(minutes_by_day[i] for i in window)
                    mc_count = sum(1 for i in window if i in mc)
                    al_count = sum(1 for i in window if i in al)
                    el_count = sum(1 for i in window if i in el)
                    adj = (mc_count + al_count + el_count) * avg_minutes              # MC & EL hours deducted from max/pref/min hours
                    eff_max_minutes = max(0, max_weekly_minutes - adj)                  # <= 42 - x
                    model.Add(window_minutes <= eff_max_minutes).OnlyEnforceIf(hard_rules["Max weekly hours"].flag)

        # Full-week minimum at each 7-day boundary (e.g. Day 6, then Day 13, etc.)
        for w in range(num_weeks):
            days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, num_days))

            if not use_sliding_window and len(days) < DAYS_PER_WEEK:
                continue  # Skip incomplete weeks for extra days if not using sliding window

            if use_sliding_window:
                weekly_minutes = sum(minutes_by_day[i] for i in days)
            else:
                weekly_minutes = sum(
                    work[n, d, s] * int(shift_durations[s])
                    for d in days for s in range(shift_types)
                )

            mc_count = sum(1 for i in days if i in mc)
            al_count = sum(1 for i in days if i in al)
            el_count = sum(1 for i in days if i in el)
            adj = (mc_count + al_count + el_count) * avg_minutes

            eff_min_minutes = max(0, min_acceptable_weekly_minutes - adj)    # >= 30 - x

            model.Add(weekly_minutes >= eff_min_minutes).OnlyEnforceIf(hard_rules["Min weekly hours"].flag)
            if not use_sliding_window:
                eff_max_minutes = max(0, max_weekly_minutes - adj)           # <= 42 - x 
                model.Add(weekly_minutes <= eff_max_minutes).OnlyEnforceIf(hard_rules["Max weekly hours"].flag)
            
            if not min_weekly_hours_hard:
                eff_pref_minutes = max(0, preferred_weekly_minutes - adj)        # >= 40 - x
                if eff_pref_minutes > eff_min_minutes:
                    flag = model.NewBoolVar(f'pref_{n}_w{w}')
                    model.Add(weekly_minutes >= eff_pref_minutes).OnlyEnforceIf(flag)
                    model.Add(weekly_minutes < eff_pref_minutes).OnlyEnforceIf(flag.Not())

                    high_priority_penalty.append(flag.Not() * PREF_HOURS_PENALTY)

    # 3. Each shift must have at least 4 nurses and at least 1 senior
    for d in range(num_days):
        for s in range(shift_types):
            model.Add(sum(work[n, d, s] for n in nurse_names) >= min_nurses_per_shift).OnlyEnforceIf(hard_rules["Min nurses"].flag)
            model.Add(sum(work[n, d, s] for n in senior_names) >= min_seniors_per_shift).OnlyEnforceIf(hard_rules["Min seniors"].flag)

    # 4. Weekend work requires rest on the same day next weekend
    if weekend_rest:
        for n in nurse_names:
            for d1, d2 in weekend_pairs:
                model.Add(sum(work[n, d1, s] for s in range(shift_types)) + 
                        sum(work[n, d2, s] for s in range(shift_types)) <= 1).OnlyEnforceIf(hard_rules["Weekend rest"].flag)
            
    # 5. Night shift will never be followed by AM shift
    if not back_to_back_shift:
        for n in nurse_names:
            for d in range(1, num_days):
                model.AddImplication(work[n, d - 1, 2], work[n, d, 0].Not()).OnlyEnforceIf(hard_rules["No b2b"].flag)

    # 6. MC/AL days: cannot assign any shift
    for n in nurse_names:
        for d in mc_sets[n] | al_sets[n]:
            model.Add(sum(work[n, d, s] for s in range(shift_types)) == 0)

    # 7. Max 2 MC days per week and no more than 2 consecutive MC days
    for n in nurse_names:
        mc = mc_sets[n]
        al = al_sets[n]
        num_weeks = (num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK

        for w in range(num_weeks):
            days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, num_days))
            mc_in_week = [d for d in days if d in mc]
            if len(mc_in_week) > MAX_MC_DAYS_PER_WEEK:
                raise InvalidMCError(
                    f"❌ Nurse {n} has more than {MAX_MC_DAYS_PER_WEEK} MCs in week {w+1}.\n"
                    f"Days: {sorted(mc_in_week)}"
                )

        sorted_mc = sorted(mc)
        for i in range(len(sorted_mc) - MAX_CONSECUTIVE_MC):
            if sorted_mc[i + 2] - sorted_mc[i] == MAX_CONSECUTIVE_MC:
                raise ConsecutiveMCError(
                    f"❌ Nurse {n} has more than 2 consecutive MC days: "
                    f"{sorted_mc[i]}, {sorted_mc[i+1]}, {sorted_mc[i+2]}"
                )

    # 8. AM coverage per day should be >=60% of total working nurses, ideally
    if not am_coverage_min_hard:
        levels = list(range(am_coverage_min_percent, 34, -am_coverage_relax_step))  # [65, 60, 55] if step = 5
        penalties = [(i + 1) * AM_COVERAGE_PENALTY for i in range(len(levels))]   # [5, 10, 15]

    for d in range(num_days):
        total_shifts = sum(work[n, d, s] for n in nurse_names for s in range(shift_types))
        am_shifts = sum(work[n, d, 0] for n in nurse_names)
        pm_shifts = sum(work[n, d, 1] for n in nurse_names)
        night_shifts = sum(work[n, d, 2] for n in nurse_names)

        if am_coverage_min_hard:
            model.Add(am_shifts * 100 >= am_coverage_min_percent * total_shifts).OnlyEnforceIf(hard_rules["AM cov min"].flag)

        else:
            flags = [model.NewBoolVar(f"day_{d}_am_min_{lvl}") for lvl in levels]   # flags for each level
            fallback = model.NewBoolVar(f"day_{d}_all_levels_failed")   # fallback flag

            for flag, lvl in zip(flags, levels):
                model.Add(am_shifts * 100 >= lvl * total_shifts).OnlyEnforceIf(flag)
                model.Add(am_shifts * 100 < lvl * total_shifts).OnlyEnforceIf(flag.Not())

            # Hard fallback: all levels failed
            model.AddBoolAnd([flag.Not() for flag in flags]).OnlyEnforceIf(fallback)
            model.AddBoolOr(flags).OnlyEnforceIf(fallback.Not())
            # Enforce AM > PM and AM > Night if all levels fail (hard constraint)
            model.Add(am_shifts > pm_shifts).OnlyEnforceIf(fallback).OnlyEnforceIf(hard_rules["AM cov majority"].flag)
            model.Add(am_shifts > night_shifts).OnlyEnforceIf(fallback).OnlyEnforceIf(hard_rules["AM cov majority"].flag)

            # Penalties for AM ratio violations (only failed levels penalised)
            for i, lvl in enumerate(levels):
                penalise = model.NewBoolVar(f"day_{d}_penalise_lvl_{lvl}_am")
                model.Add(penalise == flags[i].Not())
                high_priority_penalty.append(penalise * penalties[i])

    # 9) Seniors coverage on AM shift should be >= 60% of total AM seniors, ideally
    if not am_senior_min_hard:
        levels = list(range(am_senior_min_percent, 50, -am_senior_relax_step))  # [65, 60, 55] if step = 5
        penalties = [(i + 1) * AM_SENIOR_PENALTY for i in range(len(levels))]   # [5, 10, 15]

    for d in range(num_days):
        am_shifts = sum(work[n, d, 0] for n in nurse_names)
        am_seniors = sum(work[n, d, 0] for n in senior_names)
        am_juniors = am_shifts - am_seniors     # number of junior nurses on AM shift

        if am_senior_min_hard:
            model.Add(am_seniors * 100 >= am_senior_min_percent * am_shifts).OnlyEnforceIf(hard_rules["AM snr min"].flag)

        else:
            flags = [model.NewBoolVar(f"day_{d}_senior_min_{lvl}") for lvl in levels]   # flags for each level
            fallback = model.NewBoolVar(f"day_{d}_all_levels_failed")   # fallback flag

            for flag, lvl in zip(flags, levels):
                model.Add(am_seniors * 100 >= lvl * am_shifts).OnlyEnforceIf(flag)
                model.Add(am_seniors * 100 < lvl * am_shifts).OnlyEnforceIf(flag.Not())

            # Hard fallback: all levels failed
            model.AddBoolAnd([flag.Not() for flag in flags]).OnlyEnforceIf(fallback)
            model.AddBoolOr(flags).OnlyEnforceIf(fallback.Not())
            # Enforce AM seniors >= AM junior if all levels fail (hard constraint)
            model.Add(am_seniors >= am_juniors).OnlyEnforceIf(fallback).OnlyEnforceIf(hard_rules["AM snr majority"].flag)

            # Penalties for senior ratio violations (only failed levels penalised)
            for i, lvl in enumerate(levels):
                penalise = model.NewBoolVar(f"day_{d}_penalise_lvl_{lvl}_snr")
                model.Add(penalise == flags[i].Not())
                high_priority_penalty.append(penalise * penalties[i])


    # === Phase 2 Constraints ===

    # 1. Preference satisfaction
    for n in nurse_names:
        prefs = prefs_by_nurse[n]
        satisfied_list = []

        for d in range(num_days):
            if d in prefs:
                s = prefs[d]
                sat = model.NewBoolVar(f'sat_{n}_{d}')
                model.Add(work[n, d, s] == 1).OnlyEnforceIf(sat)
                model.Add(work[n, d, s] == 0).OnlyEnforceIf(sat.Not())
                is_satisfied[(n, d)] = sat
                satisfied_list.append(sat)
                # Add penalty if preference not satisfied
                low_priority_penalty.append(sat.Not() * PREF_MISS_PENALTY)
            else:
                satisfied_const = model.NewConstant(0)
                is_satisfied[(n, d)] = satisfied_const
                satisfied_list.append(satisfied_const)

        total_satisfied[n] = model.NewIntVar(0, num_days, f'total_sat_{n}')
        model.Add(total_satisfied[n] == sum(satisfied_list))

    # 2. Fairness constraint on preference satisfaction gap
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

    # 3. Balance in number of each shift type assigned to nurse
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
    logger.info("🚀 Phase 1A: checking feasibility...")
    # 1. Tell the model to minimize penalty sum
    model.Maximize(sum(r.flag for r in hard_rules.values()))

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

    status1a = solver.Solve(model)
    total_hards = len(hard_rules)
    logger.info(total_hards)
    satisfied_hards = int(solver.ObjectiveValue())
    logger.info(satisfied_hards)
    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
    if satisfied_hards != total_hards or status1a not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        dropped = [r.message for r in hard_rules.values() if solver.Value(r.flag) == 0]
        error_msg = "❌ No feasible solution. Identified issues:\n\n"
        error_msg += "\n".join(f"     • {m}" for m in dropped)
        logger.info("⚠️ No feasible solution found with minimal constraints.")
        raise NoFeasibleSolutionError(error_msg)
    
    logger.info("✅ Feasible solution found with minimal constraints.")
    logger.info("🚀 Phase 1B: minimising penalties...")
    model.Add(sum(r.flag for r in hard_rules.values()) == total_hards)
    model.minimize(sum(high_priority_penalty))

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

    status1b = solver.Solve(model)
    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
    logger.info(f"High Priority Penalty Phase 1B: {solver.ObjectiveValue()}")
    logger.info(f"Low Priority Penalty Phase 1B: {solver.Value(sum(low_priority_penalty))}")

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
    # only run phase 2 if low priority penalty exists, which means shifts preferences exist
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
    violations = {"Double Shifts": []}
    if not min_weekly_hours_hard:
        violations["Low Hours Nurses"] = []
    if not am_coverage_min_hard:
        violations["Low AM Days"] = []
    if not am_senior_min_hard:
        violations["Low Senior AM Days"] = []
    metrics = {}
    has_shift_prefs = any(shift_preferences.values())
    if has_shift_prefs:
        metrics = {"Preference Unmet": [], "Fairness Gap": cached_gap if use_fallback or 'gap_pct' not in locals() else solver.Value(gap_pct)}

    for n in nurse_names:
        row = []
        minutes_per_week = [0] * num_weeks
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
                minutes_per_week[week_idx] += int(shift_durations[p])
                shift_counts[p] += 1

            pref = prefs_by_nurse[n].get(d)
            if pref is not None:
                if len(picked) == 1 and picked[0] == pref:
                    prefs_met += 1
                else:
                    prefs_unmet.append(f"{dates[d].strftime('%a %Y-%m-%d')} (wanted {SHIFT_LABELS[pref]})")

        if not min_weekly_hours_hard:
            for w in range(num_weeks):
                days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, num_days))
                if len(days) < DAYS_PER_WEEK:
                    continue  # skip incomplete weeks
                mc_count_week = len(mc_sets[n] & set(days))
                el_count_week = len(el_sets[n] & set(days))
                al_count_week = len(al_sets[n] & set(days))
                eff_pref_minutes = max(0, preferred_weekly_minutes - (mc_count_week + el_count_week + al_count_week) * avg_minutes)

                if minutes_per_week[w] < eff_pref_minutes:
                    violations["Low Hours Nurses"].append(f"{n} Week {w+1}: {round(minutes_per_week[w] / 60, 1)}h; pref {round(eff_pref_minutes / 60, 1)}")        
        
        if double_shift_days:
            violations["Double Shifts"].append(f"{n}: {'; '.join(double_shift_days)}")

        if prefs_unmet:
            metrics["Preference Unmet"].append(f"{n}: {'; '.join(prefs_unmet)}")

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
            summary_row[f"Hours_Week{w+1}"] = round(minutes_per_week[w] / 60, 1)
        summary_row.update({
            "Prefs_Met": prefs_met,
            "Prefs_Unmet": len(prefs_unmet),
            "Unmet_Details": "; ".join(prefs_unmet),
        })
        summary.append(summary_row)

    if not am_coverage_min_hard and not am_senior_min_hard:
        for d in range(num_days):
            if use_fallback:
                am_n = sum(cached_values[(n, d, 0)] for n in nurse_names)
                total_n = sum(cached_values[(n, d, s)] for n in nurse_names for s in range(shift_types))
                am_snr = sum(cached_values[(n, d, 0)] for n in senior_names)
            else:
                am_n = sum(solver.Value(work[n, d, 0]) for n in nurse_names)
                total_n = sum(solver.Value(work[n, d, s]) for n in nurse_names for s in range(shift_types))
                am_snr = sum(solver.Value(work[n, d, 0]) for n in senior_names)

            if not am_coverage_min_hard and total_n and am_n / total_n < (am_coverage_min_percent / 100):
                violations["Low AM Days"].append(f"{dates[d].strftime('%a %Y-%m-%d')} ({am_n/total_n:.0%})")
            if not am_senior_min_hard and am_n and am_snr / am_n < (am_senior_min_percent / 100):
                violations["Low Senior AM Days"].append(f"{dates[d].strftime('%a %Y-%m-%d')} (Seniors {am_snr/am_n:.0%})")

    logger.info("\n⚠️ Soft Constraint Violations Summary:")
    for key, items in violations.items():
        logger.info(f"🔸 {key}: {len(items) if isinstance(items, list) else items} cases")
        if isinstance(items, list):
            for item in sorted(items):
                logger.info(f"   - {item}")

    if has_shift_prefs:
        logger.info("\n📊 Preferences Satisfaction and Fairness Summary:")
        total_unmet = sum(s["Prefs_Unmet"] for s in summary)
        logger.info(f"🔸 Preference Unmet: {total_unmet} unmet preferences across {len(metrics['Preference Unmet'])} nurses")
        logger.info(f"🔸 Fairness Gap: {metrics['Fairness Gap']}%")
        for key, items in metrics.items():
            if isinstance(items, list):
                for item in sorted(items):
                    logger.info(f"   - {item}")

    logger.info("📁 Schedule and summary generated.")
    schedule_df = pd.DataFrame.from_dict(schedule, orient='index', columns=headers).reindex(og_nurse_names)
    summary_df = pd.DataFrame(summary).set_index("Nurse").reindex(og_nurse_names).reset_index()
    return schedule_df, summary_df, violations, metrics