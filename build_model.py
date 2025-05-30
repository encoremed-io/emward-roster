import pandas as pd
from ortools.sat.python import cp_model
from datetime import timedelta, date as dt_date
from typing import Optional, Dict, Tuple, Set
from collections import defaultdict
from pathlib import Path
import logging
import json

LOG_PATH = Path(__file__).parent / "schedule_run.log"

# grab our module’s logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only add the handler once
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

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

PREFERRED_WEEKLY_HOURS = constants["PREFERRED_WEEKLY_HOURS"]
MIN_ACCEPTABLE_WEEKLY_HOURS = constants["MIN_ACCEPTABLE_WEEKLY_HOURS"]
PREF_HOURS_PENALTY = constants["PREF_HOURS_PENALTY"]
MIN_HOURS_PENALTY = constants["MIN_HOURS_PENALTY"]

DOUBLE_SHIFT_PENALTY = constants["DOUBLE_SHIFT_PENALTY"]

AM_COVERAGE_MIN_PERCENT = constants["AM_COVERAGE_MIN_PERCENT"]
AM_COVERAGE_PENALTIES = constants["AM_COVERAGE_PENALTIES"]

PREF_MISS_PENALTY = constants["PREF_MISS_PENALTY"]
FAIRNESS_GAP_PENALTY = constants["FAIRNESS_GAP_PENALTY"]
FAINRESS_GAP_THRESHOLD = constants["FAIRNESS_GAP_THRESHOLD"]

def load_nurse_profiles(path='data/nurse_profiles.xlsx') -> pd.DataFrame:
    df = pd.read_excel(path)
    df['Name'] = df['Name'].str.strip().str.upper()
    return df


def load_shift_preferences(path='data/nurse_preferences.xlsx') -> pd.DataFrame:
    df = pd.read_excel(path)
    df.rename(columns={df.columns[0]: 'Name'}, inplace=True)
    df.set_index('Name', inplace=True)
    # parse date columns
    cleaned = []
    for col in df.columns:
        # assume format contains YYYY-MM-DD
        dt = pd.to_datetime(str(col).strip().split()[-1], format="%Y-%m-%d").date()
        cleaned.append(dt)
    df.columns = cleaned
    df.index = df.index.str.strip().str.upper()
    return df


def validate_nurse_data(profiles_df: pd.DataFrame, preferences_df: pd.DataFrame):
    profile_names = set(profiles_df['Name'].str.strip())
    preference_names = set(preferences_df.index.str.strip())
    missing = profile_names - preference_names
    extra = preference_names - profile_names
    if missing or extra:
        return missing, extra
    return None, None  # valid


def build_schedule_model(profiles_df: pd.DataFrame,
                         preferences_df: pd.DataFrame,
                         start_date: pd.Timestamp | dt_date,
                         num_days: int,
                         rl_assignment=None,
                         fixed_assignments: Optional[Dict[Tuple[str,int], str]] = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a nurse schedule satisfying hard constraints and optimizing soft preferences.
    Returns a schedule DataFrame and a summary DataFrame.
    """

    # === Model setup ===
    logger.info("📋 Building model...")
    model = cp_model.CpModel()
    nurses = profiles_df.to_dict(orient='records')
    nurse_names = [n['Name'] for n in nurses]
    senior_names = {n['Name'] for n in nurses if n['Title'] == 'Senior'}    # Assume senior nurses have ≥3 years experience
    shift_str_to_idx = {label.upper(): i for i, label in enumerate(SHIFT_LABELS)}

    if isinstance(start_date, pd.Timestamp):
        date_start: dt_date = start_date.date()
    else:
        date_start = start_date

    # === Normalise EL days ===
    if fixed_assignments is None:
        fixed_assignments = {}
    else:
        # ensure keys are uppercase names and valid indices
        cleaned = {}
        for (nurse, day_idx), shift in fixed_assignments.items():
            name = nurse.strip().upper()
            if name not in nurse_names:
                raise ValueError(f"Unknown nurse in fixed_assignments: {nurse}")
            if not (0 <= day_idx < num_days):
                raise ValueError(f"Day index out of range for {nurse}: {day_idx}")
            cleaned[(name, day_idx)] = shift.strip().upper()
        fixed_assignments = cleaned

    # === Preferences and MC days ===
    shift_preferences = {}
    mc_days = {}

    for nurse, row in preferences_df.iterrows():
        shift_preferences[nurse] = {}
        mc_days[nurse] = set()
        for label, val in row.items():
            if isinstance(label, pd.Timestamp):
                d = label.date()
            elif isinstance(label, dt_date):
                d = label
            else:
                d = pd.to_datetime(str(label)).date()

            day_idx = (d - date_start).days
            if not pd.notna(val) or not (0 <= day_idx < num_days):
                continue
            val = str(val).strip().upper()
            if val == 'MC':
                mc_days[nurse].add(day_idx)
            elif val in shift_str_to_idx:
                shift_preferences[nurse][day_idx] = shift_str_to_idx[val]

    weekend_days = [
        (i, i + 1) for i in range(num_days - 1)
        if (date_start + timedelta(days=i)).weekday() == 5
    ]

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

    min_sat = model.NewIntVar(0, num_days, "min_satisfaction")
    max_sat = model.NewIntVar(0, num_days, "max_satisfaction")

    # === Hard Constraints ===

    # === Fix EL as no work ===
    el_days: Set[int] = set()
    el_days_per_nurse: Dict[str, Set[int]] = defaultdict(set)

    for (nurse, day_idx), shift_label in fixed_assignments.items():
        label = shift_label.upper()

        if label in {"EL", "MC", "REST"}:
            # Block all shifts
            for s in range(shift_types):
                model.Add(work[nurse, day_idx, s] == 0)
            # Record EL
            if label == "EL":
                el_days.add(day_idx)
                el_days_per_nurse[nurse].add(day_idx)

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
    # If no EL, each nurse can work at most one shift per day
    # If EL, allow 2 shifts per nurse if cannot satisfy other hard constraints for that day
    two_shifts = {}  

    for n in nurse_names:
        for d in range(num_days):
            if d not in el_days:
            # no EL here: enforce original rule
                model.AddAtMostOne(work[n, d, s] for s in range(shift_types))
            else:
            # EL day: allow either 1 or 2 shifts
                ts = model.NewBoolVar(f"two_shifts_{n}_{d}")
                two_shifts[(n, d)] = ts

                # If ts==False → sum_s work ≤ 1
                model.Add(sum(work[n, d, s] for s in range(shift_types)) <= 1).OnlyEnforceIf(ts.Not())
                # If ts==True  → sum_s work == 2
                model.Add(sum(work[n, d, s] for s in range(shift_types)) == 2).OnlyEnforceIf(ts)

                # If double shift, apply penalty
                high_priority_penalty.append(ts * DOUBLE_SHIFT_PENALTY)

    # 2. Each nurse works <= 42 hours/week (hard), adjustable based on MC; ideally min 40 (soft), at least 30 (hard)
    for n in nurse_names:
        for w in range(2):
            days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, num_days))
            weekly_hours = sum(work[n, d, s] * SHIFT_HOURS[s] for d in days for s in range(shift_types))

            mc_count = sum(1 for d in days if d in mc_days.get(n, set()))
            el_count = sum(1 for d in days if d in el_days_per_nurse.get(n, set()))

            adjustment = (mc_count + el_count) * AVG_HOURS                      # MC & EL hours deducted from max/pref/min hours
            eff_max_hours = max(0, MAX_WEEKLY_HOURS - adjustment)               # <= 42 - x
            eff_pref_hours = max(0, PREFERRED_WEEKLY_HOURS - adjustment)        # >= 40 - x
            eff_min_hours = max(0, MIN_ACCEPTABLE_WEEKLY_HOURS - adjustment)    # >= 30 - x

            model.Add(weekly_hours <= eff_max_hours)
            model.Add(weekly_hours >= eff_min_hours)

            # Soft preferences on hours
            if eff_pref_hours > eff_min_hours:
                min_pref = model.NewBoolVar(f'pref_{n}_w{w}')
                model.Add(weekly_hours >= eff_pref_hours).OnlyEnforceIf(min_pref)    # prefer 40 - x
                model.Add(weekly_hours < eff_pref_hours).OnlyEnforceIf(min_pref.Not())

                high_priority_penalty.append(min_pref.Not() * PREF_HOURS_PENALTY)

    # 3. Each shift must have at least 4 nurses and at least 1 senior
    for d in range(num_days):
        for s in range(shift_types):
            model.Add(sum(work[n, d, s] for n in nurse_names) >= MIN_NURSES_PER_SHIFT)
            model.Add(sum(work[n, d, s] for n in senior_names) >= MIN_SENIORS_PER_SHIFT)

    # 4. Weekend work requires rest on the same day next weekend
    for n in nurse_names:
        for d1, d2 in weekend_days:
            for day in (d1, d2):
                if day + 7 < num_days:
                    model.Add(sum(work[n, day, s] for s in range(shift_types)) <=
                              1 - sum(work[n, day + 7, s] for s in range(shift_types)))

    # 5. MC days: cannot assign any shift
    for n in nurse_names:
        for d in mc_days.get(n, []):
            for s in range(shift_types):
                model.Add(work[n, d, s] == 0)

    # 6. Max 2 MC days/week and no more than 2 consecutive MC days
    for n in nurse_names:
        mc_set = mc_days.get(n, set())

        for w in range(2):
            days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, num_days))
            mc_in_week = sum(1 for d in days if d in mc_set)
            if mc_in_week > MAX_MC_DAYS_PER_WEEK:
                raise ValueError(f"❌ Nurse {n} has more than {MAX_MC_DAYS_PER_WEEK} MCs in week {w+1}.")

        sorted_mc = sorted(mc_set)
        for i in range(len(sorted_mc) - 2):
            if sorted_mc[i + 2] - sorted_mc[i] == 2:
                raise ValueError(f"❌ Nurse {n} has more than 2 consecutive MC days: {sorted_mc[i]}, {sorted_mc[i+1]}, {sorted_mc[i+2]}.")

    # === Soft Constraints ===

    # 1. AM coverage per day should be >=60%, ideally
    for d in range(num_days):
        total_shifts = sum(work[n, d, s] for n in nurse_names for s in range(shift_types))
        am_shifts = sum(work[n, d, 0] for n in nurse_names)
        am_seniors = sum(work[n, d, 0] for n in nurse_names if n in senior_names)

        level1 = AM_COVERAGE_MIN_PERCENT          # typically 60
        level2 = max(level1 - 10, 0)              # 50
        level3 = max(level1 - 20, 0)              # 40

        # Coverage level flags
        level1_ok = model.NewBoolVar(f'day_{d}_am_level1')
        level2_ok = model.NewBoolVar(f'day_{d}_am_level2')
        level3_ok = model.NewBoolVar(f'day_{d}_am_level3')

        # Soft targets
        model.Add(am_shifts * 100 >= level1 * total_shifts).OnlyEnforceIf(level1_ok)
        model.Add(am_shifts * 100 <  level1 * total_shifts).OnlyEnforceIf(level1_ok.Not())

        model.Add(am_shifts * 100 >= level2 * total_shifts).OnlyEnforceIf([level1_ok.Not(), level2_ok])
        model.Add(am_shifts * 100 <  level2 * total_shifts).OnlyEnforceIf([level1_ok.Not(), level2_ok.Not()])

        model.Add(am_shifts * 100 >= level3 * total_shifts).OnlyEnforceIf([level1_ok.Not(), level2_ok.Not(), level3_ok])
        model.Add(am_shifts * 100 <  level3 * total_shifts).OnlyEnforceIf([level1_ok.Not(), level2_ok.Not(), level3_ok.Not()])

        # Hard fallback condition
        all_levels_failed = model.NewBoolVar(f'day_{d}_all_levels_failed')
        model.AddBoolAnd([level1_ok.Not(), level2_ok.Not(), level3_ok.Not()]).OnlyEnforceIf(all_levels_failed)
        model.AddBoolOr([level1_ok, level2_ok, level3_ok]).OnlyEnforceIf(all_levels_failed.Not())

        # Explicit PM and Night shift counts
        pm_shift_nurses = sum(work[n, d, 1] for n in nurse_names)
        pm_shift_seniors = sum(work[n, d, 1] for n in nurse_names if n in senior_names)

        night_shift_nurses = sum(work[n, d, 2] for n in nurse_names)
        night_shift_seniors = sum(work[n, d, 2] for n in nurse_names if n in senior_names)

        # Enforce AM > PM and AM > Night if all levels fail (hard constraint)
        model.Add(am_shifts > pm_shift_nurses).OnlyEnforceIf(all_levels_failed)
        model.Add(am_shifts > night_shift_nurses).OnlyEnforceIf(all_levels_failed)
        model.Add(am_seniors > pm_shift_seniors).OnlyEnforceIf(all_levels_failed)
        model.Add(am_seniors > night_shift_seniors).OnlyEnforceIf(all_levels_failed)

        # Penalties for failing soft levels
        high_priority_penalty.append(level1_ok.Not() * AM_COVERAGE_PENALTIES[0])

        level2_penalty_cond = model.NewBoolVar(f'day_{d}_level2_penalty')
        model.AddBoolAnd([level1_ok.Not(), level2_ok.Not()]).OnlyEnforceIf(level2_penalty_cond)
        high_priority_penalty.append(level2_penalty_cond * AM_COVERAGE_PENALTIES[1])

        level3_penalty_cond = model.NewBoolVar(f'day_{d}_level3_penalty')
        model.AddBoolAnd([level1_ok.Not(), level2_ok.Not(), level3_ok.Not()]).OnlyEnforceIf(level3_penalty_cond)
        high_priority_penalty.append(level3_penalty_cond * AM_COVERAGE_PENALTIES[2])

    # 2. Preference satisfaction
    for n in nurse_names:
        prefs = shift_preferences.get(n, {})
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

    # 3. Fairness constraint on preference satisfaction gap
    pref_count = {
        n: len(shift_preferences.get(n, {}))
        for n in nurse_names
    }

    pct_sat = {}
    for n, count in pref_count.items():
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
        model.AddMaxEquality(over_gap, [gap_pct - FAINRESS_GAP_THRESHOLD, 0])
        low_priority_penalty.append(gap_pct * FAIRNESS_GAP_PENALTY)

    # === Add RL assignment hints (warm start) ===
    if rl_assignment is not None:
        N = len(nurse_names)
        D = num_days
        ND = N * D
        NDS = ND * 3       # 3 types of shift (AM, PM, Night)

        # container for all the (n,d,s) triples we want to turn on
        hinted_ones: set[tuple[str,int,int]] = set()
        hints: list[tuple[tuple[str,int],int]]   = []

        # Case A: dict of (n,d) -> s
        if isinstance(rl_assignment, dict):
            hints = list(rl_assignment.items())

        # Case B: flat list of shift‐indices length ND
        elif isinstance(rl_assignment, list) and len(rl_assignment) == ND:
            for idx, s in enumerate(rl_assignment):
                n_idx, d = divmod(idx, D)
                n = nurse_names[n_idx]
                hints.append(((n, d), s))

        # Case C: flat list of 0/1 per (n,d,s), length NDS
        elif isinstance(rl_assignment, list) and len(rl_assignment) == NDS:
            for idx, bit in enumerate(rl_assignment):
                if bit:
                    n_idx = idx // (D * 3)
                    rem   = idx %  (D * 3)
                    d     = rem // 3
                    s     = rem %  3
                    n     = nurse_names[n_idx]
                    hinted_ones.add((n, d, s))
            # build hints list from the ones
            hints = [(((n, d), s)) for (n, d, s) in hinted_ones]

        else:
            raise ValueError(
                f"rl_assignment must be either:\n"
                f" • dict[(n,d)->s]\n"
                f" • list of length {ND} (one shift index per nurse/day)\n"
                f" • list of length {NDS} (one 0/1 per nurse/day/shift)\n"
                f"Got {type(rl_assignment)} of length "
                f"{len(rl_assignment) if isinstance(rl_assignment, list) else 'N/A'}"
            )

        # now apply the “1” hints:
        for (n, d), s in hints:
            # hint that work[n,d,s] should be 1
            model.AddHint(work[n, d, s], 1)
            hinted_ones.add((n, d, s))

        # and apply explicit “0” hints on everything else
        # for n in nurse_names:
        #     for d in range(num_days):
        #         for s in range(shift_types):
        #             if (n, d, s) not in hinted_ones:
        #                 model.AddHint(work[n, d, s], 0)


    # === Objective ===
    # === Phase 1: minimize total penalties ===
    logger.info("🚀 Phase 1: minimizing penalties…")
    # 1. Tell the model to minimize penalty sum
    model.Minimize(sum(high_priority_penalty))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0  # tunable
    solver.parameters.random_seed = 42
    solver.parameters.relative_gap_limit = 0.01

    status1 = solver.Solve(model)
    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
    logger.info(f"High Priority Penalty Phase 1: {solver.ObjectiveValue()}")
    logger.info(f"Low Priority Penalty Phase 1: {solver.Value(sum(low_priority_penalty))}")
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("❌ No feasible solution even for penalties‐only!")

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
            pref = shift_preferences.get(n, {}).get(d)
            if pref is not None and len(picked) == 1 and pref in picked:
                cached_total_prefs_met += 1

    cached_gap = solver.Value(gap_pct) if valid_pcts else "N/A"
    high1 = solver.ObjectiveValue()
    best_penalty = solver.ObjectiveValue() + solver.Value(sum(low_priority_penalty))
    logger.info(f"▶️ Phase 1 complete: best total penalty = {best_penalty}; best fairness gap = {cached_gap}")

    # === Phase 2: maximize preferences under that penalty bound ===
    # only run phase 2 if shift preferences exist
    if any(shift_preferences.values()):
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

        # 4. Re-solve (you can reset your time budget)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 180.0
        solver.parameters.random_seed = 42
        solver.parameters.relative_gap_limit = 0.01
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
                    pref = shift_preferences.get(n, {}).get(d)
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
    schedule = {}
    summary = []
    violations = {"Low_AM_Days": [], "Low_Hours_Nurses": [], "Preference_Unmet": [], "Fairness_Gap": cached_gap if use_fallback or 'gap_pct' not in locals() else solver.Value(gap_pct)}

    for n in nurse_names:
        row = []
        hours_w1 = hours_w2 = 0
        counts = [0, 0, 0]
        prefs_met = 0
        prefs_unmet = []

        for d in range(num_days):
            picked = []
            
            if d in mc_days.get(n, set()):
                shift = "MC"
            elif (n, d) in fixed_assignments and fixed_assignments[(n, d)].upper() == "EL":
                shift = "EL"
            else:
                if use_fallback:
                    picked = [s for s in range(shift_types) if cached_values[(n, d, s)]]
                else:
                    picked = [s for s in range(shift_types) if solver.Value(work[n, d, s])]
                
                match(len(picked)):
                    case 0:
                        shift = "Rest"
                    case 1:
                        shift = SHIFT_LABELS[picked[0]]
                    case 2:
                        first, second = sorted(picked)
                        shift = f"{SHIFT_LABELS[first]}/{SHIFT_LABELS[second]}*"
                    case _:
                        shift = "OVER*"
            row.append(shift)

            for p in picked:
                hours = SHIFT_HOURS[p]
                if d < DAYS_PER_WEEK:
                    hours_w1 += hours
                else:
                    hours_w2 += hours
                counts[p] += 1

            pref = shift_preferences.get(n, {}).get(d)
            if pref is not None:
                if len(picked) == 1 and picked[0] == pref:
                    prefs_met += 1
                else:
                    prefs_unmet.append(f"{dates[d].strftime('%a %Y-%m-%d')} (wanted {SHIFT_LABELS[pref]})")

        for w, (begin, end) in enumerate([(0, 7), (7, num_days)], 1):
            days = range(begin, end)
            hours_worked = hours_w1 if w == 1 else hours_w2
            mc_count_week = mc_days.get(n, set()).intersection(days)
            el_count_week = el_days_per_nurse.get(n, set()).intersection(days)
            eff_pref_hours = max(0, PREFERRED_WEEKLY_HOURS - (len(mc_count_week | el_count_week) * AVG_HOURS))

            if hours_worked < eff_pref_hours:
                violations["Low_Hours_Nurses"].append(f"{n} Week {w}: {hours_worked}h; pref {eff_pref_hours}")

        if prefs_unmet:
            violations["Preference_Unmet"].append(f"{n}: {'; '.join(prefs_unmet)}")

        schedule[n] = row
        summary.append({
            'Nurse': n,
            'Hours_Week1': hours_w1,
            'Hours_Week2': hours_w2,
            'AM': counts[0],
            'PM': counts[1],
            'Night': counts[2],
            'Rest': row.count("Rest"),
            'MC_Days': len(mc_days.get(n, [])),
            'Prefs_Met': prefs_met,
            'Prefs_Unmet': len(prefs_unmet),
            'Unmet_Details': "; ".join(prefs_unmet)
        })

    for d in range(num_days):
        if use_fallback:
            am = sum(cached_values[(n, d, 0)] for n in nurse_names)
            total = sum(cached_values[(n, d, s)] for n in nurse_names for s in range(shift_types))
        else:
            am = sum(solver.Value(work[n, d, 0]) for n in nurse_names)
            total = sum(solver.Value(work[n, d, s]) for n in nurse_names for s in range(shift_types))

        if total and am / total < 0.6:
            violations["Low_AM_Days"].append(f"{dates[d].strftime('%a %Y-%m-%d')} ({am/total:.0%})")

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
    return pd.DataFrame.from_dict(schedule, orient='index', columns=headers), pd.DataFrame(summary)