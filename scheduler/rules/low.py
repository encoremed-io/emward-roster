from core.state import ScheduleState
from utils.constants import *
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set
import pandas as pd

"""
This module contains the low priority rules for the nurse scheduling problem.
"""


def preference_rule(model, state: ScheduleState):
    """
    Add constraints to the model to enforce the preference satisfaction of nurses.

    The soft constraint is that a nurse should work their preferred shift on each day.
    A penalty is incurred if a nurse does not work their preferred shift.

    The total number of satisfied preferences is also calculated for each nurse.
    """
    # Preference satisfaction
    is_satisfied = {}
    for n in state.nurse_names:
        prefs = state.prefs_by_nurse[n]
        satisfied_list = []

        for d in range(state.num_days):
            if d in prefs:
                # prefs[d] is be (shift_idx, timestamp)
                entry = prefs[d]
                if isinstance(entry, tuple):
                    s, ts = entry
                else:
                    s = entry
                sat = model.NewBoolVar(f"sat_{n}_{d}")
                model.Add(state.work[n, d, s] == 1).OnlyEnforceIf(sat)
                model.Add(state.work[n, d, s] == 0).OnlyEnforceIf(sat.Not())
                is_satisfied[(n, d)] = sat
                satisfied_list.append(sat)
                # Add penalty if preference not satisfied
                state.low_priority_penalty.append(sat.Not() * state.pref_miss_penalty)
            else:
                satisfied_const = model.NewConstant(0)
                is_satisfied[(n, d)] = satisfied_const
                satisfied_list.append(satisfied_const)

        state.total_satisfied[n] = model.NewIntVar(0, state.num_days, f"total_sat_{n}")
        model.Add(state.total_satisfied[n] == sum(satisfied_list))


def preference_rule_ts(model, state: ScheduleState):
    """
    Hybrid preference rule with dynamic scaling:
    - Strong penalty for missed preferences, scaled by number of prefs.
    - No blanket penalty for all non-preferences (keeps solver fast).
    - Soft constraint: each nurse should get at least 1 preference, but it's not hard.
    """

    from collections import defaultdict

    slot_reqs = defaultdict(list)
    requested_slots_by_nurse = {n: set() for n in state.nurse_names}

    # ✅ Ensure container exists to collect all preference-satisfaction BoolVars
    if not hasattr(state, "pref_sat_vars"):
        state.pref_sat_vars = []

    # Collect requests
    for n in state.nurse_names:
        for day, entries in state.prefs_by_nurse.get(n, {}).items():
            if isinstance(entries, tuple):
                entries = [entries]
            for shift_idx, ts in entries:
                slot_reqs[(day, shift_idx)].append((n, ts))
                requested_slots_by_nurse[n].add((day, shift_idx))

    sats_by_nurse = {n: [] for n in state.nurse_names}

    # Preferences with ranking
    for (day, shift), reqs in slot_reqs.items():
        reqs.sort(key=lambda x: x[1])  # earlier timestamp = higher priority
        ranks = {n: i for i, (n, _) in enumerate(reqs)}
        max_rank = max(ranks.values()) if ranks else 0

        for nurse, ts in reqs:
            sat = state.work[nurse, day, shift]
            rank = ranks[nurse]

            # ✅ Collect for diagnostics / stronger objectives
            state.pref_sat_vars.append(sat)

            # dynamic penalty: stronger if nurse has more prefs overall
            num_prefs = len(state.prefs_by_nurse.get(nurse, {}))
            penalty = (
                state.pref_miss_penalty
                * (max_rank - rank + 1)
                * max(1, num_prefs)
                * 100
            )

            state.low_priority_penalty.append((1 - sat) * penalty)
            sats_by_nurse[nurse].append(sat)

    # Track total satisfied + soft floor (scaled by number of prefs)
    for nurse, sat_list in sats_by_nurse.items():
        if sat_list:
            total = model.NewIntVar(0, len(sat_list), f"total_sat_{nurse}")
            model.Add(total == sum(sat_list))
            state.total_satisfied[nurse] = total

            # Soft floor: penalize heavily if nurse gets 0 prefs satisfied
            has_pref = model.NewBoolVar(f"has_pref_{nurse}")
            model.Add(total >= 1).OnlyEnforceIf(has_pref)
            model.Add(total == 0).OnlyEnforceIf(has_pref.Not())

            # Penalty grows with number of prefs
            state.low_priority_penalty.append(has_pref.Not() * len(sat_list) * 500)


def fairness_gap_rule(model, state: ScheduleState):
    """
    Fairness gap soft constraint.

    For each nurse, calculate the percentage of satisfied preferences
    (out of total preferences). prefs are stored as a dict:
    {pref_index: (shift_id, timestamp)} where pref_index = day index.
    """

    pct_sat = {}

    for n, prefs in state.prefs_by_nurse.items():
        count = len(prefs)
        if count > 0:
            satisfied_vars = []
            for d, (s, _) in prefs.items():
                # d = day index (pref key), s = shift index
                satisfied_vars.append(state.work[n, d, s])

            total_sat = model.NewIntVar(0, count, f"total_sat_{n}")
            model.Add(total_sat == sum(satisfied_vars))

            # percentage satisfied = (total_sat / count) * 100
            p = model.NewIntVar(0, 100, f"pct_sat_{n}")
            model.Add(p * count == total_sat * 100)
            pct_sat[n] = p
        else:
            pct_sat[n] = None

    valid_pcts = [p for p in pct_sat.values() if p is not None]
    if not valid_pcts:
        return

    min_pct = model.NewIntVar(0, 100, "min_pct")
    max_pct = model.NewIntVar(0, 100, "max_pct")
    model.AddMinEquality(min_pct, valid_pcts)
    model.AddMaxEquality(max_pct, valid_pcts)

    gap_pct = model.NewIntVar(0, 100, "gap_pct")
    model.Add(gap_pct == max_pct - min_pct)
    state.gap_pct = gap_pct

    diff = model.NewIntVar(-100, 100, "diff_gap")
    model.Add(diff == gap_pct - state.fairness_gap_threshold)

    over_gap = model.NewIntVar(0, 100, "over_gap")
    model.AddMaxEquality(over_gap, [diff, model.NewConstant(0)])
    state.low_priority_penalty.append(over_gap * state.fairness_gap_penalty)


def shift_balance_rule(model, state: ScheduleState):
    """
    Hybrid shift balance rule:
    - Hard constraint: prevent extreme imbalance (distribution gap cannot exceed a max cap).
    - Soft penalty: still encourage tighter balance when gap is larger than a preferred threshold.
    """

    if state.shift_balance:
        counts = {}
        for n in state.nurse_names:
            for s in range(state.shift_types):
                C = model.NewIntVar(0, state.num_days, f"count_{n}_s{s}")
                model.Add(C == sum(state.work[n, d, s] for d in range(state.num_days)))
                counts[(n, s)] = C

        for n in state.nurse_names:
            c_vars = [counts[(n, s)] for s in range(state.shift_types)]
            minC = model.NewIntVar(0, state.num_days, f"min_count_{n}")
            maxC = model.NewIntVar(0, state.num_days, f"max_count_{n}")

            model.AddMinEquality(minC, c_vars)
            model.AddMaxEquality(maxC, c_vars)

            distribution_gap = model.NewIntVar(0, state.num_days, f"gap_{n}")
            model.Add(distribution_gap == maxC - minC)

            # ---------------- HARD CAP ----------------
            # Define a maximum allowed imbalance (e.g., 3)
            hard_cap = getattr(
                state, "shift_imbalance_hard_cap", state.shift_imbalance_threshold + 1
            )
            model.Add(distribution_gap <= hard_cap)

            # ---------------- SOFT PENALTY ----------------
            # Apply penalty if gap exceeds the softer threshold
            diff = model.NewIntVar(
                -(state.num_days + state.shift_imbalance_threshold),
                state.num_days + state.shift_imbalance_threshold,
                f"diff_{n}",
            )
            model.Add(diff == distribution_gap - state.shift_imbalance_threshold)

            over_gap = model.NewIntVar(
                0, state.num_days + state.shift_imbalance_threshold, f"over_gap_{n}"
            )
            model.AddMaxEquality(over_gap, [diff, model.NewConstant(0)])
            state.low_priority_penalty.append(over_gap * state.shift_imbalance_penalty)
