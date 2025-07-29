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
                sat = model.NewBoolVar(f'sat_{n}_{d}')
                model.Add(state.work[n, d, s] == 1).OnlyEnforceIf(sat)
                model.Add(state.work[n, d, s] == 0).OnlyEnforceIf(sat.Not())
                is_satisfied[(n, d)] = sat
                satisfied_list.append(sat)
                # Add penalty if preference not satisfied
                state.low_priority_penalty.append(sat.Not() * PREF_MISS_PENALTY)
            else:
                satisfied_const = model.NewConstant(0)
                is_satisfied[(n, d)] = satisfied_const
                satisfied_list.append(satisfied_const)

        state.total_satisfied[n] = model.NewIntVar(0, state.num_days, f'total_sat_{n}')
        model.Add(state.total_satisfied[n] == sum(satisfied_list))


def preference_rule_ts(model, state: ScheduleState):
    """
    Add constraints to the model to enforce the preference satisfaction of nurses, taking into account timestamps.

    The soft constraint is that a nurse should work their preferred shift on each day, with earlier timestamps given priority.
    A penalty is incurred if a nurse does not work their preferred shift.

    The total number of satisfied preferences is also calculated for each nurse.

    Preferences are prioritized on a first-come, first-serve basis based on the timestamp.
    Tied timestamps have the same rank. Penalty is calculated as:
        penalty = PREF_MISS_PENALTY * (max_rank - rank + 1)
    """
    from collections import defaultdict
    from itertools import groupby

    # 1) bucket all requests by (day, shift)
    slot_reqs: Dict[Tuple[int, int], List[Tuple[str, pd.Timestamp]]] = defaultdict(list)
    for n in state.nurse_names:
        for day, entry in state.prefs_by_nurse[n].items():
            shift_idx, ts = entry  # timestamp is guaranteed not None
            slot_reqs[(day, shift_idx)].append((n, ts))

    # 2) prepare per‐nurse list for aggregation
    sats_by_nurse: Dict[str, List[Any]] = {n: [] for n in state.nurse_names}

    # 3) for each slot, apply dense ranking and assign penalties
    for (day, shift), reqs in slot_reqs.items():
        # Sort by timestamp
        reqs.sort(key=lambda x: x[1])  # ascending

        # Dense ranking by timestamp
        ranks: Dict[str, int] = {}
        grouped = groupby(reqs, key=lambda x: x[1])
        for rank, (_, group) in enumerate(grouped):
            for nurse, _ in group:
                ranks[nurse] = rank

        max_rank = max(ranks.values())

        for nurse, ts in reqs:
            sat = model.NewBoolVar(f"pref_sat_{nurse}_{day}_{shift}")
            model.Add(state.work[nurse, day, shift] == 1).OnlyEnforceIf(sat)
            model.Add(state.work[nurse, day, shift] == 0).OnlyEnforceIf(sat.Not())

            rank = ranks[nurse]
            penalty = PREF_MISS_PENALTY * (max_rank - rank + 1)
            state.low_priority_penalty.append(sat.Not() * penalty)
            sats_by_nurse[nurse].append(sat)
            # import logging
            # logging.debug(f"Penalty assigned → Nurse={nurse}, Day={day}, Shift={shift}, Rank={rank}, Penalty={penalty}")

    # 4) total up per‐nurse satisfied prefs
    for nurse, sat_list in sats_by_nurse.items():
        total = model.NewIntVar(0, len(sat_list), f"total_sat_{nurse}")
        model.Add(total == sum(sat_list))
        state.total_satisfied[nurse] = total


def fairness_gap_rule(model, state: ScheduleState):
    """
    Fairness gap constraint: add a penalty proportional to the difference between the highest and lowest percentage of satisfied preferences among all nurses.

    The penalty is only incurred if the gap is greater than or equal to the threshold of 60.

    The gap is calculated as (max_pct - min_pct).

    The penalty is proportional to the distance from 60, i.e. (gap_pct - 60) if gap_pct >= 60, else 0.

    The total penalty is stored in state.low_priority_penalty.
    """
    # Fairness gap
    pct_sat = {}
    for n, prefs in state.prefs_by_nurse.items():
        count = len(prefs)
        if count > 0:
            p = model.NewIntVar(0, 100, f"pct_sat_{n}")
            pct_sat[n] = p
            model.Add(p * count == state.total_satisfied[n] * 100)
        else:
            pct_sat[n] = None

    valid_pcts = [p for p in pct_sat.values() if p is not None]
    if not valid_pcts:
        return
    else:
        min_pct = model.NewIntVar(0, 100, "min_pct")
        max_pct = model.NewIntVar(0, 100, "max_pct")
        model.AddMinEquality(min_pct, valid_pcts)
        model.AddMaxEquality(max_pct, valid_pcts)

        gap_pct = model.NewIntVar(0, 100, "gap_pct")
        model.Add(gap_pct == max_pct - min_pct)

        state.gap_pct = gap_pct     # Only store gap_pct if we have valid percentages, else remain None
        diff = model.NewIntVar(-100, 100, f"diff_{n}")
        model.Add(diff == gap_pct - FAIRNESS_GAP_THRESHOLD)

        # Start penalise fairness when gap_pct >= 60 based on distance from 60
        over_gap  = model.NewIntVar(0, 100, "over_gap")
        model.AddMaxEquality(over_gap, [diff, model.NewConstant(0)])
        state.low_priority_penalty.append(over_gap * FAIRNESS_GAP_PENALTY)


def shift_balance_rule(model, state: ScheduleState):
    """
    Shift balance rule.

    If shift_balance is True, this adds a rule to the model to penalise uneven shift distribution for each nurse.

    The penalty is proportional to the gap between the maximum and minimum number of shifts assigned to each nurse.

    The gap is calculated as (max_count - min_count), where max_count and min_count are the maximum and minimum counts of shifts assigned to the nurse.

    The penalty is only incurred if the gap is greater than or equal to the threshold of 2.
    The penalty is proportional to the distance from 2, i.e. (gap - 2) if gap >= 2, else 0.

    The total penalty is stored in state.low_priority_penalty.
    """
    if state.shift_balance:
        # Precompute counts just once
        counts = {}
        for n in state.nurse_names:
            for s in range(state.shift_types):
                C = model.NewIntVar(0, state.num_days, f"count_{n}_s{s}")
                model.Add(C == sum(state.work[n, d, s] for d in range(state.num_days)))
                counts[(n, s)] = C

        # For each nurse, build min/max/gap
        for n in state.nurse_names:
            c_vars = [counts[(n, s)] for s in range(state.shift_types)]
            minC = model.NewIntVar(0, state.num_days, f"min_count_{n}")
            maxC = model.NewIntVar(0, state.num_days, f"max_count_{n}")

            model.AddMinEquality(minC, c_vars)
            model.AddMaxEquality(maxC, c_vars)

            distribution_gap = model.NewIntVar(0, state.num_days, f"gap_{n}")
            model.Add(distribution_gap == maxC - minC)

            diff = model.NewIntVar(-state.num_days, state.num_days, f"diff_{n}")
            model.Add(diff == distribution_gap - SHIFT_IMBALANCE_THRESHOLD)

            # soft penalise when distribution_gap >= 2 based on distance from 2
            over_gap = model.NewIntVar(0, state.num_days, f"over_gap_{n}")
            model.AddMaxEquality(over_gap, [diff, model.NewConstant(0)])
            state.low_priority_penalty.append(over_gap * SHIFT_IMBALANCE_PENALTY)
