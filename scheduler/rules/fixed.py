from core.state import ScheduleState
from utils.constants import *
from exceptions.custom_errors import InvalidMCError, ConsecutiveMCError
from utils.shift_utils import normalise_date
import pandas as pd
"""
This module contains the rules for handling fixed assignments and previous schedules, leave days and training shifts for the nurse scheduling problem.
"""

def handle_fixed_assignments(model, state: ScheduleState):
    """
    Add constraints to the model based on the fixed assignments.

    Multiple cases to handle:
    1. If the fixed assignment is a no-work label (MC, REST, AL, EL), block all shifts for that day.
    2. If the fixed assignment is a double-shift (e.g. "AM/PM*"), force both component shifts on and all other shifts off.
    3. If the fixed assignment is a single shift (e.g. "AM"), force that shift on and all other shifts off.

    We also record any MC/EL overrides in the state, so that we can check them later.

    :param model: The model to add constraints to.
    :param state: The ScheduleState instance that contains the fixed assignments.
    """
    for (nurse, day_idx), shift_label in state.fixed_assignments.items():
        label = shift_label.strip().upper()

        # Fix REST, MC, EL, AL, TR as no work
        if label in NO_WORK_LABELS:
            # Block all shifts
            for s in range(state.shift_types):
                model.Add(state.work[nurse, day_idx, s] == 0)
            # Record MC overrides
            if label == "MC":
                state.mc_sets[nurse].add(day_idx)
            if label == "AL":
                state.al_sets[nurse].add(day_idx)
            # EL already recorded in el_sets

        # handle double-shifts, e.g. "AM/PM*"
        elif "/" in label:
            # remove any trailing "*" and split
            parts = label.rstrip("*").split("/")
            # validate
            try:
                idxs = [ state.shift_str_to_idx[p] for p in parts ]
            except KeyError as e:
                raise ValueError(f"Unknown shift part '{e.args[0]}' in double-shift '{label}' for {nurse}")
            # force both component shifts on, others off
            for s in idxs:
                model.Add(state.work[nurse, day_idx, s] == 1)
            for other_s in set(range(state.shift_types)) - set(idxs):
                model.Add(state.work[nurse, day_idx, other_s] == 0)

        # Force that one shift and turn off the others
        else:
            if label not in state.shift_str_to_idx:
                raise ValueError(f"Unknown shift '{label}' for {nurse}")
            s = state.shift_str_to_idx[label]
            model.Add(state.work[nurse, day_idx, s] == 1)
            for other_s in (set(range(state.shift_types)) - {s}):
                model.Add(state.work[nurse, day_idx, other_s] == 0)


def leave_rules(model, state: ScheduleState):
    """ Adds leave rules to the model """
    define_leave(model, state)
    max_weekly_leave(state)
    max_consecutive_leave(state)


def define_leave(model, state: ScheduleState):
    """ Ensure that no shifts are assigned on days marked as Leave (MC/EL/AL) """
    # MC/AL days: cannot assign any shift -> Leave means no work
    for n in state.nurse_names:
        for d in state.mc_sets[n] | state.al_sets[n]:
            model.Add(sum(state.work[n, d, s] for s in range(state.shift_types)) == 0)


def max_weekly_leave(state: ScheduleState):
    """ Ensure that no nurse has more than 2 MC days in a week """
    # Max 2 MC days per week
    for n in state.nurse_names:
        mc = state.mc_sets[n]
        num_weeks = (state.num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK

        for w in range(num_weeks):
            days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days))
            mc_in_week = [d for d in days if d in mc]
            if len(mc_in_week) > MAX_MC_DAYS_PER_WEEK:
                raise InvalidMCError(
                    f"❌ Nurse {n} has more than {MAX_MC_DAYS_PER_WEEK} MCs in week {w+1}.\n"
                    f"Days: {sorted(mc_in_week)}"
                )
            

def max_consecutive_leave(state: ScheduleState):
    """ Ensure that no nurse has more than 2 consecutive MC days """
    # No more than 2 consecutive MC days
    for n in state.nurse_names:
        sorted_mc = sorted(state.mc_sets[n])

        for i in range(len(sorted_mc) - MAX_CONSECUTIVE_MC):
            if sorted_mc[i + 2] - sorted_mc[i] == MAX_CONSECUTIVE_MC:
                raise ConsecutiveMCError(
                    f"❌ Nurse {n} has more than 2 consecutive MC days: "
                    f"{sorted_mc[i]}, {sorted_mc[i+1]}, {sorted_mc[i+2]}"
                )
            

def training_shift_rules(model, state: ScheduleState):
    """ Adds training shift rules to the model """
    define_training_shifts(model, state)


def define_training_shifts(model, state: ScheduleState):
    """ Ensure that if a nurse is training on a shift, they cannot work that shift that day. """
    # Cannot assign same shift to nurse when they are training on that shift
    for n in state.nurse_names:
        for d, s in state.training_by_nurse.get(n, {}).items():
            if s == -1:
                for shift_idx in range(state.shift_types):
                    model.Add(state.work[n, d, shift_idx] == 0)
            else:
                model.Add(state.work[n, d, s] == 0)


def previous_schedule_rules(model, state: ScheduleState):
    """
    Adds constraints to the model based on the previous schedule (if any).
    
    The previous schedule is a pandas DataFrame with nurse names as index and day
    dates as columns. Each entry in the DataFrame is a string that can be one of

    - a known shift label (AM, PM, Night, ...) to fix exactly that shift
    - a label in NO_WORK_LABELS (EL, MC, ...) to block all shifts
    - blank string or NaN to skip
    - any other string (garbage text) to raise an error (or skip, if desired)

    The constraints are added to the model based on the contents of the previous
    schedule. The model is not modified if the previous schedule is None or empty.
    """
    prev = state.previous_schedule
    if prev is None or prev.empty:
        return
    
    for col in sorted(prev.columns):
        # col should be a date
        col_date = normalise_date(col)
        day_idx = (col_date - state.start_date).days

        for nurse in prev.index:
            raw = prev.at[nurse, col]
            if pd.isna(raw):
                continue   # no data → skip
            label = str(raw).strip().upper()
            if not label:
                continue   # blank string → skip

            # If it's a known shift label, fix exactly that shift:
            if label in state.shift_str_to_idx:
                s_fixed = state.shift_str_to_idx[label]
                for s in range(state.shift_types):
                    model.Add(
                        state.work[nurse, day_idx, s] == (1 if s == s_fixed else 0)
                    )
            else:
                # If it's “EL”, “MC”, etc., block all shifts
                if label in NO_WORK_LABELS:
                    for s in range(state.shift_types):
                        model.Add(state.work[nurse, day_idx, s] == 0)
                # Otherwise (garbage text), you might warn or skip
                else:
                    # Option A: raise an error
                    raise ValueError(f"Unknown entry in previous schedule '{label}' for {nurse} on {col}")
                    # Option B: skip
                    continue
