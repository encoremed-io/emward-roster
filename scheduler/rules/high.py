from core import state
from core.state import ScheduleState
from utils.constants import *
import statistics
import logging
import math
from utils.shift_utils import make_shift_index
import re
import sys

"""
This module contains the high priority rules for the nurse scheduling problem.
"""


# High priority rules are those that must be satisfied for a feasible solution to exist.
def shifts_per_day_rule(model, state: ScheduleState):
    """Ensure that each nurse works 1 shift per day. If a nurse has an EL day, allow either 1 or 2 shifts."""
    for n in state.nurse_names:
        for d in range(-state.prev_days, state.num_days):
            if d not in state.el_sets[n]:
                # no EL here: enforce original rule
                model.AddAtMostOne(
                    state.work[n, d, s] for s in range(state.shift_types)
                )
            else:
                # EL day: allow either 1 or 2 shifts
                ts = model.NewBoolVar(f"two_shifts_{n}_{d}")

                if (
                    d in state.el_sets[n]
                ):  # if nurse has el on that day, explicitly make ts = false for the nurse
                    model.Add(ts == 0)

                # If ts==False → sum_s state.state.work ≤ 1
                model.Add(
                    sum(state.work[n, d, s] for s in range(state.shift_types)) <= 1
                ).OnlyEnforceIf(ts.Not())
                # If ts==True  → sum_s state.state.work ≤ 2
                model.Add(
                    sum(state.work[n, d, s] for s in range(state.shift_types)) <= 2
                ).OnlyEnforceIf(ts)

                # If double shift, apply penalty
                state.high_priority_penalty.append(ts * DOUBLE_SHIFT_PENALTY)


def weekly_working_hours_rules(model, state: ScheduleState):
    """Add weekly working hours rules."""
    max_weekly_hours_rule(model, state)
    pref_min_weekly_hours_rule(model, state)


def max_weekly_hours_rule(model, state: ScheduleState):
    """
    Enforce maximum weekly working hours for each nurse.

    This rule ensures that nurses do not exceed the maximum allowed weekly hours. It accounts for leave
    (MC, AL, EL days) by deducting average shift hours from the maximum. The rule can apply over a
    sliding 7-day window or at fixed weekly boundaries.

    If `state.use_sliding_window` is True, the rule checks every 7-day sliding window within the
    scheduling period. Otherwise, it checks each complete week of days.

    Maximum weekly hours can be reduced by the number of MC/AL/EL days.
    """
    # convert to minutes
    max_weekly_minutes = state.max_weekly_hours * 60
    avg_minutes = statistics.mean(state.shift_durations)

    for n in state.nurse_names:
        mc = state.mc_sets[n]
        al = state.al_sets[n]
        el = state.el_sets[n]
        num_weeks = (state.num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK

        # Precompute daily-hours expressions if using sliding window
        if state.use_sliding_window:
            minutes_by_day = [
                sum(
                    state.work[n, d, s] * int(state.shift_durations[s])
                    for s in range(state.shift_types)
                )
                for d in range(-state.prev_days, state.num_days)
            ]

            for d in range(DAYS_PER_WEEK - 1, state.num_days):
                # Maximum working hours every 7 day sliding window (exp: Day 0 to Day 6, then Day 1 to Day 7, etc.)
                window = range(d - (DAYS_PER_WEEK - 1), d + 1)
                window_minutes = sum(minutes_by_day[i] for i in window)
                leave_count = sum(1 for i in window if i in mc or i in al or i in el)
                adj = (
                    leave_count * avg_minutes
                )  # MC/AL/EL hours deducted from max/pref/min hours
                eff_max_minutes = max(0, max_weekly_minutes - adj)  # <= 42 - x
                model.Add(window_minutes <= eff_max_minutes).OnlyEnforceIf(
                    state.hard_rules["Max weekly hours"].flag
                )

        # Full-week minimum at each 7-day boundary (e.g. Day 6, then Day 13, etc.)
        else:
            for w in range(num_weeks):
                days = range(
                    w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days)
                )

                if len(days) < DAYS_PER_WEEK:
                    continue  # Skip incomplete weeks for extra days if not using sliding window

                weekly_minutes = sum(
                    state.work[n, d, s] * int(state.shift_durations[s])
                    for d in days
                    for s in range(state.shift_types)
                )

                leave_count = sum(1 for i in days if i in mc or i in al or i in el)
                adj = leave_count * avg_minutes
                eff_max_minutes = max(0, max_weekly_minutes - adj)

                if not state.use_sliding_window:
                    eff_max_minutes = max(0, max_weekly_minutes - adj)  # <= 42 - x
                    model.Add(weekly_minutes <= eff_max_minutes).OnlyEnforceIf(
                        state.hard_rules["Max weekly hours"].flag
                    )


def pref_min_weekly_hours_rule(model, state: ScheduleState):
    """
    Ensure that nurses work at least the minimum acceptable weekly hours per complete week.

    This constraint is soft if pref_weekly_hours_hard is False, and hard otherwise.

    If soft, a penalty is incurred for each week that the nurse is assigned less than the preferred weekly hours.
    The penalty is proportional to the number of hours below the preferred weekly hours.

    Preferred weekly hours and minimum acceptable weekly hours can be reduced by the number of MC/AL/EL days.
    """
    preferred_weekly_minutes = state.preferred_weekly_hours * 60
    min_acceptable_weekly_minutes = state.min_acceptable_weekly_hours * 60
    avg_minutes = statistics.mean(state.shift_durations)

    for n in state.nurse_names:
        mc = state.mc_sets[n]
        al = state.al_sets[n]
        el = state.el_sets[n]
        num_weeks = (state.num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK

        # Full-week minimum at each 7-day boundary (e.g. Day 6, then Day 13, etc.)
        for w in range(num_weeks):
            days = range(
                w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days)
            )

            if len(days) < DAYS_PER_WEEK:
                continue  # Skip incomplete weeks for extra days

            weekly_minutes = sum(
                state.work[n, d, s] * int(state.shift_durations[s])
                for d in days
                for s in range(state.shift_types)
            )

            leave_count = sum(1 for i in days if i in mc or i in al or i in el)
            adj = leave_count * avg_minutes
            eff_min_minutes = max(0, min_acceptable_weekly_minutes - adj)  # >= 30 - x
            eff_pref_minutes = max(0, preferred_weekly_minutes - adj)  # >= 40 - x

            if state.pref_weekly_hours_hard:
                model.Add(weekly_minutes >= eff_pref_minutes).OnlyEnforceIf(
                    state.hard_rules["Pref weekly hours"].flag
                )
            else:
                if eff_pref_minutes > eff_min_minutes:
                    flag = model.NewBoolVar(f"pref_{n}_w{w}")
                    model.Add(weekly_minutes >= eff_pref_minutes).OnlyEnforceIf(flag)
                    model.Add(weekly_minutes < eff_pref_minutes).OnlyEnforceIf(
                        flag.Not()
                    )
                    model.Add(weekly_minutes >= eff_min_minutes).OnlyEnforceIf(
                        state.hard_rules["Min weekly hours"].flag
                    )

                    state.high_priority_penalty.append(flag.Not() * PREF_HOURS_PENALTY)


def min_staffing_per_shift_rule(model, state: ScheduleState):
    print("senior names\n", state.senior_names)
    """Ensure that each shift has a minimum number of nurses and seniors."""
    for d in range(-state.prev_days, state.num_days):
        for s in range(state.shift_types):
            model.Add(
                sum(state.work[n, d, s] for n in state.nurse_names)
                >= state.min_nurses_per_shift
            ).OnlyEnforceIf(state.hard_rules["Min nurses"].flag)
            model.Add(
                sum(state.work[n, d, s] for n in state.senior_names)
                >= state.min_seniors_per_shift
            ).OnlyEnforceIf(state.hard_rules["Min seniors"].flag)


def min_rest_per_week_rule(model, state: ScheduleState):
    """Ensure that nurses have a minimum number of rest days per week."""
    num_full_weeks = state.num_days // DAYS_PER_WEEK

    for n in state.nurse_names:
        for w in range(num_full_weeks):
            days = range(w * DAYS_PER_WEEK, (w + 1) * DAYS_PER_WEEK)
            rest_days = []
            for d in days:
                rest = model.NewBoolVar(f"rest_{n}_{d}")
                model.Add(
                    sum(state.work[n, d, s] for s in range(state.shift_types)) == 0
                ).OnlyEnforceIf(rest)
                model.Add(
                    sum(state.work[n, d, s] for s in range(state.shift_types)) >= 1
                ).OnlyEnforceIf(rest.Not())
                rest_days.append(rest)

            model.Add(sum(rest_days) >= state.min_weekly_rest).OnlyEnforceIf(
                state.hard_rules["Min weekly rest"].flag
            )


def weekend_rest_rule(model, state: ScheduleState):
    """Ensure that nurses who work on a weekend must rest on the corresponding days the following weekend only if state.weekend_rest is True."""
    if state.weekend_rest:
        for n in state.nurse_names:
            for d1, d2 in state.weekend_pairs:
                # skip any pair outside the built horizon
                if (n, d1, 0) not in state.work or (n, d2, 0) not in state.work:
                    continue
                model.Add(
                    sum(state.work[n, d1, s] for s in range(state.shift_types))
                    + sum(state.work[n, d2, s] for s in range(state.shift_types))
                    <= 1
                )


def no_back_to_back_shift_rule(model, state: ScheduleState):
    """Ensure that nurses do not work back-to-back shifts on the same day if state.back_to_back_shift is True."""
    if not state.back_to_back_shift:
        for n in state.nurse_names:
            for d in range(-state.prev_days, state.num_days):
                # No back-to-back shifts on the same day (double shifts)
                model.Add(state.work[n, d, 0] + state.work[n, d, 1] <= 1).OnlyEnforceIf(
                    state.hard_rules["No b2b"].flag
                )  # AM + PM on same day
                model.Add(state.work[n, d, 1] + state.work[n, d, 2] <= 1).OnlyEnforceIf(
                    state.hard_rules["No b2b"].flag
                )  # PM + Night on same day
                if d > 0 or (d == 0 and state.prev_days > 0):
                    # Night shift on day d cannot be followed by AM shift on day d+1
                    model.AddImplication(
                        state.work[n, d - 1, 2], state.work[n, d, 0].Not()
                    ).OnlyEnforceIf(state.hard_rules["No b2b"].flag)


# def am_coverage_rule(model, state: ScheduleState):
#     """
#     Apply AM coverage rules based on user configuration.

#     This function applies constraints to ensure a minimum percentage of nurses work the AM shift each day.
#     If `am_coverage_min_hard` is True, it strictly enforces the minimum AM coverage percentage. If False,
#     it employs a series of relaxed levels, penalizing deviations from the desired coverage incrementally.

#     Parameters:
#     - model: The constraint model to which rules are added.
#     - state: The current schedule state, containing configuration and data for nurse scheduling.

#     Behavior:
#     - Hard Constraint: Enforces the minimum AM coverage percentage across all shifts if `am_coverage_min_hard` is enabled.
#     - Soft Constraint: Uses relaxed coverage levels and applies penalties for failing to meet each level.
#     - Fallback: Ensures AM shifts outnumber PM and Night shifts if all relaxed levels fail.

#     This rule supports gradual relaxation using `am_coverage_relax_step` and applies corresponding penalties
#     from `AM_COVERAGE_PENALTY` for each level violation.
#     """

#     if state.activate_am_cov:
#         if not state.am_coverage_min_hard:
#             levels = list(
#                 range(state.am_coverage_min_percent, 34, -state.am_coverage_relax_step)
#             )  # [65, 60, 55] if step = 5
#             penalties = [
#                 (i + 1) * AM_COVERAGE_PENALTY for i in range(len(levels))
#             ]  # [5, 10, 15]

#         for d in range(-state.prev_days, state.num_days):
#             total_shifts = sum(
#                 state.work[n, d, s]
#                 for n in state.nurse_names
#                 for s in range(state.shift_types)
#             )
#             am_shifts = sum(state.work[n, d, 0] for n in state.nurse_names)
#             pm_shifts = sum(state.work[n, d, 1] for n in state.nurse_names)
#             night_shifts = sum(state.work[n, d, 2] for n in state.nurse_names)

#             if state.am_coverage_min_hard:
#                 model.Add(
#                     am_shifts * 100 >= state.am_coverage_min_percent * total_shifts
#                 ).OnlyEnforceIf(state.hard_rules["AM cov min"].flag)

#             else:
#                 flags = [
#                     model.NewBoolVar(f"day_{d}_am_min_{lvl}") for lvl in levels
#                 ]  # flags for each level
#                 fallback = model.NewBoolVar(
#                     f"day_{d}_all_levels_failed"
#                 )  # fallback flag

#                 for flag, lvl in zip(flags, levels):
#                     model.Add(am_shifts * 100 >= lvl * total_shifts).OnlyEnforceIf(flag)
#                     model.Add(am_shifts * 100 < lvl * total_shifts).OnlyEnforceIf(
#                         flag.Not()
#                     )

#                 # Hard fallback: all levels failed
#                 model.AddBoolAnd([flag.Not() for flag in flags]).OnlyEnforceIf(fallback)
#                 model.AddBoolOr(flags).OnlyEnforceIf(fallback.Not())
#                 # Enforce AM > PM and AM > Night if all levels fail (hard constraint)
#                 model.Add(am_shifts > pm_shifts).OnlyEnforceIf(fallback).OnlyEnforceIf(
#                     state.hard_rules["AM cov majority"].flag
#                 )
#                 model.Add(am_shifts > night_shifts).OnlyEnforceIf(
#                     fallback
#                 ).OnlyEnforceIf(state.hard_rules["AM cov majority"].flag)

#                 # Penalties for AM ratio violations (only failed levels penalised)
#                 for i, lvl in enumerate(levels):
#                     penalise = model.NewBoolVar(f"day_{d}_penalise_lvl_{lvl}_am")
#                     model.Add(penalise == flags[i].Not())
#                     state.high_priority_penalty.append(penalise * penalties[i])


# def am_senior_staffing_lvl_rule(model, state: ScheduleState):
#     """
#     Enforce the minimum percentage of senior nurses on AM shifts. If the flag is set to "hard", the constraint is enforced strictly.
#     If not, the constraint is relaxed by the specified step value, with a penalty incurred for each level that is not met.

#     The levels are specified as a list of percentages, e.g. [65, 60, 55] if the step is 5. The penalty for each level is the same as the step value, e.g. [5, 10, 15] in the above example.

#     The constraint is enforced for each day separately, and the penalty is only incurred if the constraint is not met for that day.

#     """
#     if not state.am_senior_min_hard:
#         levels = list(
#             range(state.am_senior_min_percent, 50, -state.am_senior_relax_step)
#         )  # [65, 60, 55] if step = 5
#         penalties = [
#             (i + 1) * AM_SENIOR_PENALTY for i in range(len(levels))
#         ]  # [5, 10, 15]

#     for d in range(-state.prev_days, state.num_days):
#         am_shifts = sum(state.work[n, d, 0] for n in state.nurse_names)
#         am_seniors = sum(state.work[n, d, 0] for n in state.senior_names)
#         am_juniors = am_shifts - am_seniors  # number of junior nurses on AM shift

#         if state.am_senior_min_hard:
#             model.Add(
#                 am_seniors * 100 >= state.am_senior_min_percent * am_shifts
#             ).OnlyEnforceIf(state.hard_rules["AM snr min"].flag)

#         else:
#             flags = [
#                 model.NewBoolVar(f"day_{d}_senior_min_{lvl}") for lvl in levels
#             ]  # flags for each level
#             fallback = model.NewBoolVar(f"day_{d}_all_levels_failed")  # fallback flag

#             for flag, lvl in zip(flags, levels):
#                 model.Add(am_seniors * 100 >= lvl * am_shifts).OnlyEnforceIf(flag)
#                 model.Add(am_seniors * 100 < lvl * am_shifts).OnlyEnforceIf(flag.Not())

#             # Hard fallback: all levels failed
#             model.AddBoolAnd([flag.Not() for flag in flags]).OnlyEnforceIf(fallback)
#             model.AddBoolOr(flags).OnlyEnforceIf(fallback.Not())
#             # Enforce AM seniors >= AM junior if all levels fail (hard constraint)
#             model.Add(am_seniors >= am_juniors).OnlyEnforceIf(fallback).OnlyEnforceIf(
#                 state.hard_rules["AM snr majority"].flag
#             )

#             # Penalties for senior ratio violations (only failed levels penalised)
#             for i, lvl in enumerate(levels):
#                 penalise = model.NewBoolVar(f"day_{d}_penalise_lvl_{lvl}_snr")
#                 model.Add(penalise == flags[i].Not())
#                 state.high_priority_penalty.append(penalise * penalties[i])


# staff allocation rule
def staff_allocation_rule(model, state: ScheduleState):
    """Enforce senior staff percentage allocation, with fallback if main target is not met."""

    alloc = state.staff_allocation
    if not alloc or not alloc.seniorStaffAllocation:
        logging.info("Staff allocation not enabled or not required.")
        return

    base_percent = alloc.seniorStaffPercentage
    fallback_step = alloc.seniorStaffAllocationRefinementValue or 0
    fallback_percent = max(0, base_percent - fallback_step)

    for d in range(-state.prev_days, state.num_days):
        for s in range(state.shift_types):
            total_seniors = sum(state.work[n, d, s] for n in state.senior_names)

            # Compute required seniors for base and fallback percent
            base_required = max(
                state.min_seniors_per_shift,
                math.ceil(state.min_nurses_per_shift * base_percent / 100),
            )
            fallback_required = max(
                state.min_seniors_per_shift,
                math.ceil(state.min_nurses_per_shift * fallback_percent / 100),
            )

            # Create flags
            base_ok = model.NewBoolVar(f"base_senior_ok_d{d}_s{s}")
            fallback_ok = model.NewBoolVar(f"fallback_senior_ok_d{d}_s{s}")

            model.Add(total_seniors >= base_required).OnlyEnforceIf(base_ok)
            model.Add(total_seniors < base_required).OnlyEnforceIf(base_ok.Not())

            model.Add(total_seniors >= fallback_required).OnlyEnforceIf(fallback_ok)
            model.Add(total_seniors < fallback_required).OnlyEnforceIf(
                fallback_ok.Not()
            )

            # Apply at least fallback as hard rule
            model.AddBoolOr([base_ok, fallback_ok]).OnlyEnforceIf(
                state.hard_rules["staff_allocation_not_satisfied"].flag
            )

            # Optional: apply soft penalty if base failed but fallback passed
            if fallback_step > 0:
                penalty_flag = model.NewBoolVar(f"penalty_fallback_used_d{d}_s{s}")
                model.AddBoolAnd([base_ok.Not(), fallback_ok]).OnlyEnforceIf(
                    penalty_flag
                )
                state.high_priority_penalty.append(penalty_flag * fallback_step)


# shift details rule (ICU)
# def shift_details_rule(model, state: ScheduleState):

#     if not state.shift_details:
#         return

#     for rule in state.shift_details:
#         shift_label = rule.shiftType
#         max_shifts = rule.maxWorkingShift
#         required_rest = rule.restDayEligible

#         # Convert shift label to index
#         if isinstance(shift_label, str):
#             shift_type = state.shift_str_to_idx.get(shift_label.upper())
#             if shift_type is None:
#                 raise ValueError(f"Unknown shiftType '{shift_label}' in shift_details")
#         else:
#             shift_type = shift_label  # Already an int

#         for nurse in state.nurse_names:
#             for start_day in range(state.num_days - max_shifts - required_rest + 1):
#                 # Working streak variables
#                 work_segment = [
#                     state.work[nurse, start_day + i, shift_type]
#                     for i in range(max_shifts)
#                 ]
#                 worked_full_streak = model.NewBoolVar(
#                     f"{nurse}_worked_{shift_label}_{start_day}"
#                 )
#                 model.Add(sum(work_segment) == max_shifts).OnlyEnforceIf(
#                     worked_full_streak
#                 )
#                 model.Add(sum(work_segment) != max_shifts).OnlyEnforceIf(
#                     worked_full_streak.Not()
#                 )

#                 # Rest day variables after the streak
#                 rest_vars = []
#                 for i in range(required_rest):
#                     rest_day = start_day + max_shifts + i
#                     is_rest = model.NewBoolVar(f"{nurse}_rest_{rest_day}")
#                     model.Add(
#                         sum(
#                             state.work[nurse, rest_day, s]
#                             for s in range(state.shift_types)
#                         )
#                         == 0
#                     ).OnlyEnforceIf(is_rest)
#                     model.Add(
#                         sum(
#                             state.work[nurse, rest_day, s]
#                             for s in range(state.shift_types)
#                         )
#                         != 0
#                     ).OnlyEnforceIf(is_rest.Not())
#                     rest_vars.append(is_rest)

#                 # Enforce: if overworked → must rest
#                 model.AddBoolAnd(rest_vars).OnlyEnforceIf(worked_full_streak)


def shift_details_rule(model, state: ScheduleState):
    if not state.shift_details:
        return

    # Ensure id→index map exists
    if not getattr(state, "shift_str_to_idx", None):
        raise ValueError("shift_str_to_idx is not initialized on state.")

    for rule in state.shift_details:
        raw = rule.shiftType
        if not isinstance(raw, str):
            raise TypeError("shiftType must be a string ID.")
        shift_id = raw.strip()

        if shift_id not in state.shift_str_to_idx:
            raise ValueError(
                f"Unknown shiftType id '{shift_id}'. Known IDs: {list(state.shift_str_to_idx.keys())}"
            )

        shift_type = state.shift_str_to_idx[shift_id]  # ← integer index for solver
        max_shifts = int(rule.maxWorkingShift)
        required_rest = int(rule.restDayEligible)

        horizon = state.num_days - max_shifts - required_rest
        if horizon < 0:
            continue

        # sanitize for var names if you log/inspect them
        safe_lbl = re.sub(r"[^A-Za-z0-9]+", "_", str(shift_id))

        for nurse in state.nurse_names:
            for start_day in range(horizon + 1):
                # Working streak variables
                work_segment = [
                    state.work[nurse, start_day + i, shift_type]
                    for i in range(max_shifts)
                ]
                worked_full_streak = model.NewBoolVar(
                    f"{nurse}_worked_{safe_lbl}_{start_day}"
                )
                model.Add(sum(work_segment) == max_shifts).OnlyEnforceIf(
                    worked_full_streak
                )
                model.Add(sum(work_segment) != max_shifts).OnlyEnforceIf(
                    worked_full_streak.Not()
                )

                # Rest day variables after the streak
                rest_vars = []
                for i in range(required_rest):
                    rest_day = start_day + max_shifts + i
                    is_rest = model.NewBoolVar(f"{nurse}_rest_{rest_day}")
                    model.Add(
                        sum(
                            state.work[nurse, rest_day, s]
                            for s in range(state.shift_types)
                        )
                        == 0
                    ).OnlyEnforceIf(is_rest)
                    model.Add(
                        sum(
                            state.work[nurse, rest_day, s]
                            for s in range(state.shift_types)
                        )
                        != 0
                    ).OnlyEnforceIf(is_rest.Not())
                    rest_vars.append(is_rest)

                # Enforce: if overworked → must rest
                model.AddBoolAnd(rest_vars).OnlyEnforceIf(worked_full_streak)


# nurses who can work double shifts
# def double_shift_rule(model, state: ScheduleState):
#     """Only restrict double shifts for nurses who are not eligible."""

#     for nurse_name in state.nurse_names:
#         for d in range(-state.prev_days, state.num_days):
#             total_shifts = sum(
#                 state.work[nurse_name, d, s] for s in range(state.shift_types)
#             )

#             if not state.allow_double_shift:
#                 # Enforce max 1 shift per day when double shifts are not allowed
#                 model.Add(total_shifts <= 1)
#                 continue  # Skip extra double-shift logic

#             if nurse_name in state.double_shift_nurses:
#                 model.Add(total_shifts <= 2)
#             else:
#                 model.Add(total_shifts <= 1)

#             # Apply: if 2 shifts today → no AM tomorrow
#             if d + 1 < state.num_days:
#                 double_shift_today = model.NewBoolVar(f"{nurse_name}_double_{d}")
#                 model.Add(total_shifts >= 2).OnlyEnforceIf(double_shift_today)
#                 model.Add(total_shifts < 2).OnlyEnforceIf(double_shift_today.Not())

#                 # Block AM shift next day if double shift today
#                 model.Add(state.work[nurse_name, d + 1, 0] == 0).OnlyEnforceIf(
#                     double_shift_today
#                 )


def double_shift_rule(model, state: ScheduleState):
    """Only restrict double shifts for nurses who are not eligible."""

    for nurse_name in state.nurse_names:
        for d in range(-state.prev_days, state.num_days):
            total_shifts = sum(
                state.work[nurse_name, d, s] for s in range(state.shift_types)
            )

            if not state.allow_double_shift:
                # Enforce max 1 shift per day when double shifts are not allowed
                model.Add(total_shifts <= 1)
                continue  # Skip extra double-shift logic

            if nurse_name in state.double_shift_nurses:
                model.Add(total_shifts <= 2)
            else:
                model.Add(total_shifts <= 1)

            # Apply: if 2 shifts today → no AM tomorrow
            if d + 1 < state.num_days:
                double_shift_today = model.NewBoolVar(f"{nurse_name}_double_{d}")
                model.Add(total_shifts >= 2).OnlyEnforceIf(double_shift_today)
                model.Add(total_shifts < 2).OnlyEnforceIf(double_shift_today.Not())

                # Block AM shift next day if double shift today
                model.Add(state.work[nurse_name, d + 1, 0] == 0).OnlyEnforceIf(
                    double_shift_today
                )
