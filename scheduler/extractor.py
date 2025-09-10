import pandas as pd
from core.state import ScheduleState
from .solver import SolverResult
from datetime import timedelta
from utils.constants import NO_WORK_LABELS, DAYS_PER_WEEK
import statistics
from typing import List
import logging
from utils.nurse_utils import (
    get_nurse_name_by_id,
)

logger = logging.getLogger(__name__)


def get_total_prefs_met(state: ScheduleState, result: SolverResult) -> int:
    """Return the total number of preferences met in the solution."""
    total_prefs_met = 0
    for n in state.nurse_names:
        for d in range(state.num_days):
            picked = [
                s for s in range(state.shift_types) if result.cached_values[(n, d, s)]
            ]
            raw_pref = state.prefs_by_nurse[n].get(d)
            # unpack the preference tuple if it exists
            pref = raw_pref[0] if isinstance(raw_pref, tuple) else raw_pref
            if pref is not None and len(picked) == 1 and pref in picked:
                total_prefs_met += 1

    return total_prefs_met


def extract_schedule_and_summary(
    state: ScheduleState, result: SolverResult, og_nurse_order: List[str]
):
    """
    Extract a schedule and summary from a solver result.

    Args:
        state (ScheduleState): The state of the scheduling problem.
        result (SolverResult): The result of the solver.
        og_nurse_order (list[str]): The original order of the nurse names.

    Returns:
        tuple: A tuple of (schedule_df, summary_df, violations, metrics)
    """
    dates = [state.start_date + timedelta(days=i) for i in range(state.num_days)]
    headers = [d.strftime("%a %Y-%m-%d") for d in dates]
    num_weeks = (state.num_days + 6) // 7

    schedule = {}
    summary = []
    violations = {
        "Double Shifts": [] if state.fixed_assignments else [],
        "Low Hours Nurses": [] if not state.pref_weekly_hours_hard else [],
    }
    metrics = {}
    has_prefs = bool(any(state.prefs_by_nurse.values()))
    if has_prefs:
        metrics = {
            "Preference Met": 0,
            "Preference Unmet": [],
            "Fairness Gap": (
                result.fairness_gap if result.fairness_gap is not None else "N/A"
            ),
        }

    for n in state.nurse_names:
        row = []
        minutes_per_week = [0] * num_weeks
        shift_counts = [0, 0, 0, 0]  # AM, PM, Night, REST
        training_counts = [0, 0, 0, 0]  # AM, PM, Night, FULL
        double_shift_days = []
        prefs_met = 0
        prefs_unmet = []

        leaves_for_nurse = state.leaves_by_nurse.get(str(n).lower(), {})

        for d in range(state.num_days):
            picked = [
                s
                for s in range(state.shift_types)
                if result.cached_values.get((n, d, s), 0) > 0
            ]
            leave_entry = leaves_for_nurse.get(d)

            # -------------------------
            # Normalize output format
            # -------------------------
            if isinstance(leave_entry, dict):
                shift = leave_entry  # AL/MC/EL object kept intact

            elif d in state.training_by_nurse.get(n, {}):
                # Training leave
                shift = {"id": None, "type": "LEAVE", "name": "TR"}

            else:
                if len(picked) == 0:
                    shift = {"id": None, "type": "REST", "name": "REST"}
                    shift_counts[3] += 1
                elif len(picked) == 1:
                    s = picked[0]
                    shift_obj = state.shifts[s]  # Shifts object
                    shift = {
                        "id": str(shift_obj.id),
                        "type": "SHIFT",
                        "name": getattr(shift_obj, "name", str(shift_obj)),
                    }
                    shift_counts[s] += 1
                else:
                    shift = [
                        {
                            "id": str(state.shifts[s].id),
                            "type": "SHIFT",
                            "name": getattr(
                                state.shifts[s], "name", str(state.shifts[s])
                            ),
                        }
                        for s in sorted(picked)
                    ]
                    for s in picked:
                        shift_counts[s] += 1

                    double_shift_days.append(dates[d].strftime("%a %Y-%m-%d"))

            row.append(shift)

            # ---- unchanged: weekly hours calc ----
            week_idx = d // DAYS_PER_WEEK
            for p in picked:
                minutes_per_week[week_idx] += int(state.shift_durations[p])
                shift_counts[p] += 1

            raw = state.prefs_by_nurse[n].get(d)

            if raw is not None:
                if (n, d) in state.fixed_assignments and state.fixed_assignments[
                    (n, d)
                ].upper() in NO_WORK_LABELS:
                    continue

                idx = raw[0] if isinstance(raw, tuple) else raw
                if len(picked) == 1 and picked[0] == idx:
                    prefs_met += 1
                    metrics["Preference Met"] += 1
                else:
                    prefs_unmet.append(
                        f"{dates[d].strftime('%a %Y-%m-%d')} (wanted {state.shifts[idx].name})"
                    )

        # ---- summary row ----
        summary_row = {
            "Nurse": n,
            "AL": sum(
                1
                for d, t in leaves_for_nurse.items()
                if isinstance(t, dict)
                and t.get("name") == "AL"
                and 0 <= d < state.num_days
            ),
            "MC": sum(
                1
                for d, t in leaves_for_nurse.items()
                if isinstance(t, dict)
                and t.get("name") == "MC"
                and 0 <= d < state.num_days
            ),
            "EL": sum(
                1
                for d, t in leaves_for_nurse.items()
                if isinstance(t, dict)
                and t.get("name") == "EL"
                and 0 <= d < state.num_days
            ),
            "AM (Training)": training_counts[0],
            "PM (Training)": training_counts[1],
            "Night (Training)": training_counts[2],
            "TR (Full Day Training)": training_counts[3],
            "Rest": shift_counts[3],
            "AM": shift_counts[0],
            "PM": shift_counts[1],
            "Night": shift_counts[2],
            "Double Shifts": len(double_shift_days),
        }

        # ---- weekly AL credit hours ----
        if not state.pref_weekly_hours_hard:
            preferred_weekly_minutes = state.preferred_weekly_hours * 60
            avg_minutes = statistics.mean(state.shift_durations)
            for w in range(num_weeks):
                days = range(
                    w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days)
                )
                if len(days) < DAYS_PER_WEEK:
                    continue

                al_this_week = sum(
                    1
                    for d in days
                    if leaves_for_nurse.get(d, {}).get("name") == "AL"  # type: ignore
                )

                credit_hrs = round(
                    (minutes_per_week[w] / 60) + ((al_this_week * avg_minutes) / 60), 1
                )
                actual_hrs = round(minutes_per_week[w] / 60, 1)

                summary_row[f"Hours_Week{w+1}_Real"] = actual_hrs
                summary_row[f"Hours_Week{w+1}_InclAL"] = credit_hrs

        summary_row.update(
            {
                "Prefs_Met": prefs_met,
                "Prefs_Unmet": len(prefs_unmet),
                "Unmet_Details": "; ".join(prefs_unmet),
            }
        )
        schedule[n] = row
        summary.append(summary_row)

        if double_shift_days:
            violations["Double Shifts"].append(
                f"{get_nurse_name_by_id(state.profiles_df,n)}: {'; '.join(double_shift_days)}"
            )

        if prefs_unmet:
            metrics["Preference Unmet"].append(
                f"{get_nurse_name_by_id(state.profiles_df,n)}: {'; '.join(prefs_unmet)}"
            )

    # unchanged: hard rule check
    for rule_name, rule in state.hard_rules.items():
        try:
            if result.solver.BooleanValue(rule.flag):
                violations[rule_name] = rule.message.strip()
        except Exception as e:
            logger.warning(f"⚠️ Unable to check hard rule '{rule_name}': {e}")

    schedule_df = pd.DataFrame.from_dict(
        schedule, orient="index", columns=headers
    ).reindex(og_nurse_order)
    summary_df = (
        pd.DataFrame(summary).set_index("Nurse").reindex(og_nurse_order).reset_index()
    )
    return schedule_df, summary_df, violations, metrics
