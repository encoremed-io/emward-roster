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
    Computes global Preference Met / Unmet counts automatically.
    Dynamically handles any number of shift types from state.shifts.
    """
    from datetime import timedelta
    import pandas as pd
    import logging
    from utils.constants import NO_WORK_LABELS, DAYS_PER_WEEK
    from utils.nurse_utils import get_nurse_name_by_id

    logger = logging.getLogger(__name__)

    dates = [state.start_date + timedelta(days=i) for i in range(state.num_days)]
    headers = [d.strftime("%a %Y-%m-%d") for d in dates]
    num_weeks = (state.num_days + 6) // 7

    # --- compute total possible preferences (upper bound) ---
    total_possible_prefs = sum(
        len(state.prefs_by_nurse.get(n, {})) for n in state.nurse_names
    )

    schedule = {}
    summary = []
    violations = {
        "Double Shifts": [] if state.fixed_assignments else [],
        "Low Hours Nurses": [] if not state.pref_weekly_hours_hard else [],
    }
    metrics = {}
    has_prefs = total_possible_prefs > 0
    if has_prefs:
        metrics = {
            "Preference Met": 0,
            "Preference Unmet": [],  # detailed list
            "Preference Unmet Count": 0,  # ✅ new aggregate
            "Fairness Gap": (
                result.fairness_gap if result.fairness_gap is not None else "N/A"
            ),
        }

    # Build dynamic shift name map
    shift_names = [getattr(s, "name", str(s)) for s in state.shifts]

    for n in state.nurse_names:
        row = []
        minutes_per_week = [0] * num_weeks

        # ✅ dynamic shift counters
        shift_counts = {name: 0 for name in shift_names}
        shift_counts["REST"] = 0
        training_counts = {name: 0 for name in shift_names}
        training_counts["TR"] = 0

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

            # ---- Determine shift displayed ----
            if isinstance(leave_entry, dict):
                shift = leave_entry
            elif d in state.training_by_nurse.get(n, {}):
                shift = {"id": None, "type": "LEAVE", "name": "TR"}
                training_counts["TR"] += 1
            else:
                if len(picked) == 0:
                    shift = {"id": None, "type": "REST", "name": "REST"}
                    shift_counts["REST"] += 1
                elif len(picked) == 1:
                    s = picked[0]
                    shift_obj = state.shifts[s]
                    shift_name = getattr(shift_obj, "name", str(shift_obj))
                    shift = {
                        "id": str(shift_obj.id),
                        "type": "SHIFT",
                        "name": shift_name,
                    }
                    shift_counts[shift_name] += 1
                else:
                    shift = []
                    for s in sorted(picked):
                        shift_obj = state.shifts[s]
                        shift_name = getattr(shift_obj, "name", str(shift_obj))
                        shift.append(
                            {
                                "id": str(shift_obj.id),
                                "type": "SHIFT",
                                "name": shift_name,
                            }
                        )
                        shift_counts[shift_name] += 1
                    double_shift_days.append(dates[d].strftime("%a %Y-%m-%d"))

            row.append(shift)

            # ---- Weekly hours ----
            week_idx = d // DAYS_PER_WEEK
            for p in picked:
                minutes_per_week[week_idx] += int(state.shift_durations[p])
                shift_name = getattr(state.shifts[p], "name", str(state.shifts[p]))
                shift_counts[shift_name] += 1  # ✅ dynamic instead of index

            # ---- Preference check ----
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

        # ---- Build summary row dynamically ----
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
            "Double Shifts": len(double_shift_days),
            "Prefs_Met": prefs_met,
            "Prefs_Unmet": len(prefs_unmet),
            "Unmet_Details": "; ".join(prefs_unmet),
        }

        # add training counts (temporary hidden)
        # for tname, count in training_counts.items():
        #     summary_row[f"{tname} (Training)"] = count

        # add dynamic shift counts
        for sname, count in shift_counts.items():
            summary_row[sname] = count

        schedule[n] = row
        summary.append(summary_row)

        if double_shift_days:
            violations["Double Shifts"].append(
                f"{get_nurse_name_by_id(state.profiles_df, n)}: {'; '.join(double_shift_days)}"
            )
        if prefs_unmet:
            metrics["Preference Unmet"].append(
                f"{get_nurse_name_by_id(state.profiles_df, n)}: {'; '.join(prefs_unmet)}"
            )

    # ✅ compute global unmet count automatically
    if has_prefs:
        metrics["Preference Unmet Count"] = (
            total_possible_prefs - metrics["Preference Met"]
        )

    # ---- Hard rule check ----
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
