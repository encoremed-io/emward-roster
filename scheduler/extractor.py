import pandas as pd
from core.state import ScheduleState
from .solver import SolverResult
from datetime import timedelta
from utils.constants import NO_WORK_LABELS, DAYS_PER_WEEK
import statistics
from typing import List
import logging

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


# do not delete
# def extract_schedule_and_summary(
#     state: ScheduleState, result: SolverResult, og_nurse_order: List[str]
# ):
#     """
#     Extract a schedule and summary from a solver result.

#     Args:
#         state (ScheduleState): The state of the scheduling problem.
#         result (SolverResult): The result of the solver.
#         og_nurse_order (list[str]): The original order of the nurse names.

#     Returns:
#         tuple: A tuple of (schedule_df, summary_df, violations, metrics)
#             schedule_df (pd.DataFrame): A DataFrame containing the extracted schedule.
#             summary_df (pd.DataFrame): A DataFrame containing the extracted summary.
#             violations (dict): A dictionary containing the soft constraint violations.
#             metrics (dict): A dictionary containing the preference satisfaction and fairness metrics.
#     """
#     dates = [state.start_date + timedelta(days=i) for i in range(state.num_days)]
#     headers = [d.strftime("%a %Y-%m-%d") for d in dates]
#     num_weeks = (state.num_days + 6) // 7

#     schedule = {}
#     summary = []
#     violations = {
#         "Double Shifts": [] if state.fixed_assignments else [],
#         "Low Hours Nurses": [] if not state.pref_weekly_hours_hard else [],
#         # "Low AM Days": (
#         #     [] if state.activate_am_cov and not state.am_coverage_min_hard else []
#         # ),
#         # "Low Senior AM Days": [] if not state.am_senior_min_hard else [],
#     }
#     metrics = {}
#     has_prefs = bool(
#         any(state.prefs_by_nurse.values())
#     )  # only have metrics if there are preferences
#     if has_prefs:
#         metrics = {
#             "Preference Met": 0,
#             "Preference Unmet": [],
#             "Fairness Gap": (
#                 result.fairness_gap if result.fairness_gap is not None else "N/A"
#             ),
#         }

#     for n in state.nurse_names:
#         row = []
#         minutes_per_week = [0] * num_weeks
#         shift_counts = [0, 0, 0, 0]  # AM, PM, Night, REST
#         training_counts = [0, 0, 0, 0]  # AM, PM, Night, FULL
#         double_shift_days = []
#         prefs_met = 0
#         prefs_unmet = []

#         for d in range(state.num_days):
#             # if (n, d, s) is assigned, return 1, else 0 (if not assigned for any reason)
#             picked = [
#                 s
#                 for s in range(state.shift_types)
#                 if result.cached_values.get((n, d, s), 0) > 0
#             ]
#             tr = state.training_by_nurse.get(n, {})  # training by nurse

#             if d in state.mc_sets[n]:
#                 shift = NO_WORK_LABELS[1]  # MC
#             elif d in state.al_sets[n]:
#                 shift = NO_WORK_LABELS[3]  # AL
#             elif (n, d) in state.fixed_assignments and state.fixed_assignments[
#                 (n, d)
#             ].strip().upper() == NO_WORK_LABELS[2]:
#                 shift = NO_WORK_LABELS[2]  # EL
#             else:
#                 if len(picked) == 2:
#                     print(
#                         f"🟡 DOUBLE SHIFT: {n} on {dates[d]} → {[state.shifts[p] for p in picked]}"
#                     )
#                     double_shift_days.append(dates[d].strftime("%a %Y-%m-%d"))

#                 if len(picked) == 0:
#                     shift = ["REST"]
#                     shift_counts[3] += 1
#                 else:
#                     shift = [state.shifts[s] for s in sorted(picked)]

#             row.append(shift)

#             week_idx = d // DAYS_PER_WEEK
#             for p in picked:
#                 minutes_per_week[week_idx] += int(state.shift_durations[p])
#                 shift_counts[p] += 1

#             raw = state.prefs_by_nurse[n].get(d)
#             if raw is not None:
#                 if (n, d) in state.fixed_assignments and state.fixed_assignments[
#                     (n, d)
#                 ].upper() in NO_WORK_LABELS:
#                     continue

#                 idx = raw[0] if isinstance(raw, tuple) else raw
#                 if len(picked) == 1 and picked[0] == idx:
#                     prefs_met += 1
#                     metrics["Preference Met"] += 1
#                 else:
#                     prefs_unmet.append(
#                         f"{dates[d].strftime('%a %Y-%m-%d')} (wanted {state.shifts[idx]})"
#                     )

#         if not state.pref_weekly_hours_hard:
#             preferred_weekly_minutes = state.preferred_weekly_hours * 60
#             avg_minutes = statistics.mean(state.shift_durations)
#             for w in range(num_weeks):
#                 days = range(
#                     w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days)
#                 )
#                 if len(days) < DAYS_PER_WEEK:
#                     continue  # skip incomplete weeks
#                 mc_count_week = len(state.mc_sets[n] & set(days))
#                 el_count_week = len(state.el_sets[n] & set(days))
#                 al_count_week = len(state.al_sets[n] & set(days))
#                 eff_pref_minutes = max(
#                     0,
#                     preferred_weekly_minutes
#                     - (mc_count_week + el_count_week + al_count_week) * avg_minutes,
#                 )

#                 if minutes_per_week[w] < eff_pref_minutes:
#                     violations["Low Hours Nurses"].append(
#                         f"{n} Week {w+1}: {round(minutes_per_week[w] / 60, 1)}h; pref {round(eff_pref_minutes / 60, 1)}"
#                     )

#         if double_shift_days:
#             violations["Double Shifts"].append(f"{n}: {'; '.join(double_shift_days)}")

#         if prefs_unmet:
#             metrics["Preference Unmet"].append(f"{n}: {'; '.join(prefs_unmet)}")

#         schedule[n] = row
#         summary_row = {
#             "Nurse": n,
#             "AL": sum(1 for d in state.al_sets[n] if 0 <= d < state.num_days),
#             "MC": sum(1 for d in state.mc_sets[n] if 0 <= d < state.num_days),
#             "EL": sum(1 for d in state.el_sets[n] if 0 <= d < state.num_days),
#             "AM (Training)": training_counts[0],
#             "PM (Training)": training_counts[1],
#             "Night (Training)": training_counts[2],
#             "TR (Full Day Training)": training_counts[3],
#             "Rest": shift_counts[3],
#             "AM": shift_counts[0],
#             "PM": shift_counts[1],
#             "Night": shift_counts[2],
#             "Double Shifts": len(double_shift_days),
#         }
#         for w in range(num_weeks):
#             actual_hrs = round(minutes_per_week[w] / 60, 1)
#             days_this_week = set(
#                 range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days))
#             )
#             al_this_week = len(state.al_sets[n] & days_this_week)
#             credit_hrs = round(
#                 actual_hrs + ((al_this_week * avg_minutes) / 60), 1
#             )  # credit hours include AL
#             summary_row[f"Hours_Week{w+1}_Real"] = actual_hrs
#             summary_row[f"Hours_Week{w+1}_InclAL"] = credit_hrs
#         summary_row.update(
#             {
#                 "Prefs_Met": prefs_met,
#                 "Prefs_Unmet": len(prefs_unmet),
#                 "Unmet_Details": "; ".join(prefs_unmet),
#             }
#         )
#         summary.append(summary_row)

#     # if not state.am_coverage_min_hard and not state.am_senior_min_hard:
#     #     for d in range(state.num_days):
#     #         am_n = sum(result.cached_values[(n, d, 0)] for n in state.nurse_names)
#     #         total_n = sum(
#     #             result.cached_values[(n, d, s)]
#     #             for n in state.nurse_names
#     #             for s in range(state.shift_types)
#     #         )
#     #         am_snr = sum(result.cached_values[(n, d, 0)] for n in state.senior_names)

#     #         if (
#     #             state.activate_am_cov
#     #             and not state.am_coverage_min_hard
#     #             and total_n
#     #             and am_n / total_n < (state.am_coverage_min_percent / 100)
#     #         ):
#     #             violations["Low AM Days"].append(
#     #                 f"{dates[d].strftime('%a %Y-%m-%d')} ({am_n/total_n:.0%})"
#     #             )
#     #         if (
#     #             not state.am_senior_min_hard
#     #             and am_n
#     #             and am_snr / am_n < (state.am_senior_min_percent / 100)
#     #         ):
#     #             violations["Low Senior AM Days"].append(
#     #                 f"{dates[d].strftime('%a %Y-%m-%d')} (Seniors {am_snr/am_n:.0%})"
#     #             )

#     if violations:
#         logger.info("\n⚠️ Soft Constraint Violations Summary:")
#         for key, items in violations.items():
#             logger.info(
#                 f"🔸 {key}: {len(items) if isinstance(items, list) else items} cases"
#             )
#             if isinstance(items, list):
#                 for item in sorted(items):
#                     logger.info(f"   - {item}")

#     if has_prefs:
#         logger.info("\n📊 Preferences Satisfaction and Fairness Summary:")
#         total_unmet = sum(s["Prefs_Unmet"] for s in summary)
#         logger.info(f"🔸 Preference Met: {metrics['Preference Met']} preferences met")
#         logger.info(
#             f"🔸 Preference Unmet: {total_unmet} unmet preferences across {len(metrics['Preference Unmet'])} nurses"
#         )
#         logger.info(f"🔸 Fairness Gap: {metrics['Fairness Gap']}%")
#         for key, items in metrics.items():
#             # if not isinstance(items, list):
#             #     logger.info(f"🔸 {key}: {items}")
#             if isinstance(items, list):
#                 for item in sorted(items):
#                     logger.info(f"   - {item}")

#     # ✅ Append hard rule violations
#     for rule_name, rule in state.hard_rules.items():
#         try:
#             if result.solver.BooleanValue(rule.flag):
#                 violations[rule_name] = rule.message.strip()
#         except Exception as e:
#             logger.warning(f"⚠️ Unable to check hard rule '{rule_name}': {e}")

#     schedule_df = pd.DataFrame.from_dict(
#         schedule, orient="index", columns=headers
#     ).reindex(og_nurse_order)
#     summary_df = (
#         pd.DataFrame(summary).set_index("Nurse").reindex(og_nurse_order).reset_index()
#     )
#     return schedule_df, summary_df, violations, metrics


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
            schedule_df (pd.DataFrame): A DataFrame containing the extracted schedule.
            summary_df (pd.DataFrame): A DataFrame containing the extracted summary.
            violations (dict): A dictionary containing the soft constraint violations.
            metrics (dict): A dictionary containing the preference satisfaction and fairness metrics.
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

        leaves_for_nurse = state.leaves_by_nurse.get(str(n), {})

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
                # keep the full object
                shift = leave_entry

            elif d in state.training_by_nurse.get(n, {}):
                # Training leave
                shift = {"id": None, "type": "LEAVE", "name": "TR"}

            else:
                if len(picked) == 0:
                    shift = {"id": None, "type": "REST", "name": "REST"}
                    shift_counts[3] += 1
                elif len(picked) == 1:
                    s = picked[0]
                    shift_obj = state.shifts[s]  # this is a Shifts object
                    shift = {
                        "id": str(shift_obj.id),
                        "type": "SHIFT",
                        "name": getattr(
                            shift_obj, "name", str(shift_obj)
                        ),  # safe access
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

                    print(
                        f"DOUBLE SHIFT: {n} on {dates[d]} → {[state.shifts[p] for p in picked]}"
                    )
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
                        f"{dates[d].strftime('%a %Y-%m-%d')} (wanted {state.shifts[idx]})"
                    )

        # ---- unchanged: weekly violation check, summary row build ----
        if not state.pref_weekly_hours_hard:
            preferred_weekly_minutes = state.preferred_weekly_hours * 60
            avg_minutes = statistics.mean(state.shift_durations)
            for w in range(num_weeks):
                days = range(
                    w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days)
                )
                if len(days) < DAYS_PER_WEEK:
                    continue
                leave_types_week = []
                for d in days:
                    entry = state.leaves_by_nurse.get(str(n), {}).get(d)
                    if isinstance(entry, dict):
                        leave_types_week.append(entry.get("leavename"))
                    elif isinstance(entry, str):
                        leave_types_week.append(entry)
                    else:
                        leave_types_week.append(None)

                leave_count = sum(
                    1 for leave in leave_types_week if leave in {"MC", "EL", "AL"}
                )
                eff_pref_minutes = max(
                    0, preferred_weekly_minutes - leave_count * avg_minutes
                )

                if minutes_per_week[w] < eff_pref_minutes:
                    violations["Low Hours Nurses"].append(
                        f"{n} Week {w+1}: {round(minutes_per_week[w] / 60, 1)}h; pref {round(eff_pref_minutes / 60, 1)}"
                    )

        if double_shift_days:
            violations["Double Shifts"].append(f"{n}: {'; '.join(double_shift_days)}")

        if prefs_unmet:
            metrics["Preference Unmet"].append(f"{n}: {'; '.join(prefs_unmet)}")

        schedule[n] = row
        summary_row = {
            "Nurse": n,
            "AL": sum(
                1
                for d, t in leaves_for_nurse.items()
                if t == "AL" and 0 <= d < state.num_days
            ),
            "MC": sum(
                1
                for d, t in leaves_for_nurse.items()
                if t == "MC" and 0 <= d < state.num_days
            ),
            "EL": sum(
                1
                for d, t in leaves_for_nurse.items()
                if t == "EL" and 0 <= d < state.num_days
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
        for w in range(num_weeks):
            actual_hrs = round(minutes_per_week[w] / 60, 1)
            days_this_week = set(
                range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days))
            )
            al_this_week = sum(
                1 for d in days_this_week if leaves_for_nurse.get(d) == "AL"
            )
            credit_hrs = round(actual_hrs + ((al_this_week * avg_minutes) / 60), 1)
            summary_row[f"Hours_Week{w+1}_Real"] = actual_hrs
            summary_row[f"Hours_Week{w+1}_InclAL"] = credit_hrs
        summary_row.update(
            {
                "Prefs_Met": prefs_met,
                "Prefs_Unmet": len(prefs_unmet),
                "Unmet_Details": "; ".join(prefs_unmet),
            }
        )
        summary.append(summary_row)

    # ✅ unchanged: violations & metrics logs, hard rule check, final df build
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
