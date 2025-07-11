import pandas as pd
from core.state import ScheduleState
from .solver import SolverResult
from datetime import timedelta
from utils.constants import NO_WORK_LABELS, SHIFT_LABELS, DAYS_PER_WEEK
import statistics
import logging

logger = logging.getLogger(__name__)

def get_total_prefs_met(state: ScheduleState, result: SolverResult) -> int:
    total_prefs_met = 0
    for n in state.nurse_names:
        for d in range(state.num_days):
            picked = [s for s in range(state.shift_types) if result.cached_values[(n, d, s)]]
            pref = state.prefs_by_nurse[n].get(d)
            if pref is not None and len(picked) == 1 and pref in picked:
                total_prefs_met += 1

    return total_prefs_met


def extract_schedule_and_summary(state: ScheduleState, result: SolverResult, og_nurse_order: list[str]):
    dates = [state.start_date + timedelta(days=i) for i in range(state.num_days)]
    headers = [d.strftime('%a %Y-%m-%d') for d in dates]
    num_weeks = (state.num_days + 6) // 7

    schedule = {}
    summary = []
    violations = {}
    if state.fixed_assignments:
        violations["Double Shifts"] = []
    if not state.pref_weekly_hours_hard:
        violations["Low Hours Nurses"] = []
    if state.activate_am_cov and not state.am_coverage_min_hard:
        violations["Low AM Days"] = []
    if not state.am_senior_min_hard:
        violations["Low Senior AM Days"] = []
    metrics = {}
    has_prefs = bool(state.low_priority_penalty)
    if has_prefs:
        metrics = {
            "Preference Unmet": [],
            "Fairness Gap": result.fairness_gap
        }

    for n in state.nurse_names:
        row = []
        minutes_per_week = [0] * num_weeks
        shift_counts = [0, 0, 0, 0]  # AM, PM, Night, REST
        double_shift_days = []
        prefs_met = 0
        prefs_unmet = []

        for d in range(state.num_days):
            # if (n, d, s) is assigned, return 1, else 0 (if not assigned for any reason)
            picked = [s for s in range(state.shift_types) if result.cached_values.get((n, d, s), 0) > 0]

            if d in state.mc_sets[n]:
                shift = NO_WORK_LABELS[1]      # MC
            elif d in state.al_sets[n]:
                shift = NO_WORK_LABELS[3]      # AL
            elif (n, d) in state.fixed_assignments and state.fixed_assignments[(n, d)].strip().upper() == NO_WORK_LABELS[2]:
                shift = NO_WORK_LABELS[2]      # EL
            else:
                if len(picked) == 2:
                    double_shift_days.append(dates[d].strftime('%a %Y-%m-%d'))
                
                match(len(picked)):
                    case 0:
                        shift = NO_WORK_LABELS[0]   # REST
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
                minutes_per_week[week_idx] += int(state.shift_durations[p])
                shift_counts[p] += 1

            pref = state.prefs_by_nurse[n].get(d)
            if pref is not None:
                if (n, d) in state.fixed_assignments and state.fixed_assignments[(n, d)].upper() in NO_WORK_LABELS:
                    continue

                if len(picked) == 1 and picked[0] == pref:
                    prefs_met += 1
                else:
                    prefs_unmet.append(f"{dates[d].strftime('%a %Y-%m-%d')} (wanted {SHIFT_LABELS[pref]})")

        if not state.pref_weekly_hours_hard:
            preferred_weekly_minutes = state.preferred_weekly_hours * 60
            avg_minutes = statistics.mean(state.shift_durations)
            for w in range(num_weeks):
                days = range(w * DAYS_PER_WEEK, min((w + 1) * DAYS_PER_WEEK, state.num_days))
                if len(days) < DAYS_PER_WEEK:
                    continue  # skip incomplete weeks
                mc_count_week = len(state.mc_sets[n] & set(days))
                el_count_week = len(state.el_sets[n] & set(days))
                al_count_week = len(state.al_sets[n] & set(days))
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
            "AL":       len(state.al_sets[n]),
            "MC":       len(state.mc_sets[n]),
            "EL":       len(state.el_sets[n]),
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

    if not state.am_coverage_min_hard and not state.am_senior_min_hard:
        for d in range(state.num_days):
            am_n = sum(result.cached_values[(n, d, 0)] for n in state.nurse_names)
            total_n = sum(result.cached_values[(n, d, s)] for n in state.nurse_names for s in range(state.shift_types))
            am_snr = sum(result.cached_values[(n, d, 0)] for n in state.senior_names)

            if state.activate_am_cov and not state.am_coverage_min_hard and total_n and am_n / total_n < (state.am_coverage_min_percent / 100):
                violations["Low AM Days"].append(f"{dates[d].strftime('%a %Y-%m-%d')} ({am_n/total_n:.0%})")
            if not state.am_senior_min_hard and am_n and am_snr / am_n < (state.am_senior_min_percent / 100):
                violations["Low Senior AM Days"].append(f"{dates[d].strftime('%a %Y-%m-%d')} (Seniors {am_snr/am_n:.0%})")

    if violations:
        logger.info("\n⚠️ Soft Constraint Violations Summary:")
        for key, items in violations.items():
            logger.info(f"🔸 {key}: {len(items) if isinstance(items, list) else items} cases")
            if isinstance(items, list):
                for item in sorted(items):
                    logger.info(f"   - {item}")

    if has_prefs:
        logger.info("\n📊 Preferences Satisfaction and Fairness Summary:")
        total_unmet = sum(s["Prefs_Unmet"] for s in summary)
        logger.info(f"🔸 Preference Unmet: {total_unmet} unmet preferences across {len(metrics['Preference Unmet'])} nurses")
        logger.info(f"🔸 Fairness Gap: {metrics['Fairness Gap']}%")
        for key, items in metrics.items():
            if isinstance(items, list):
                for item in sorted(items):
                    logger.info(f"   - {item}")

    schedule_df = pd.DataFrame.from_dict(schedule, orient='index', columns=headers).reindex(og_nurse_order)
    summary_df = pd.DataFrame(summary).set_index("Nurse").reindex(og_nurse_order).reset_index()
    return schedule_df, summary_df, violations, metrics
