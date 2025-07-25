import pandas as pd
from datetime import date as dt_date
from typing import Optional, Dict, Tuple
from config.paths import LOG_PATH
import logging
from utils.constants import *       # import all constants
from utils.validate import *
from utils.shift_utils import *
from exceptions.custom_errors import *
from scheduler.setup import setup_model
from core.state import ScheduleState
from core.constraint_manager import ConstraintManager
from scheduler.rules import *
from scheduler.runner import solve_schedule

logging.basicConfig(
    filename=LOG_PATH,
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

# == Build Schedule Model ==
def build_schedule_model(profiles_df: pd.DataFrame,
                         preferences_df: pd.DataFrame,
                         training_shifts_df: pd.DataFrame,
                         start_date: pd.Timestamp | dt_date,
                         num_days: int,
                         shift_durations: List[int] = SHIFT_DURATIONS,
                         min_nurses_per_shift: int = MIN_NURSES_PER_SHIFT,
                         min_seniors_per_shift: int = MIN_SENIORS_PER_SHIFT,
                         max_weekly_hours: int = MAX_WEEKLY_HOURS,
                         preferred_weekly_hours: int = PREFERRED_WEEKLY_HOURS,
                         pref_weekly_hours_hard: bool = False,
                         min_acceptable_weekly_hours: int = MIN_ACCEPTABLE_WEEKLY_HOURS,
                         activate_am_cov: bool = False,
                         am_coverage_min_percent: int = AM_COVERAGE_MIN_PERCENT,
                         am_coverage_min_hard: bool = False,
                         am_coverage_relax_step: int = AM_COVERAGE_RELAX_STEP,
                         am_senior_min_percent: int = AM_SENIOR_MIN_PERCENT,
                         am_senior_min_hard: bool = False,
                         am_senior_relax_step: int = AM_SENIOR_RELAX_STEP,
                         weekend_rest: bool = True,
                         back_to_back_shift: bool = False,
                         use_sliding_window: bool = False,
                         shift_balance: bool = False,
                         fixed_assignments: Optional[Dict[Tuple[str,int], str]] = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Builds a nurse schedule satisfying hard constraints and optimizing soft preferences.
    Returns a schedule DataFrame, a summary DataFrame, and a violations dictionary.
    """
    # === Validate inputs ===
    validate_data(profiles_df, preferences_df, "profiles", "preferences", False)
    validate_data(profiles_df, training_shifts_df, "profiles", "training shifts", False)

    # === Model setup ===
    logger.info("📋 Building model...")
    model, nurse_names, og_nurse_names, senior_names, shift_str_to_idx, date_start, \
    hard_rules, shift_preferences, prefs_by_nurse, training_shifts, training_by_nurse, fixed_assignments, mc_sets, \
    al_sets, el_sets, days_with_el, weekend_pairs, shift_types, work = setup_model(
        profiles_df, preferences_df, training_shifts_df, start_date, num_days, SHIFT_LABELS, NO_WORK_LABELS, fixed_assignments
    )

    state = ScheduleState(
        work=work,
        nurse_names=nurse_names,
        senior_names=senior_names,
        shift_str_to_idx=shift_str_to_idx,
        fixed_assignments=fixed_assignments,
        mc_sets=mc_sets,
        al_sets=al_sets,
        el_sets=el_sets,
        weekend_pairs=weekend_pairs,
        prefs_by_nurse=prefs_by_nurse,
        training_by_nurse=training_by_nurse,
        num_days=num_days,
        shift_types=shift_types,
        shift_durations=shift_durations,
        start_date=date_start,
        min_nurses_per_shift=min_nurses_per_shift,
        min_seniors_per_shift=min_seniors_per_shift,
        max_weekly_hours=max_weekly_hours,
        preferred_weekly_hours=preferred_weekly_hours,
        pref_weekly_hours_hard=pref_weekly_hours_hard,
        min_acceptable_weekly_hours=min_acceptable_weekly_hours,
        activate_am_cov=activate_am_cov,
        am_coverage_min_percent=am_coverage_min_percent,
        am_coverage_min_hard=am_coverage_min_hard,
        am_coverage_relax_step=am_coverage_relax_step,
        am_senior_min_percent=am_senior_min_percent,
        am_senior_min_hard=am_senior_min_hard,
        am_senior_relax_step=am_senior_relax_step,
        weekend_rest=weekend_rest,
        back_to_back_shift=back_to_back_shift,
        use_sliding_window=use_sliding_window,
        shift_balance=shift_balance,
        hard_rules=hard_rules,
        days_with_el=days_with_el,
        total_satisfied={},
        high_priority_penalty=[],
        low_priority_penalty=[]
    )

    cm = ConstraintManager(model, state)
    # Fixed rules
    cm.add_rule(handle_fixed_assignments)
    cm.add_rule(leave_rules)
    cm.add_rule(training_shift_rules)

    # High priority rules
    cm.add_rule(shifts_per_day_rule)
    cm.add_rule(weekly_working_hours_rules)
    cm.add_rule(min_staffing_per_shift_rule)
    cm.add_rule(weekend_rest_rule)
    cm.add_rule(no_back_to_back_shift_rule)
    cm.add_rule(am_coverage_rule)
    cm.add_rule(am_senior_staffing_lvl_rule)

    # Low priority rules
    # cm.add_rule(preference_rule)
    cm.add_rule(preference_rule_ts)
    cm.add_rule(fairness_gap_rule)
    cm.add_rule(shift_balance_rule)

    cm.apply_all()  # Apply all rules

    schedule_df, summary_df, violations, metrics = solve_schedule(model, state, og_nurse_names)
    return schedule_df, summary_df, violations, metrics
