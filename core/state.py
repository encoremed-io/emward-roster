from dataclasses import dataclass
from ortools.sat.python import cp_model
from typing import Any, Dict, List, Set, Tuple, Optional
from datetime import date
import pandas as pd

@dataclass
class ScheduleState:
    """
    A dataclass to hold all the state relevant to creating and solving a nurse
    scheduling problem.
    """

    # model inputs
    work: Dict[Tuple[str,int,int], cp_model.IntVar]
    """A dictionary with keys `(nurse_name, day, shift_type)` and values a
    boolean variable indicating if the nurse is assigned to that shift.
    """
    nurse_names: List[str]
    """A list of all nurse names."""
    senior_names: Set[str]
    """A set of all senior nurse names."""
    shift_str_to_idx: Dict[str,int]
    """A dictionary mapping shift strings (e.g. 'AM', 'PM', 'Night') to their
    corresponding integer values.
    """
    previous_schedule: pd.DataFrame
    """A pandas DataFrame of the previous schedule."""
    fixed_assignments: Dict[Tuple[str,int], str]
    """A dictionary mapping `(nurse_name, day)` tuples to shift strings.
    """
    mc_sets: Dict[str, Set[int]]
    """A dictionary mapping each nurse to a set of days when they are not
    available to work due to Medical Leave.
    """
    al_sets: Dict[str, Set[int]]
    """A dictionary mapping each nurse to a set of days when they are not
    available to work due to Annual Leave.
    """
    el_sets: Dict[str, Set[int]]
    """A dictionary mapping each nurse to a set of days when they are not
    available to work due to Emergency Leave.
    """
    weekend_pairs: List[Tuple[int,int]]
    """A list of tuples of days (Sunday, Monday) indicating weekends."""
    prefs_by_nurse: Dict[str, Dict[int, Tuple[int, Any]]]
    """A dictionary mapping each nurse to a dictionary of their preferred
    shift types with timestamps.
    """
    training_by_nurse: Dict[str, Dict[int, int]]
    """A dictionary mapping each nurse to a dictionary of their training
    requirements.
    """

    # model params
    num_days: int
    """The number of days in the scheduling period."""
    prev_days: int
    """The number of days in the previous schedule."""
    shift_types: int
    """The number of shift types."""
    shift_durations: List[int]
    """A list of integers representing the duration of each shift type."""
    start_date: date
    """The date of the first day of the scheduling period."""
    min_nurses_per_shift: int
    """The minimum number of nurses required per shift."""
    min_seniors_per_shift: int
    """The minimum number of senior nurses required per shift."""
    max_weekly_hours: int
    """The maximum number of hours a nurse can work in a week."""
    preferred_weekly_hours: int
    """The preferred number of hours a nurse should work in a week."""
    pref_weekly_hours_hard: bool
    """A boolean indicating if the preferred weekly hours is a hard constraint."""
    min_acceptable_weekly_hours: int
    """The minimum acceptable number of hours a nurse should work in a week."""
    min_weekly_rest: int
    """The minimum rest days per week."""
    activate_am_cov: bool
    """A boolean indicating if the morning coverage constraint is active."""
    am_coverage_min_percent: int
    """The minimum percentage of shifts that must be covered by senior nurses in
    the morning.
    """
    am_coverage_min_hard: bool
    """A boolean indicating if the morning coverage constraint is a hard
    constraint.
    """
    am_coverage_relax_step: int
    """The step size for relaxing the morning coverage constraint."""
    am_senior_min_percent: int
    """The minimum percentage of shifts that must be covered by senior nurses in
    the morning."""
    am_senior_min_hard: bool
    """A boolean indicating if the morning coverage constraint is a hard
    constraint."""
    am_senior_relax_step: int
    """The step size for relaxing the morning coverage constraint."""
    weekend_rest: bool
    """A boolean indicating if weekend rest is enforced."""
    back_to_back_shift: bool
    """A boolean indicating if back-to-back shifts are allowed."""
    use_sliding_window: bool
    """A boolean indicating if a sliding window should be used to find the best
    schedule.
    """
    shift_balance: bool
    """A boolean indicating if the shift balance constraint is active."""
    pref_miss_penalty: int
    """The penalty for missing preferences."""
    fairness_gap_penalty: int
    """The penalty for the fairness gap of preferences met."""
    fairness_gap_threshold: int
    """The threshold for penalsing the fairness gap of preferences met."""
    shift_imbalance_penalty: int
    """The penalty for the shift distribution imbalance."""
    shift_imbalance_threshold: int
    """The threshold for penalsing the shift distribution imbalance."""

    # collections to fill
    hard_rules: Dict[str, Any]
    """A dictionary to store the hard rules of the problem."""
    days_with_el: Set[int]
    """A set of days with EL declarations."""
    total_satisfied: Dict[str, cp_model.IntVar]
    """A dictionary of total satisfied variables for each nurse."""
    high_priority_penalty: List[Any]
    """A list of high priority penalty terms."""
    low_priority_penalty: List[Any]
    """A list of low priority penalty terms."""
    gap_pct: Optional[cp_model.IntVar] = None
    """A variable to store the gap percentage from the fairness rule."""
