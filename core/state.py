from dataclasses import dataclass
from ortools.sat.python import cp_model
from typing import Any, Dict, List, Set, Tuple

@dataclass
class ScheduleState:
    # model inputs
    work: Dict[Tuple[str,int,int], cp_model.IntVar]
    nurse_names: List[str]
    senior_names: Set[str]
    shift_str_to_idx: Dict[str,int]
    fixed_assignments: Dict[Tuple[str,int], str]
    mc_sets: Dict[str, Set[int]]
    al_sets: Dict[str, Set[int]]
    el_sets: Dict[str, Set[int]]
    weekend_pairs: List[Tuple[int,int]]
    prefs_by_nurse: Dict[str, Dict[int,int]]

    # model params
    num_days: int
    shift_types: int
    shift_durations: List[int]
    min_nurses_per_shift: int
    min_seniors_per_shift: int
    max_weekly_hours: int
    preferred_weekly_hours: int
    pref_weekly_hours_hard: bool
    min_acceptable_weekly_hours: int
    activate_am_cov: bool
    am_coverage_min_percent: int
    am_coverage_min_hard: bool
    am_coverage_relax_step: int
    am_senior_min_percent: int
    am_senior_min_hard: bool
    am_senior_relax_step: int
    weekend_rest: bool
    back_to_back_shift: bool
    use_sliding_window: bool

    # collections to fill
    hard_rules: Dict[str, Any]
    high_priority_penalty: List[Any]
    low_priority_penalty: List[Any]
